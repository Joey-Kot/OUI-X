import asyncio
import json
import logging
import random
import threading
import time
import uuid
from typing import Any, Optional

import redis

from open_webui.retrieval.vector.main import (
    GetResult,
    SearchResult,
    VectorDBBase,
    VectorItem,
)

log = logging.getLogger(__name__)


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"), default=str)


class ChromaWriteCoordinator:
    def __init__(
        self,
        redis_client,
        key_prefix: str,
        stream_key: str = "chroma:write:stream",
        result_key_prefix: str = "chroma:write:result",
        wait_timeout_seconds: int = 120,
    ):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.stream_key = f"{key_prefix}:{stream_key}"
        self.result_key_prefix = f"{key_prefix}:{result_key_prefix}"
        self.group_name = f"{key_prefix}:chroma:write:group"
        self.leader_key = f"{key_prefix}:chroma:worker:leader"
        self.leader_meta_key = f"{key_prefix}:chroma:worker:leader_meta"
        self.wait_timeout_seconds = wait_timeout_seconds

    def _result_key(self, task_id: str) -> str:
        return f"{self.result_key_prefix}:{task_id}"

    def _safe_sync_call(self, name: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {"error": type(e).__name__, "detail": str(e), "op": name}

    def collect_queue_snapshot_sync(self, result_key: str) -> dict:
        snapshot = {
            "stream_exists": self._safe_sync_call(
                "exists", self.redis.exists, self.stream_key
            ),
            "stream_len": self._safe_sync_call("xlen", self.redis.xlen, self.stream_key),
            "group_info": self._safe_sync_call(
                "xinfo_groups", self.redis.xinfo_groups, self.stream_key
            ),
            "consumer_info": self._safe_sync_call(
                "xinfo_consumers",
                self.redis.xinfo_consumers,
                self.stream_key,
                self.group_name,
            ),
            "pending_info": self._safe_sync_call(
                "xpending", self.redis.xpending, self.stream_key, self.group_name
            ),
            "pending_sample": self._safe_sync_call(
                "xpending_range",
                self.redis.xpending_range,
                self.stream_key,
                self.group_name,
                "-",
                "+",
                10,
            ),
            "leader_value": self._safe_sync_call("get", self.redis.get, self.leader_key),
            "leader_ttl": self._safe_sync_call("ttl", self.redis.ttl, self.leader_key),
            "leader_meta": self._safe_sync_call(
                "get", self.redis.get, self.leader_meta_key
            ),
            "result_queue_len": self._safe_sync_call("llen", self.redis.llen, result_key),
        }
        return snapshot

    def enqueue_and_wait(self, op: str, collection_name: Optional[str], payload: dict):
        task_id = str(uuid.uuid4())
        enqueue_ts = time.monotonic()
        result_key = self._result_key(task_id)
        task = {
            "task_id": task_id,
            "op": op,
            "collection_name": collection_name,
            "payload": payload,
            "created_at": int(time.time()),
        }

        self.redis.xadd(self.stream_key, {"task": _json_dumps(task)})

        log.info(
            "Queued chroma write task_id=%s op=%s collection=%s stream=%s result_key=%s wait_timeout=%ss",
            task_id,
            op,
            collection_name,
            self.stream_key,
            result_key,
            self.wait_timeout_seconds,
        )

        result = self.redis.blpop(result_key, timeout=self.wait_timeout_seconds)
        if not result:
            elapsed_ms = int((time.monotonic() - enqueue_ts) * 1000)
            snapshot = self.collect_queue_snapshot_sync(result_key)

            in_event_loop = False
            try:
                asyncio.get_running_loop()
                in_event_loop = True
            except RuntimeError:
                in_event_loop = False

            log.error(
                "Timed out waiting queued chroma write result task_id=%s op=%s collection=%s elapsed_ms=%s in_event_loop=%s thread=%s snapshot=%s",
                task_id,
                op,
                collection_name,
                elapsed_ms,
                in_event_loop,
                threading.current_thread().name,
                snapshot,
            )
            raise TimeoutError(
                f"Timed out waiting for queued Chroma write result: op={op}, task_id={task_id}"
            )

        _, data = result
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        response = json.loads(data)
        if response.get("ok"):
            elapsed_ms = int((time.monotonic() - enqueue_ts) * 1000)
            log.info(
                "Received queued chroma write result task_id=%s op=%s collection=%s elapsed_ms=%s",
                task_id,
                op,
                collection_name,
                elapsed_ms,
            )
            return response.get("data")

        error = response.get("error") or "unknown queued Chroma write failure"
        elapsed_ms = int((time.monotonic() - enqueue_ts) * 1000)
        log.warning(
            "Queued chroma write task failed task_id=%s op=%s collection=%s elapsed_ms=%s error=%s",
            task_id,
            op,
            collection_name,
            elapsed_ms,
            error,
        )
        raise RuntimeError(error)


class ChromaWriteWorker:
    def __init__(
        self,
        inner_client,
        redis_client,
        key_prefix: str,
        *,
        stream_key: str = "chroma:write:stream",
        group_name: str = "chroma:write:group",
        result_key_prefix: str = "chroma:write:result",
        consumer_name: Optional[str] = None,
    ):
        self.inner = inner_client
        self.redis = redis_client
        self.stream_key = f"{key_prefix}:{stream_key}"
        self.group_name = f"{key_prefix}:{group_name}"
        self.result_key_prefix = f"{key_prefix}:{result_key_prefix}"
        self.lock_prefix = f"{key_prefix}:chroma:lock:collection"
        self.leader_key = f"{key_prefix}:chroma:worker:leader"
        self.leader_meta_key = f"{key_prefix}:chroma:worker:leader_meta"
        self.consumer_name = consumer_name or f"worker-{uuid.uuid4()}"

        self.max_batch = 100
        self.batch_window_ms = 50
        self.read_block_ms = 250
        self.result_ttl_seconds = 300

        self.leader_lock_token: Optional[str] = None
        self.leader_lock_ttl_seconds = 90
        self.leader_heartbeat_interval_seconds = 3
        self._leader_alive = False
        self._leader_heartbeat_task: Optional[asyncio.Task] = None

        self.collection_lock_ttl_seconds = 60
        self.collection_lock_wait_seconds = 3
        self.pending_min_idle_ms = 5_000
        self.nogroup_log_interval_seconds = 10
        self._last_nogroup_log_ts = 0.0
        self.renew_fallback_log_interval_seconds = 60
        self._last_renew_fallback_log_ts = 0.0
        self.reacquire_backoff_min_seconds = 0.2
        self.reacquire_backoff_max_seconds = 0.5

    @staticmethod
    def _error_class(exc: Exception) -> str:
        if isinstance(exc, redis.exceptions.ConnectionError):
            return "connection"
        if isinstance(exc, redis.exceptions.TimeoutError):
            return "connection"
        if isinstance(exc, redis.exceptions.ResponseError) and "NOGROUP" in str(exc):
            return "nogroup"
        if isinstance(exc, redis.exceptions.ResponseError):
            return "other_response_error"
        return "other"

    async def run(self):
        await self._ensure_group()

        while True:
            try:
                acquired = await self._acquire_leader()
                if not acquired:
                    await asyncio.sleep(1)
                    continue

                self._leader_alive = True
                self._leader_heartbeat_task = asyncio.create_task(
                    self._leader_heartbeat_loop()
                )
                await self._run_as_leader()
                await asyncio.sleep(
                    random.uniform(
                        self.reacquire_backoff_min_seconds,
                        self.reacquire_backoff_max_seconds,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Chroma write worker loop failed")
                await asyncio.sleep(1)
            finally:
                if self._leader_heartbeat_task is not None:
                    self._leader_heartbeat_task.cancel()
                    try:
                        await self._leader_heartbeat_task
                    except asyncio.CancelledError:
                        pass
                    self._leader_heartbeat_task = None

                self._leader_alive = False
                await self._release_leader()

    async def _run_as_leader(self):
        log.info(
            "Chroma write worker entered leader loop consumer=%s stream=%s group=%s heartbeat_interval=%ss",
            self.consumer_name,
            self.stream_key,
            self.group_name,
            self.leader_heartbeat_interval_seconds,
        )

        await self._recover_pending_messages()

        while True:
            if not self._leader_alive:
                snapshot = await self._collect_runtime_snapshot()
                log.warning(
                    "Leader heartbeat marked worker unhealthy consumer=%s stream=%s group=%s snapshot=%s",
                    self.consumer_name,
                    self.stream_key,
                    self.group_name,
                    snapshot,
                )
                return

            await self._recover_pending_messages()
            messages = await self._read_batch()
            if not messages:
                continue

            for batch in self._coalesce_messages(messages):
                await self._process_batch(batch)

    async def _ensure_group(self):
        try:
            await self.redis.xgroup_create(
                name=self.stream_key,
                groupname=self.group_name,
                id="0",
                mkstream=True,
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def _write_leader_meta(self):
        if not self.leader_lock_token:
            return

        payload = _json_dumps(
            {
                "consumer": self.consumer_name,
                "token": self.leader_lock_token,
                "heartbeat_ts": int(time.time()),
            }
        )
        await self.redis.set(
            self.leader_meta_key,
            payload,
            ex=self.leader_lock_ttl_seconds,
        )

    async def _leader_heartbeat_loop(self):
        while self._leader_alive:
            await asyncio.sleep(self.leader_heartbeat_interval_seconds)
            if not self._leader_alive:
                return

            try:
                renewed = await self._renew_leader()
            except Exception as e:
                self._leader_alive = False
                log.error(
                    "Leader heartbeat failed error_class=renew_failed redis_exception_type=%s consumer=%s",
                    type(e).__name__,
                    self.consumer_name,
                )
                return

            if not renewed:
                self._leader_alive = False
                snapshot = await self._collect_runtime_snapshot()
                log.warning(
                    "Lost chroma redis worker leader lock consumer=%s stream=%s group=%s snapshot=%s",
                    self.consumer_name,
                    self.stream_key,
                    self.group_name,
                    snapshot,
                )
                return

            await self._write_leader_meta()

    async def _renew_leader_fallback(self, lua_error: Exception) -> bool:
        current = await self.redis.get(self.leader_key)
        if current is None:
            reason = "key_missing"
            renewed = False
        elif current != self.leader_lock_token:
            reason = "token_mismatch"
            renewed = False
        else:
            expire_result = await self.redis.expire(
                self.leader_key, self.leader_lock_ttl_seconds
            )
            renewed = bool(expire_result)
            reason = "expire_failed" if not renewed else None

        now = time.monotonic()
        if now - self._last_renew_fallback_log_ts >= self.renew_fallback_log_interval_seconds:
            if renewed:
                log.warning(
                    "Leader lock renew fallback used error_class=lua_incompatible renew_path=fallback consumer=%s leader_key=%s token=%s lua_error=%s",
                    self.consumer_name,
                    self.leader_key,
                    self.leader_lock_token[:8],
                    str(lua_error),
                )
            else:
                log.warning(
                    "Leader lock renew fallback failed error_class=renew_failed renew_path=fallback consumer=%s leader_key=%s token=%s fallback_reason=%s lua_error=%s",
                    self.consumer_name,
                    self.leader_key,
                    self.leader_lock_token[:8],
                    reason,
                    str(lua_error),
                )
            self._last_renew_fallback_log_ts = now

        return renewed

    @staticmethod
    def _is_nogroup_error(exc: Exception) -> bool:
        return isinstance(exc, redis.exceptions.ResponseError) and "NOGROUP" in str(exc)

    async def _recover_stream_group_if_missing(self):
        await self._ensure_group()

        now = time.monotonic()
        if now - self._last_nogroup_log_ts >= self.nogroup_log_interval_seconds:
            log.warning(
                "Detected missing Chroma Redis stream/group error_class=nogroup command=consumer_group recover_action=ensure_group stream=%s group=%s",
                self.stream_key,
                self.group_name,
            )
            self._last_nogroup_log_ts = now

    async def _acquire_leader(self) -> bool:
        token = str(uuid.uuid4())
        ok = await self.redis.set(
            self.leader_key,
            token,
            nx=True,
            ex=self.leader_lock_ttl_seconds,
        )
        if ok:
            self.leader_lock_token = token
            await self._write_leader_meta()
            log.info(
                "Acquired chroma worker leader lock consumer=%s token=%s ttl=%ss",
                self.consumer_name,
                token[:8],
                self.leader_lock_ttl_seconds,
            )
            return True
        log.debug(
            "Leader lock currently held by another worker consumer=%s leader_key=%s",
            self.consumer_name,
            self.leader_key,
        )
        return False

    async def _renew_leader(self) -> bool:
        if not self.leader_lock_token:
            log.warning(
                "Leader lock renew skipped: no local token consumer=%s",
                self.consumer_name,
            )
            return False

        script = (
            "if redis.call('GET', KEYS[1]) == ARGV[1] then "
            "return redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2])) "
            "else return 0 end"
        )
        try:
            renewed = await self.redis.eval(
                script,
                1,
                self.leader_key,
                self.leader_lock_token,
                str(self.leader_lock_ttl_seconds),
            )
        except redis.exceptions.ResponseError as e:
            renewed = await self._renew_leader_fallback(e)
            if renewed:
                log.debug(
                    "Leader lock renewed renew_path=fallback consumer=%s token=%s ttl=%ss",
                    self.consumer_name,
                    self.leader_lock_token[:8],
                    self.leader_lock_ttl_seconds,
                )
                return True

            log.warning(
                "Leader lock renew failed error_class=renew_failed renew_path=fallback consumer=%s leader_key=%s token=%s",
                self.consumer_name,
                self.leader_key,
                self.leader_lock_token[:8],
            )
            return False
        if renewed:
            log.debug(
                "Leader lock renewed renew_path=lua consumer=%s token=%s ttl=%ss",
                self.consumer_name,
                self.leader_lock_token[:8],
                self.leader_lock_ttl_seconds,
            )
            return True

        current = await self.redis.get(self.leader_key)
        if current is None:
            reason = "key_missing"
        elif current != self.leader_lock_token:
            reason = "token_mismatch"
        else:
            reason = "expire_failed"

        log.warning(
            "Leader lock renew failed reason=%s consumer=%s leader_key=%s",
            reason,
            self.consumer_name,
            self.leader_key,
        )
        return False

    async def _release_leader(self):
        if not self.leader_lock_token:
            return

        current = await self.redis.get(self.leader_key)
        if current == self.leader_lock_token:
            await self.redis.delete(self.leader_key)
            await self.redis.delete(self.leader_meta_key)
            log.info(
                "Released chroma worker leader lock consumer=%s token=%s",
                self.consumer_name,
                self.leader_lock_token[:8],
            )
        else:
            log.debug(
                "Skip leader lock release consumer=%s token=%s reason=not_owner",
                self.consumer_name,
                self.leader_lock_token[:8],
            )

        self.leader_lock_token = None

    async def _collect_runtime_snapshot(self, result_key: Optional[str] = None) -> dict:
        try:
            snapshot = {
                "stream_exists": await self.redis.exists(self.stream_key),
                "stream_len": await self.redis.xlen(self.stream_key),
                "group_info": await self.redis.xinfo_groups(self.stream_key),
                "consumer_info": await self.redis.xinfo_consumers(
                    self.stream_key, self.group_name
                ),
                "pending_info": await self.redis.xpending(
                    self.stream_key, self.group_name
                ),
                "pending_sample": await self.redis.xpending_range(
                    self.stream_key,
                    self.group_name,
                    "-",
                    "+",
                    10,
                ),
                "leader_value": await self.redis.get(self.leader_key),
                "leader_ttl": await self.redis.ttl(self.leader_key),
                "leader_meta": await self.redis.get(self.leader_meta_key),
            }
            if result_key:
                snapshot["result_queue_len"] = await self.redis.llen(result_key)
            return snapshot
        except Exception as e:
            return {
                "error": type(e).__name__,
                "detail": str(e),
            }

    async def _recover_pending_messages(self):
        start_id = "0-0"
        while True:
            retry_delays = [0.2, 0.5, 1.0]
            retry_idx = 0
            while True:
                try:
                    response = await self.redis.xautoclaim(
                        name=self.stream_key,
                        groupname=self.group_name,
                        consumername=self.consumer_name,
                        min_idle_time=self.pending_min_idle_ms,
                        start_id=start_id,
                        count=self.max_batch,
                    )
                    break
                except Exception as e:
                    if self._is_nogroup_error(e):
                        log.warning(
                            "Failed xautoclaim error_class=nogroup redis_exception_type=%s command=xautoclaim recover_action=ensure_group",
                            type(e).__name__,
                        )
                        await self._recover_stream_group_if_missing()
                        return

                    if self._error_class(e) == "connection" and retry_idx < len(
                        retry_delays
                    ):
                        delay = retry_delays[retry_idx]
                        retry_idx += 1
                        log.warning(
                            "Failed xautoclaim error_class=connection redis_exception_type=%s command=xautoclaim recover_action=retry delay=%.1fs attempt=%s",
                            type(e).__name__,
                            delay,
                            retry_idx,
                        )
                        await asyncio.sleep(delay)
                        continue

                    log.error(
                        "Failed xautoclaim error_class=%s redis_exception_type=%s command=xautoclaim recover_action=none",
                        self._error_class(e),
                        type(e).__name__,
                    )
                    raise

            claimed = []
            if isinstance(response, (list, tuple)):
                start_id = response[0] if len(response) > 0 else start_id
                claimed = response[1] if len(response) > 1 else []

            if not claimed:
                return

            parsed = self._parse_messages(claimed)
            for batch in self._coalesce_messages(parsed):
                await self._process_batch(batch)

            if len(claimed) < self.max_batch:
                return

    async def _read_batch(self) -> list[dict]:
        try:
            records = await self.redis.xreadgroup(
                groupname=self.group_name,
                consumername=self.consumer_name,
                streams={self.stream_key: ">"},
                count=1,
                block=self.read_block_ms,
            )
        except Exception as e:
            if self._is_nogroup_error(e):
                log.warning(
                    "Failed xreadgroup error_class=nogroup redis_exception_type=%s command=xreadgroup recover_action=ensure_group",
                    type(e).__name__,
                )
                await self._recover_stream_group_if_missing()
                return []
            log.error(
                "Failed xreadgroup error_class=%s redis_exception_type=%s command=xreadgroup recover_action=none",
                self._error_class(e),
                type(e).__name__,
            )
            raise

        if not records:
            return []

        entry_count = sum(len(entries) for _stream, entries in records)
        first_id = records[0][1][0][0] if records and records[0][1] else None
        last_id = records[-1][1][-1][0] if records and records[-1][1] else None
        log.debug(
            "Read queued chroma messages consumer=%s count=%s first_id=%s last_id=%s",
            self.consumer_name,
            entry_count,
            first_id,
            last_id,
        )

        messages = self._parse_stream_records(records)

        deadline = time.monotonic() + (self.batch_window_ms / 1000)
        while len(messages) < self.max_batch and time.monotonic() < deadline:
            remaining = self.max_batch - len(messages)
            try:
                more = await self.redis.xreadgroup(
                    groupname=self.group_name,
                    consumername=self.consumer_name,
                    streams={self.stream_key: ">"},
                    count=remaining,
                    block=self.batch_window_ms,
                )
            except Exception as e:
                if self._is_nogroup_error(e):
                    log.warning(
                        "Failed xreadgroup(batch) error_class=nogroup redis_exception_type=%s command=xreadgroup recover_action=ensure_group",
                        type(e).__name__,
                    )
                    await self._recover_stream_group_if_missing()
                    return []
                log.error(
                    "Failed xreadgroup(batch) error_class=%s redis_exception_type=%s command=xreadgroup recover_action=none",
                    self._error_class(e),
                    type(e).__name__,
                )
                raise

            if not more:
                break
            messages.extend(self._parse_stream_records(more))

        return messages

    def _parse_stream_records(self, records) -> list[dict]:
        messages = []
        for _stream_name, entries in records:
            messages.extend(self._parse_messages(entries))
        return messages

    def _parse_messages(self, entries) -> list[dict]:
        messages = []
        for message_id, fields in entries:
            raw_task = fields.get("task")
            if isinstance(raw_task, bytes):
                raw_task = raw_task.decode("utf-8")

            try:
                task = json.loads(raw_task)
            except Exception as e:
                log.warning("Invalid chroma queued task payload id=%s err=%s", message_id, e)
                messages.append(
                    {
                        "message_id": message_id,
                        "task": {
                            "task_id": None,
                            "op": None,
                            "collection_name": None,
                            "payload": {},
                            "parse_error": str(e),
                        },
                    }
                )
                continue

            messages.append({"message_id": message_id, "task": task})
        return messages

    def _coalesce_messages(self, messages: list[dict]) -> list[dict]:
        batches: list[dict] = []
        merge_index: dict[tuple[str, str], int] = {}

        task_count_before = len(messages)

        for message in messages:
            task = message["task"]
            task_id = task.get("task_id")
            op = task.get("op")
            collection = task.get("collection_name")
            payload = task.get("payload") or {}

            if task.get("parse_error"):
                batches.append(
                    {
                        "op": None,
                        "collection_name": collection,
                        "payload": {},
                        "tasks": [task],
                        "message_ids": [message["message_id"]],
                    }
                )
                continue

            if op in {"insert", "upsert"} and collection:
                key = (op, collection)
                if key in merge_index:
                    batch = batches[merge_index[key]]
                    batch["payload"]["items"].extend(payload.get("items") or [])
                    batch["tasks"].append(task)
                    batch["message_ids"].append(message["message_id"])
                    continue

                merge_index[key] = len(batches)
                batches.append(
                    {
                        "op": op,
                        "collection_name": collection,
                        "payload": {"items": list(payload.get("items") or [])},
                        "tasks": [task],
                        "message_ids": [message["message_id"]],
                    }
                )
                continue

            batches.append(
                {
                    "op": op,
                    "collection_name": collection,
                    "payload": payload,
                    "tasks": [task],
                    "message_ids": [message["message_id"]],
                }
            )

        log.debug(
            "Coalesced chroma batches tasks_before=%s batches_after=%s",
            task_count_before,
            len(batches),
        )
        return batches

    async def _process_batch(self, batch: dict):
        op = batch["op"]
        collection_name = batch["collection_name"]
        task_ids = [task.get("task_id") for task in batch["tasks"]]
        started_at = time.monotonic()
        log.info(
            "Processing chroma batch op=%s collection=%s task_count=%s message_count=%s task_ids=%s",
            op,
            collection_name,
            len(batch["tasks"]),
            len(batch["message_ids"]),
            task_ids,
        )

        if not op:
            await self._emit_batch_failure(task_ids, "Invalid queued chroma task payload")
            await self.redis.xack(self.stream_key, self.group_name, *batch["message_ids"])
            return

        lock_token = None
        try:
            if collection_name:
                lock_token = await self._acquire_collection_lock(collection_name)
                if not lock_token:
                    raise RuntimeError(
                        f"Timed out waiting for collection write lock: {collection_name}"
                    )

            result = await asyncio.to_thread(
                self._execute_op,
                op,
                collection_name,
                batch["payload"],
            )
            await self._emit_batch_success(task_ids, result)
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            log.info(
                "Processed chroma batch success op=%s collection=%s task_count=%s elapsed_ms=%s ack_count=%s",
                op,
                collection_name,
                len(task_ids),
                elapsed_ms,
                len(batch["message_ids"]),
            )
        except Exception as e:
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            await self._emit_batch_failure(task_ids, str(e))
            log.error(
                "Processed chroma batch failure op=%s collection=%s task_ids=%s elapsed_ms=%s exception_type=%s error=%s",
                op,
                collection_name,
                task_ids,
                elapsed_ms,
                type(e).__name__,
                e,
            )
        finally:
            if lock_token and collection_name:
                await self._release_collection_lock(collection_name, lock_token)

            await self.redis.xack(self.stream_key, self.group_name, *batch["message_ids"])

    def _execute_op(self, op: str, collection_name: Optional[str], payload: dict):
        if op == "insert":
            return self.inner.insert(collection_name=collection_name, items=payload["items"])
        if op == "upsert":
            return self.inner.upsert(collection_name=collection_name, items=payload["items"])
        if op == "delete":
            return self.inner.delete(
                collection_name=collection_name,
                ids=payload.get("ids"),
                filter=payload.get("filter"),
            )
        if op == "delete_collection":
            return self.inner.delete_collection(collection_name=collection_name)
        if op == "reset":
            return self.inner.reset()

        raise ValueError(f"Unsupported chroma queued op: {op}")

    async def _acquire_collection_lock(self, collection_name: str) -> Optional[str]:
        lock_name = f"{self.lock_prefix}:{collection_name}"
        token = str(uuid.uuid4())

        deadline = time.monotonic() + self.collection_lock_wait_seconds
        while time.monotonic() < deadline:
            ok = await self.redis.set(
                lock_name,
                token,
                nx=True,
                ex=self.collection_lock_ttl_seconds,
            )
            if ok:
                return token
            await asyncio.sleep(0.1)

        return None

    async def _release_collection_lock(self, collection_name: str, token: str):
        lock_name = f"{self.lock_prefix}:{collection_name}"
        current = await self.redis.get(lock_name)
        if current == token:
            await self.redis.delete(lock_name)

    async def _emit_batch_success(self, task_ids: list[Optional[str]], data: Any):
        for task_id in task_ids:
            if not task_id:
                continue
            key = f"{self.result_key_prefix}:{task_id}"
            await self.redis.rpush(key, _json_dumps({"ok": True, "data": data}))
            await self.redis.expire(key, self.result_ttl_seconds)

    async def _emit_batch_failure(self, task_ids: list[Optional[str]], error: str):
        for task_id in task_ids:
            if not task_id:
                continue
            key = f"{self.result_key_prefix}:{task_id}"
            await self.redis.rpush(key, _json_dumps({"ok": False, "error": error}))
            await self.redis.expire(key, self.result_ttl_seconds)


class ChromaQueuedClient(VectorDBBase):
    def __init__(self, inner_client: VectorDBBase, coordinator: ChromaWriteCoordinator):
        self._inner = inner_client
        self._coordinator = coordinator
        self.client = getattr(inner_client, "client", None)

    def __getattr__(self, item):
        return getattr(self._inner, item)

    @staticmethod
    def _normalize_items(items: list[VectorItem]) -> list[dict]:
        normalized = []
        for item in items:
            if hasattr(item, "model_dump"):
                normalized.append(item.model_dump())
            else:
                normalized.append(dict(item))
        return normalized

    def has_collection(self, collection_name: str) -> bool:
        return self._inner.has_collection(collection_name)

    def delete_collection(self, collection_name: str):
        return self._coordinator.enqueue_and_wait(
            op="delete_collection",
            collection_name=collection_name,
            payload={},
        )

    def search(
        self, collection_name: str, vectors: list[list[float | int]], limit: int
    ) -> Optional[SearchResult]:
        return self._inner.search(collection_name=collection_name, vectors=vectors, limit=limit)

    def query(
        self, collection_name: str, filter: dict, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        return self._inner.query(collection_name=collection_name, filter=filter, limit=limit)

    def get(self, collection_name: str) -> Optional[GetResult]:
        return self._inner.get(collection_name=collection_name)

    def insert(self, collection_name: str, items: list[VectorItem]):
        return self._coordinator.enqueue_and_wait(
            op="insert",
            collection_name=collection_name,
            payload={"items": self._normalize_items(items)},
        )

    def upsert(self, collection_name: str, items: list[VectorItem]):
        return self._coordinator.enqueue_and_wait(
            op="upsert",
            collection_name=collection_name,
            payload={"items": self._normalize_items(items)},
        )

    def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ):
        return self._coordinator.enqueue_and_wait(
            op="delete",
            collection_name=collection_name,
            payload={"ids": ids, "filter": filter},
        )

    def reset(self):
        return self._coordinator.enqueue_and_wait(
            op="reset",
            collection_name=None,
            payload={},
        )
