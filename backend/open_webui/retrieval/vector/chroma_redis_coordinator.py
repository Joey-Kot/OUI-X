import asyncio
import json
import logging
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
        self.wait_timeout_seconds = wait_timeout_seconds

    def _result_key(self, task_id: str) -> str:
        return f"{self.result_key_prefix}:{task_id}"

    def enqueue_and_wait(self, op: str, collection_name: Optional[str], payload: dict):
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "op": op,
            "collection_name": collection_name,
            "payload": payload,
            "created_at": int(time.time()),
        }

        self.redis.xadd(self.stream_key, {"task": _json_dumps(task)})

        result = self.redis.blpop(self._result_key(task_id), timeout=self.wait_timeout_seconds)
        if not result:
            raise TimeoutError(
                f"Timed out waiting for queued Chroma write result: op={op}, task_id={task_id}"
            )

        _, data = result
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        response = json.loads(data)
        if response.get("ok"):
            return response.get("data")

        error = response.get("error") or "unknown queued Chroma write failure"
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
        self.consumer_name = consumer_name or f"worker-{uuid.uuid4()}"

        self.max_batch = 100
        self.batch_window_ms = 50
        self.read_block_ms = 1000
        self.result_ttl_seconds = 300

        self.leader_lock_token: Optional[str] = None
        self.leader_lock_ttl_seconds = 30
        self.leader_renew_interval_seconds = 10

        self.collection_lock_ttl_seconds = 60
        self.collection_lock_wait_seconds = 3
        self.pending_min_idle_ms = 5_000
        self.nogroup_log_interval_seconds = 10
        self._last_nogroup_log_ts = 0.0

    async def run(self):
        await self._ensure_group()

        while True:
            try:
                acquired = await self._acquire_leader()
                if not acquired:
                    await asyncio.sleep(1)
                    continue

                await self._run_as_leader()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Chroma write worker loop failed")
                await asyncio.sleep(1)
            finally:
                await self._release_leader()

    async def _run_as_leader(self):
        last_renew_at = time.monotonic()

        while True:
            now = time.monotonic()
            if now - last_renew_at >= self.leader_renew_interval_seconds:
                renewed = await self._renew_leader()
                if not renewed:
                    log.warning("Lost chroma redis worker leader lock")
                    return
                last_renew_at = now

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
                id="$",
                mkstream=True,
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    @staticmethod
    def _is_nogroup_error(exc: Exception) -> bool:
        return isinstance(exc, redis.exceptions.ResponseError) and "NOGROUP" in str(exc)

    async def _recover_stream_group_if_missing(self):
        await self._ensure_group()

        now = time.monotonic()
        if now - self._last_nogroup_log_ts >= self.nogroup_log_interval_seconds:
            log.warning(
                "Detected missing Chroma Redis stream/group, recreated stream=%s group=%s",
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
            return True
        return False

    async def _renew_leader(self) -> bool:
        if not self.leader_lock_token:
            return False

        current = await self.redis.get(self.leader_key)
        if current != self.leader_lock_token:
            return False

        renewed = await self.redis.set(
            self.leader_key,
            self.leader_lock_token,
            xx=True,
            ex=self.leader_lock_ttl_seconds,
        )
        return bool(renewed)

    async def _release_leader(self):
        if not self.leader_lock_token:
            return

        current = await self.redis.get(self.leader_key)
        if current == self.leader_lock_token:
            await self.redis.delete(self.leader_key)

        self.leader_lock_token = None

    async def _recover_pending_messages(self):
        start_id = "0-0"
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
            except Exception as e:
                if self._is_nogroup_error(e):
                    await self._recover_stream_group_if_missing()
                    return
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
                await self._recover_stream_group_if_missing()
                return []
            raise

        if not records:
            return []

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
                    await self._recover_stream_group_if_missing()
                    return []
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

        return batches

    async def _process_batch(self, batch: dict):
        op = batch["op"]
        collection_name = batch["collection_name"]
        task_ids = [task.get("task_id") for task in batch["tasks"]]

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
        except Exception as e:
            await self._emit_batch_failure(task_ids, str(e))
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
