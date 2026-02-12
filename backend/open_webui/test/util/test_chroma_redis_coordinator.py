import asyncio
import json

import pytest
import redis

from open_webui.retrieval.vector.chroma_redis_coordinator import (
    ChromaQueuedClient,
    ChromaWriteCoordinator,
    ChromaWriteWorker,
)


class DummySyncRedis:
    def __init__(self, blpop_response=None):
        self.blpop_response = blpop_response
        self.xadd_calls = []
        self.blpop_calls = []

    def xadd(self, key, fields):
        self.xadd_calls.append((key, fields))

    def blpop(self, key, timeout):
        self.blpop_calls.append((key, timeout))
        if self.blpop_response is None:
            return None
        return (key, json.dumps(self.blpop_response))


class DummyInnerClient:
    def __init__(self):
        self.calls = []

    def has_collection(self, collection_name):
        self.calls.append(("has_collection", collection_name))
        return True

    def delete_collection(self, collection_name):
        self.calls.append(("delete_collection", collection_name))

    def insert(self, collection_name, items):
        self.calls.append(("insert", collection_name, items))

    def upsert(self, collection_name, items):
        self.calls.append(("upsert", collection_name, items))

    def search(self, collection_name, vectors, limit):
        self.calls.append(("search", collection_name, vectors, limit))
        return {"ok": True}

    def query(self, collection_name, filter, limit=None):
        self.calls.append(("query", collection_name, filter, limit))
        return {"ok": True}

    def get(self, collection_name):
        self.calls.append(("get", collection_name))
        return {"ok": True}

    def delete(self, collection_name, ids=None, filter=None):
        self.calls.append(("delete", collection_name, ids, filter))

    def reset(self):
        self.calls.append(("reset",))
        return True


class DummyCoordinator:
    def __init__(self):
        self.calls = []

    def enqueue_and_wait(self, op, collection_name, payload):
        self.calls.append((op, collection_name, payload))
        return None


class DummyAsyncRedis:
    def __init__(self):
        self.xack_calls = []
        self.rpush_calls = []
        self.expire_calls = []
        self.store = {}
        self.xautoclaim_responses = []
        self.xreadgroup_responses = []
        self.xgroup_create_calls = []

    async def xack(self, stream_key, group_name, *ids):
        self.xack_calls.append((stream_key, group_name, ids))

    async def rpush(self, key, value):
        self.rpush_calls.append((key, value))

    async def expire(self, key, ttl):
        self.expire_calls.append((key, ttl))

    async def set(self, key, value, nx=False, xx=False, ex=None):
        if nx and key in self.store:
            return False
        if xx and key not in self.store:
            return False
        self.store[key] = value
        return True

    async def get(self, key):
        return self.store.get(key)

    async def delete(self, key):
        self.store.pop(key, None)

    async def xautoclaim(
        self,
        name,
        groupname,
        consumername,
        min_idle_time,
        start_id,
        count,
    ):
        if self.xautoclaim_responses:
            response = self.xautoclaim_responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return ("0-0", [], [])

    async def xreadgroup(
        self,
        groupname,
        consumername,
        streams,
        count,
        block,
    ):
        if self.xreadgroup_responses:
            response = self.xreadgroup_responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return []

    async def xgroup_create(self, name, groupname, id, mkstream=True):
        self.xgroup_create_calls.append((name, groupname, id, mkstream))
        return True


def test_coordinator_enqueue_and_wait_success():
    redis_client = DummySyncRedis(blpop_response={"ok": True, "data": {"done": True}})
    coordinator = ChromaWriteCoordinator(redis_client=redis_client, key_prefix="open-webui")

    data = coordinator.enqueue_and_wait(
        op="upsert",
        collection_name="collection-a",
        payload={"items": [{"id": "1"}]},
    )

    assert data == {"done": True}
    assert redis_client.xadd_calls
    stream_key, fields = redis_client.xadd_calls[0]
    assert stream_key == "open-webui:chroma:write:stream"
    task = json.loads(fields["task"])
    assert task["op"] == "upsert"
    assert task["collection_name"] == "collection-a"


def test_coordinator_enqueue_and_wait_failure():
    redis_client = DummySyncRedis(blpop_response={"ok": False, "error": "boom"})
    coordinator = ChromaWriteCoordinator(redis_client=redis_client, key_prefix="open-webui")

    with pytest.raises(RuntimeError, match="boom"):
        coordinator.enqueue_and_wait(op="delete", collection_name="col", payload={})


def test_coordinator_enqueue_and_wait_timeout():
    redis_client = DummySyncRedis(blpop_response=None)
    coordinator = ChromaWriteCoordinator(redis_client=redis_client, key_prefix="open-webui")

    with pytest.raises(TimeoutError):
        coordinator.enqueue_and_wait(op="delete", collection_name="col", payload={})


def test_queued_client_reads_passthrough_and_writes_queued():
    inner = DummyInnerClient()
    coordinator = DummyCoordinator()
    client = ChromaQueuedClient(inner_client=inner, coordinator=coordinator)

    assert client.search("c", [[0.1]], 1) == {"ok": True}
    assert client.query("c", {"a": 1}) == {"ok": True}
    assert client.get("c") == {"ok": True}

    client.upsert(
        "c",
        [
            {
                "id": "1",
                "text": "doc",
                "vector": [0.1],
                "metadata": {},
            }
        ],
    )

    assert any(call[0] == "search" for call in inner.calls)
    assert coordinator.calls == [
        (
            "upsert",
            "c",
            {
                "items": [
                    {
                        "id": "1",
                        "text": "doc",
                        "vector": [0.1],
                        "metadata": {},
                    }
                ]
            },
        )
    ]


def test_worker_coalesces_same_collection_upsert():
    worker = ChromaWriteWorker(
        inner_client=DummyInnerClient(),
        redis_client=DummyAsyncRedis(),
        key_prefix="open-webui",
    )

    messages = [
        {
            "message_id": "1-0",
            "task": {
                "task_id": "t1",
                "op": "upsert",
                "collection_name": "col",
                "payload": {"items": [{"id": "1"}]},
            },
        },
        {
            "message_id": "2-0",
            "task": {
                "task_id": "t2",
                "op": "upsert",
                "collection_name": "col",
                "payload": {"items": [{"id": "2"}]},
            },
        },
        {
            "message_id": "3-0",
            "task": {
                "task_id": "t3",
                "op": "delete",
                "collection_name": "col",
                "payload": {"ids": ["1"]},
            },
        },
    ]

    batches = worker._coalesce_messages(messages)
    assert len(batches) == 2
    assert batches[0]["op"] == "upsert"
    assert len(batches[0]["payload"]["items"]) == 2
    assert batches[1]["op"] == "delete"


def test_worker_process_batch_lock_failure_sends_error():
    async def _run_test():
        redis_client = DummyAsyncRedis()
        inner = DummyInnerClient()
        worker = ChromaWriteWorker(
            inner_client=inner,
            redis_client=redis_client,
            key_prefix="open-webui",
        )

        async def fail_lock(_collection):
            return None

        worker._acquire_collection_lock = fail_lock

        batch = {
            "op": "upsert",
            "collection_name": "col",
            "payload": {
                "items": [
                    {
                        "id": "1",
                        "text": "doc",
                        "vector": [0.1],
                        "metadata": {},
                    }
                ]
            },
            "tasks": [{"task_id": "task-1"}],
            "message_ids": ["1-0"],
        }

        await worker._process_batch(batch)

        assert not inner.calls
        assert redis_client.xack_calls
        assert redis_client.rpush_calls
        assert "\"ok\":false" in redis_client.rpush_calls[0][1]

    asyncio.run(_run_test())


def test_worker_recovers_pending_messages():
    async def _run_test():
        redis_client = DummyAsyncRedis()
        redis_client.xautoclaim_responses = [
            (
                "0-0",
                [
                    (
                        "1-0",
                        {
                            "task": json.dumps(
                                {
                                    "task_id": "task-1",
                                    "op": "reset",
                                    "collection_name": None,
                                    "payload": {},
                                }
                            )
                        },
                    )
                ],
                [],
            ),
            ("0-0", [], []),
        ]

        inner = DummyInnerClient()
        worker = ChromaWriteWorker(
            inner_client=inner,
            redis_client=redis_client,
            key_prefix="open-webui",
        )

        await worker._recover_pending_messages()

        assert ("reset",) in inner.calls
        assert redis_client.xack_calls

    asyncio.run(_run_test())


def test_worker_recovers_group_when_xautoclaim_returns_nogroup():
    async def _run_test():
        redis_client = DummyAsyncRedis()
        redis_client.xautoclaim_responses = [
            redis.exceptions.ResponseError("NOGROUP missing group"),
        ]

        worker = ChromaWriteWorker(
            inner_client=DummyInnerClient(),
            redis_client=redis_client,
            key_prefix="open-webui",
        )

        await worker._recover_pending_messages()

        assert redis_client.xgroup_create_calls

    asyncio.run(_run_test())


def test_worker_recovers_group_when_xreadgroup_returns_nogroup():
    async def _run_test():
        redis_client = DummyAsyncRedis()
        redis_client.xreadgroup_responses = [
            redis.exceptions.ResponseError("NOGROUP missing group"),
        ]

        worker = ChromaWriteWorker(
            inner_client=DummyInnerClient(),
            redis_client=redis_client,
            key_prefix="open-webui",
        )

        records = await worker._read_batch()

        assert records == []
        assert redis_client.xgroup_create_calls

    asyncio.run(_run_test())


def test_worker_does_not_swallow_non_nogroup_errors():
    async def _run_test():
        redis_client = DummyAsyncRedis()
        redis_client.xreadgroup_responses = [
            redis.exceptions.ResponseError("WRONGTYPE invalid key type"),
        ]

        worker = ChromaWriteWorker(
            inner_client=DummyInnerClient(),
            redis_client=redis_client,
            key_prefix="open-webui",
        )

        with pytest.raises(redis.exceptions.ResponseError, match="WRONGTYPE"):
            await worker._read_batch()

    asyncio.run(_run_test())


def test_worker_nogroup_recovery_logging_is_rate_limited(caplog):
    async def _run_test():
        redis_client = DummyAsyncRedis()
        worker = ChromaWriteWorker(
            inner_client=DummyInnerClient(),
            redis_client=redis_client,
            key_prefix="open-webui",
        )
        worker.nogroup_log_interval_seconds = 10

        caplog.clear()
        with caplog.at_level("WARNING"):
            await worker._recover_stream_group_if_missing()
            await worker._recover_stream_group_if_missing()

        messages = [
            record.message
            for record in caplog.records
            if "Detected missing Chroma Redis stream/group" in record.message
        ]
        assert len(messages) == 1

    asyncio.run(_run_test())
