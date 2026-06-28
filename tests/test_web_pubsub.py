"""Tests for the Redis Pub/Sub WebSocket fan-out in web/app.py (v1.2).

Verifies that multiple UI servers sharing a Redis backend stay in sync:
a job update broadcast by one server reaches the WebSocket clients
connected to another server via the shared Pub/Sub channel.

Uses ``fakeredis`` with a shared ``FakeServer`` so two
``RedisBroadcaster`` instances model two UI servers behind a load
balancer backed by one Redis cluster.
"""

import asyncio
import json

import pytest

pytestmark = [pytest.mark.federation, pytest.mark.web]

import fakeredis.aioredis

from web.app import (
    LocalBroadcaster,
    RedisBroadcaster,
    make_broadcaster,
    AppState,
)


# --------------------------------------------------------------------------- #
# Broadcaster unit tests
# --------------------------------------------------------------------------- #

class TestLocalBroadcaster:
    @pytest.mark.asyncio
    async def test_publish_calls_send_fn(self):
        received = []

        async def send_fn(msg):
            received.append(msg)

        b = LocalBroadcaster(send_fn)
        await b.start()
        await b.publish({"type": "ping"})
        await b.stop()
        assert received == [{"type": "ping"}]


class TestRedisBroadcaster:
    @pytest.mark.asyncio
    async def test_publish_reaches_subscriber_same_instance(self):
        """A single RedisBroadcaster: publish -> subscriber -> local send."""
        received = []

        async def send_fn(msg):
            received.append(msg)

        client = fakeredis.aioredis.FakeRedis()
        b = RedisBroadcaster(
            redis_url="redis://localhost:0", channel="vt:test",
            send_fn=send_fn, redis_client=client,
        )
        await b.start()
        # Give the subscriber a moment to subscribe.
        await asyncio.sleep(0.05)
        await b.publish({"type": "job_update", "job": {"job_id": "j1"}})
        # Allow the subscriber loop to process the message.
        await asyncio.sleep(0.1)
        await b.stop()
        assert len(received) == 1
        assert received[0]["type"] == "job_update"
        assert received[0]["job"]["job_id"] == "j1"

    @pytest.mark.asyncio
    async def test_cross_server_fan_out(self):
        """Server A publishes; server B's subscriber forwards to its clients.

        This is the HA invariant: two UI servers sharing one Redis stay
        in sync. A's clients and B's clients both receive the update.
        """
        received_a = []
        received_b = []

        async def send_a(msg):
            received_a.append(msg)

        async def send_b(msg):
            received_b.append(msg)

        # Shared FakeServer models one Redis cluster.
        server = fakeredis.aioredis.FakeServer()
        client_a = fakeredis.aioredis.FakeRedis(server=server)
        client_b = fakeredis.aioredis.FakeRedis(server=server)

        b_a = RedisBroadcaster(
            redis_url="redis://localhost:0", channel="vt:test",
            send_fn=send_a, redis_client=client_a,
        )
        b_b = RedisBroadcaster(
            redis_url="redis://localhost:0", channel="vt:test",
            send_fn=send_b, redis_client=client_b,
        )
        await b_a.start()
        await b_b.start()
        await asyncio.sleep(0.05)  # let both subscribe

        # A publishes — both A and B should receive it.
        await b_a.publish({"type": "job_update", "job": {"job_id": "shared"}})
        await asyncio.sleep(0.15)

        await b_a.stop()
        await b_b.stop()

        # Both servers' local clients got the message.
        assert any(m.get("job", {}).get("job_id") == "shared" for m in received_a)
        assert any(m.get("job", {}).get("job_id") == "shared" for m in received_b)

    @pytest.mark.asyncio
    async def test_publish_fallback_when_redis_down(self):
        """If Redis publish fails, local clients still get the message."""
        received = []

        async def send_fn(msg):
            received.append(msg)

        class BrokenRedis:
            async def publish(self, *a, **kw):
                raise ConnectionError("redis down")
            def pubsub(self):
                raise ConnectionError("redis down")
            async def aclose(self):
                pass

        b = RedisBroadcaster(
            redis_url="redis://localhost:0", channel="vt:test",
            send_fn=send_fn, redis_client=BrokenRedis(),
        )
        # Don't start the subscriber (it would fail to connect); just
        # test that publish falls back to local send.
        await b.publish({"type": "fallback"})
        assert received == [{"type": "fallback"}]


class TestMakeBroadcaster:
    def test_no_redis_url_returns_local(self):
        b = make_broadcaster(None, send_fn=None)
        assert isinstance(b, LocalBroadcaster)

    def test_empty_redis_url_returns_local(self):
        b = make_broadcaster("", send_fn=None)
        assert isinstance(b, LocalBroadcaster)

    def test_redis_url_returns_redis_broadcaster(self):
        b = make_broadcaster("redis://localhost:6379/0", send_fn=None)
        assert isinstance(b, RedisBroadcaster)


# --------------------------------------------------------------------------- #
# AppState integration
# --------------------------------------------------------------------------- #

class TestAppStateBroadcaster:
    @pytest.mark.asyncio
    async def test_default_is_local_broadcaster(self):
        from unittest.mock import MagicMock
        orch = MagicMock()
        orch.start = asyncio.coroutine(lambda: None) if False else None
        # Use a real async mock for start/cleanup.
        from unittest.mock import AsyncMock
        orch.start = AsyncMock()
        orch.cleanup = AsyncMock()
        state = AppState(orch)
        assert isinstance(state.broadcaster, LocalBroadcaster)
        await state.ensure_started()
        await state.cleanup()

    @pytest.mark.asyncio
    async def test_redis_broadcaster_lifecycle(self):
        """AppState with a RedisBroadcaster starts/stops the subscriber."""
        from unittest.mock import MagicMock, AsyncMock
        orch = MagicMock()
        orch.start = AsyncMock()
        orch.cleanup = AsyncMock()

        client = fakeredis.aioredis.FakeRedis()
        from web.app import make_broadcaster
        # Build state first (default local), then swap in Redis.
        state = AppState(orch)
        state.broadcaster = make_broadcaster(
            redis_url="redis://localhost:0",
            send_fn=state._send_to_local_clients,
            redis_client=client,
        )
        assert isinstance(state.broadcaster, RedisBroadcaster)
        await state.ensure_started()  # starts subscriber
        assert state.broadcaster._sub_task is not None
        await state.cleanup()  # stops subscriber
        assert state.broadcaster._sub_task is None
