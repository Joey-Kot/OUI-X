from types import SimpleNamespace

import pytest

from open_webui.routers import tools as tools_router


class _OAuthClientManager:
    async def get_oauth_token(self, user_id, token_key):
        return None


def _make_request(system_servers=None):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(
                    MCP_TOOL_SERVER_CONNECTIONS=system_servers or []
                ),
                oauth_client_manager=_OAuthClientManager(),
            )
        )
    )


def _make_user(user_servers=None):
    return SimpleNamespace(
        id="user-1",
        role="admin",
        settings={"ui": {"mcpToolServers": user_servers or []}},
    )


@pytest.fixture(autouse=True)
def _mock_local_tools(monkeypatch):
    monkeypatch.setattr(tools_router.Tools, "get_tools", lambda: [])
    monkeypatch.setattr(tools_router, "BYPASS_ADMIN_ACCESS_CONTROL", True)


async def _get_tool_ids(request, user):
    tools = await tools_router.get_tools(request=request, user=user)
    return [tool.id for tool in tools]


@pytest.mark.asyncio
async def test_get_tools_hides_disabled_system_mcp_connection():
    request = _make_request(
        [
            {
                "info": {"id": "sys-disabled", "name": "Disabled"},
                "config": {"enable": False},
            }
        ]
    )

    tool_ids = await _get_tool_ids(request, _make_user())

    assert tool_ids == []


@pytest.mark.asyncio
async def test_get_tools_hides_disabled_user_mcp_connection():
    request = _make_request()
    user = _make_user(
        [
            {
                "info": {"id": "user-disabled", "name": "Disabled"},
                "config": {"enable": False},
            }
        ]
    )

    tool_ids = await _get_tool_ids(request, user)

    assert tool_ids == []


@pytest.mark.asyncio
async def test_get_tools_hides_system_mcp_when_all_verified_tools_disabled():
    request = _make_request(
        [
            {
                "info": {"id": "sys-all-disabled", "name": "All Disabled"},
                "config": {
                    "tools": {
                        "alpha": {"enabled": False},
                        "beta": {"enabled": False},
                    }
                },
                "verify_cache": {
                    "tools": [{"name": "alpha"}, {"name": "beta"}]
                },
            }
        ]
    )

    tool_ids = await _get_tool_ids(request, _make_user())

    assert tool_ids == []


@pytest.mark.asyncio
async def test_get_tools_hides_user_mcp_when_all_verified_tools_disabled():
    request = _make_request()
    user = _make_user(
        [
            {
                "info": {"id": "user-all-disabled", "name": "All Disabled"},
                "config": {
                    "tools": {
                        "alpha": {"enabled": False},
                        "beta": {"enabled": False},
                    }
                },
                "verify_cache": {
                    "tools": [{"name": "alpha"}, {"name": "beta"}]
                },
            }
        ]
    )

    tool_ids = await _get_tool_ids(request, user)

    assert tool_ids == []


@pytest.mark.asyncio
async def test_get_tools_keeps_system_mcp_visible_when_any_verified_tool_enabled():
    request = _make_request(
        [
            {
                "info": {"id": "sys-partial", "name": "Partial"},
                "config": {
                    "tools": {
                        "alpha": {"enabled": False},
                        "beta": {"enabled": True},
                    }
                },
                "verify_cache": {
                    "tools": [{"name": "alpha"}, {"name": "beta"}]
                },
            }
        ]
    )

    tool_ids = await _get_tool_ids(request, _make_user())

    assert tool_ids == ["server:mcp:sys-partial"]


@pytest.mark.asyncio
async def test_get_tools_keeps_mcp_visible_when_verify_cache_missing():
    request = _make_request(
        [
            {
                "info": {"id": "sys-unverified", "name": "Unverified"},
                "config": {"enable": True},
            }
        ]
    )

    tool_ids = await _get_tool_ids(request, _make_user())

    assert tool_ids == ["server:mcp:sys-unverified"]


@pytest.mark.asyncio
async def test_get_tools_keeps_mcp_visible_when_config_missing():
    request = _make_request(
        [
            {
                "info": {"id": "sys-no-config", "name": "No Config"},
                "verify_cache": {"tools": [{"name": "alpha"}]},
            }
        ]
    )

    tool_ids = await _get_tool_ids(request, _make_user())

    assert tool_ids == ["server:mcp:sys-no-config"]


@pytest.mark.asyncio
async def test_get_tools_keeps_mcp_visible_when_tools_config_missing():
    request = _make_request(
        [
            {
                "info": {"id": "sys-no-tools-config", "name": "No Tools Config"},
                "config": {"enable": True},
                "verify_cache": {"tools": [{"name": "alpha"}]},
            }
        ]
    )

    tool_ids = await _get_tool_ids(request, _make_user())

    assert tool_ids == ["server:mcp:sys-no-tools-config"]
