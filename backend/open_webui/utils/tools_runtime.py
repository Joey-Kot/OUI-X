import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from open_webui.models.users import Users, UserModel
from open_webui.utils.mcp.client import MCPClient
from open_webui.utils.tools import get_tools

log = logging.getLogger(__name__)

ToolExecutor = Callable[..., Awaitable[Any]]


@dataclass
class ToolRegistryEntry:
    name: str
    spec: dict
    executor: ToolExecutor
    source: str
    meta: dict = field(default_factory=dict)


@dataclass
class ToolExecutionOutcome:
    tool_call_id: str
    name: str
    ok: bool
    content: str
    files: list[dict] = field(default_factory=list)
    embeds: list[Any] = field(default_factory=list)
    error: str | None = None


def normalize_function_schema(schema: dict | None) -> dict:
    schema = dict(schema or {})
    schema_type = schema.get("type")
    if schema_type != "object":
        schema["type"] = "object"

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    schema["properties"] = properties

    required = schema.get("required")
    if not isinstance(required, list):
        required = []
    schema["required"] = [key for key in required if isinstance(key, str)]

    if "additionalProperties" in schema and not isinstance(
        schema["additionalProperties"], (bool, dict)
    ):
        schema["additionalProperties"] = False

    for key, value in list(properties.items()):
        if not isinstance(value, dict):
            properties[key] = {"type": "string"}
            continue
        if value.get("type") == "str":
            value["type"] = "string"

    return schema


def _normalize_spec(spec: dict, name: str | None = None) -> dict:
    normalized = dict(spec or {})
    if name:
        normalized["name"] = name
    normalized["parameters"] = normalize_function_schema(normalized.get("parameters"))
    if not normalized.get("description"):
        normalized["description"] = normalized.get("name", "tool")
    return normalized


async def build_local_registry(
    request,
    tool_ids: list[str] | None,
    user: UserModel,
    extra_params: dict,
    model: dict,
    messages: list[dict],
    files: list[dict] | None,
    include_openapi: bool,
) -> dict[str, ToolRegistryEntry]:
    if not tool_ids:
        return {}

    filtered_tool_ids = []
    for tool_id in tool_ids:
        if tool_id.startswith("server:mcp:"):
            continue
        if not include_openapi and tool_id.startswith("server:"):
            continue
        filtered_tool_ids.append(tool_id)

    if not filtered_tool_ids:
        return {}

    local_tools = await get_tools(
        request,
        filtered_tool_ids,
        user,
        {
            **extra_params,
            "__model__": model,
            "__messages__": messages,
            "__files__": files or [],
        },
    )

    registry: dict[str, ToolRegistryEntry] = {}
    for name, tool in local_tools.items():
        spec = _normalize_spec(tool.get("spec", {}), name=name)
        registry[name] = ToolRegistryEntry(
            name=name,
            spec=spec,
            executor=tool.get("callable"),
            source="local",
            meta={
                "tool_id": tool.get("tool_id"),
                "type": tool.get("type", "local"),
                "direct": tool.get("direct", False),
                "server": tool.get("server", {}),
                "metadata": tool.get("metadata", {}),
            },
        )

    return registry


def _merge_headers(headers: dict, connection_headers: dict | None) -> dict:
    if connection_headers and isinstance(connection_headers, dict):
        for key, value in connection_headers.items():
            headers[key] = value
    return headers


async def _resolve_user_mcp_connection(request, user: UserModel, server_id: str):
    user_record = Users.get_user_by_id(user.id)
    user_settings = user_record.settings if user_record else {}
    if hasattr(user_settings, "model_dump"):
        user_settings = user_settings.model_dump()

    for server_connection in user_settings.get("ui", {}).get("mcpToolServers", []):
        if server_connection.get("info", {}).get("id") == server_id:
            return server_connection

    return None


async def _resolve_auth_headers(request, extra_params: dict, user: UserModel, connection: dict, scope: str, server_id: str) -> dict:
    auth_type = connection.get("auth_type", "")
    headers = {}

    if auth_type == "bearer":
        headers["Authorization"] = f"Bearer {connection.get('key', '')}"
    elif auth_type == "session":
        headers["Authorization"] = f"Bearer {request.state.token.credentials}"
    elif auth_type == "system_oauth":
        oauth_token = extra_params.get("__oauth_token__", None)
        if oauth_token:
            headers["Authorization"] = (
                f"Bearer {oauth_token.get('access_token', '')}"
            )
    elif auth_type == "oauth_2.1":
        try:
            oauth_server_id = server_id
            if scope == "admin":
                splits = server_id.split(":")
                oauth_server_id = splits[-1] if len(splits) > 1 else server_id
                token_key = f"mcp:{oauth_server_id}"
            else:
                token_key = f"mcp:user:{user.id}:{oauth_server_id}"

            oauth_token = await request.app.state.oauth_client_manager.get_oauth_token(
                user.id, token_key
            )
            if oauth_token:
                headers["Authorization"] = (
                    f"Bearer {oauth_token.get('access_token', '')}"
                )
        except Exception as exc:
            log.error(f"Error getting OAuth token for MCP server {server_id}: {exc}")

    return _merge_headers(headers, connection.get("headers", None))


async def build_mcp_registry(
    request,
    tool_ids: list[str] | None,
    user: UserModel,
    extra_params: dict,
    event_emitter=None,
) -> tuple[dict[str, ToolRegistryEntry], dict[str, MCPClient]]:
    registry: dict[str, ToolRegistryEntry] = {}
    mcp_clients: dict[str, MCPClient] = {}

    if not tool_ids:
        return registry, mcp_clients

    async def register_tool_specs(
        client: MCPClient,
        client_key: str,
        server_id: str,
        tool_name_prefix: str,
        tools_config: dict,
        scope: str,
        transport: str,
    ):
        tool_specs = await client.list_tool_specs()
        for tool_spec in tool_specs:
            if not tools_config.get(tool_spec.get("name", ""), {}).get("enabled", True):
                continue

            upstream_name = tool_spec.get("name", "")
            function_name = f"{tool_name_prefix}_{upstream_name}"

            async def tool_function(_name=upstream_name, **kwargs):
                return await client.call_tool(_name, function_args=kwargs)

            registry[function_name] = ToolRegistryEntry(
                name=function_name,
                spec=_normalize_spec(tool_spec, name=function_name),
                executor=tool_function,
                source="mcp",
                meta={
                    "server_id": server_id,
                    "client_key": client_key,
                    "scope": scope,
                    "transport": transport,
                },
            )

    for tool_id in tool_ids:
        try:
            if tool_id.startswith("server:mcp:user:"):
                server_id = tool_id[len("server:mcp:user:") :]
                connection = await _resolve_user_mcp_connection(request, user, server_id)
                if not connection:
                    raise RuntimeError(f"User MCP server with id {server_id} not found")

                headers = await _resolve_auth_headers(
                    request=request,
                    extra_params=extra_params,
                    user=user,
                    connection=connection,
                    scope="user",
                    server_id=server_id,
                )
                client_key = f"user_{server_id}"
                client = MCPClient()
                transport = connection.get("transport", "streamable_http")
                await client.connect(
                    url=connection.get("url", ""),
                    headers=headers if headers else None,
                    transport=transport,
                )
                mcp_clients[client_key] = client
                await register_tool_specs(
                    client=client,
                    client_key=client_key,
                    server_id=server_id,
                    tool_name_prefix=f"user_{server_id}",
                    tools_config=connection.get("config", {}).get("tools", {}) or {},
                    scope="user",
                    transport=transport,
                )

            elif tool_id.startswith("server:mcp:"):
                server_id = tool_id[len("server:mcp:") :]
                connection = None
                for server_connection in getattr(
                    request.app.state.config, "MCP_TOOL_SERVER_CONNECTIONS", []
                ):
                    if server_connection.get("info", {}).get("id") == server_id:
                        connection = server_connection
                        break

                if not connection:
                    raise RuntimeError(f"MCP server with id {server_id} not found")

                headers = await _resolve_auth_headers(
                    request=request,
                    extra_params=extra_params,
                    user=user,
                    connection=connection,
                    scope="admin",
                    server_id=server_id,
                )

                client = MCPClient()
                transport = connection.get("transport", "streamable_http")
                await client.connect(
                    url=connection.get("url", ""),
                    headers=headers if headers else None,
                    transport=transport,
                )
                mcp_clients[server_id] = client
                await register_tool_specs(
                    client=client,
                    client_key=server_id,
                    server_id=server_id,
                    tool_name_prefix=server_id,
                    tools_config=connection.get("config", {}).get("tools", {}) or {},
                    scope="admin",
                    transport=transport,
                )
        except Exception as exc:
            log.debug(exc)
            if event_emitter:
                message = (
                    f"Failed to connect to user MCP server {server_id}"
                    if tool_id.startswith("server:mcp:user:")
                    else f"Failed to connect to MCP server {server_id}"
                )
                await event_emitter(
                    {
                        "type": "chat:message:error",
                        "data": {"error": {"content": message}},
                    }
                )

    return registry, mcp_clients


def _register_with_dedupe(
    output: dict[str, ToolRegistryEntry],
    warnings: list[str],
    entry: ToolRegistryEntry,
):
    if entry.name not in output:
        output[entry.name] = entry
        return

    idx = 2
    while f"{entry.name}_v{idx}" in output:
        idx += 1
    renamed = f"{entry.name}_v{idx}"
    warnings.append(f"Tool name collision: {entry.name} -> {renamed}")

    output[renamed] = ToolRegistryEntry(
        name=renamed,
        spec=_normalize_spec(entry.spec, name=renamed),
        executor=entry.executor,
        source=entry.source,
        meta={**entry.meta, "renamed_from": entry.name},
    )


async def build_tool_registry(
    request,
    tool_ids: list[str] | None,
    user: UserModel,
    extra_params: dict,
    model: dict,
    messages: list[dict],
    files: list[dict] | None,
    event_emitter=None,
    include_openapi: bool = True,
) -> tuple[dict[str, ToolRegistryEntry], dict[str, MCPClient], list[str]]:
    local_registry = await build_local_registry(
        request=request,
        tool_ids=tool_ids,
        user=user,
        extra_params=extra_params,
        model=model,
        messages=messages,
        files=files,
        include_openapi=include_openapi,
    )
    mcp_registry, mcp_clients = await build_mcp_registry(
        request=request,
        tool_ids=tool_ids,
        user=user,
        extra_params=extra_params,
        event_emitter=event_emitter,
    )

    merged: dict[str, ToolRegistryEntry] = {}
    warnings: list[str] = []

    for entry in [*local_registry.values(), *mcp_registry.values()]:
        _register_with_dedupe(merged, warnings, entry)

    for warning in warnings:
        log.warning(warning)

    return merged, mcp_clients, warnings


def registry_to_legacy_tools(registry: dict[str, ToolRegistryEntry]) -> dict[str, dict]:
    tools = {}
    for name, entry in registry.items():
        meta = entry.meta or {}
        tools[name] = {
            "spec": _normalize_spec(entry.spec, name=name),
            "callable": entry.executor,
            "type": meta.get("type", entry.source),
            "source": entry.source,
            "scope": meta.get("scope"),
            "direct": meta.get("direct", False),
            "server": meta.get("server", {}),
            "tool_id": meta.get("tool_id"),
            "metadata": meta.get("metadata", {}),
            "client": meta.get("client"),
        }
    return tools


def serialize_tool_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)
