import logging
from typing import Any, Literal, Optional
from urllib.parse import urlparse, urlunparse

from fastmcp import Client
from fastmcp.client.transports import SSETransport, StreamableHttpTransport

MCPTransport = Literal["streamable_http", "sse"]

log = logging.getLogger(__name__)


class MCPClientError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class MCPClient:
    def __init__(self):
        self.client: Optional[Client] = None
        self._connected = False

    async def connect(
        self,
        url: str,
        headers: Optional[dict] = None,
        transport: MCPTransport = "streamable_http",
        connect_timeout: int = 15,
        initialize_timeout: int = 15,
    ):
        if transport not in ("streamable_http", "sse"):
            raise MCPClientError(f"Invalid MCP transport: {transport}", status_code=400)

        if self._connected:
            return

        try:
            transport_client = self._build_transport(
                url=url,
                transport=transport,
                headers=headers,
            )
            self.client = Client(
                transport_client,
                timeout=connect_timeout,
                init_timeout=initialize_timeout,
            )
            await self.client.__aenter__()
            self._connected = True
        except Exception as exc:
            self.client = None
            self._connected = False
            raise MCPClientError(f"Failed to connect to MCP server: {exc}") from exc

    def _build_transport(
        self,
        url: str,
        transport: MCPTransport,
        headers: Optional[dict],
    ) -> StreamableHttpTransport | SSETransport:
        transport_url = self._resolve_transport_url(url=url, transport=transport)

        if transport == "streamable_http":
            return StreamableHttpTransport(url=transport_url, headers=headers)

        return SSETransport(url=transport_url, headers=headers)

    def _resolve_transport_url(self, url: str, transport: MCPTransport) -> str:
        if not url:
            raise MCPClientError("MCP server URL is required.")

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise MCPClientError(
                "MCP server URL must be an absolute HTTP(S) URL.",
                status_code=400,
            )

        endpoint_path = "/mcp" if transport == "streamable_http" else "/sse"

        # Keep explicit endpoint URLs unchanged and only map root URLs to defaults.
        if parsed.path in ("", "/"):
            parsed = parsed._replace(path=endpoint_path)
            return urlunparse(parsed)

        return url

    def _serialize(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    async def list_tool_specs(self) -> list[dict]:
        if not self.client or not self._connected:
            raise MCPClientError("MCP client is not connected.")

        try:
            tools = await self.client.list_tools()
            tool_specs = []
            for tool in tools:
                tool_specs.append(
                    {
                        "name": tool.name,
                        "description": getattr(tool, "description", None),
                        "parameters": self._serialize(
                            getattr(tool, "inputSchema", None)
                            or getattr(tool, "input_schema", None)
                        ),
                    }
                )
            return tool_specs
        except Exception as exc:
            raise MCPClientError(f"Failed to list MCP tools: {exc}") from exc

    async def call_tool(self, function_name: str, function_args: dict) -> list[dict]:
        if not self.client or not self._connected:
            raise MCPClientError("MCP client is not connected.")

        try:
            result = await self.client.call_tool(
                function_name,
                arguments=function_args,
                raise_on_error=False,
            )
            if result.is_error:
                raise MCPClientError(str(result.content))

            return [self._serialize(content) for content in (result.content or [])]
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientError(f"Failed to call MCP tool: {exc}") from exc

    async def list_resources(self, cursor: Optional[str] = None) -> list[dict]:
        if not self.client or not self._connected:
            raise MCPClientError("MCP client is not connected.")

        try:
            if cursor is None:
                resources = await self.client.list_resources()
            else:
                result = await self.client.list_resources_mcp(cursor=cursor)
                resources = result.resources

            return [self._serialize(resource) for resource in resources]
        except Exception as exc:
            raise MCPClientError(f"Failed to list MCP resources: {exc}") from exc

    async def read_resource(self, uri: str) -> dict:
        if not self.client or not self._connected:
            raise MCPClientError("MCP client is not connected.")

        try:
            contents = await self.client.read_resource(uri)
            return {
                "contents": [self._serialize(content) for content in contents],
            }
        except Exception as exc:
            raise MCPClientError(f"Failed to read MCP resource: {exc}") from exc

    async def disconnect(self):
        if self.client is None:
            return

        try:
            if self._connected:
                await self.client.__aexit__(None, None, None)
        except Exception as exc:
            log.debug(f"Failed to disconnect MCP client cleanly: {exc}")
        finally:
            self.client = None
            self._connected = False
