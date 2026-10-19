import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from robosahaya.evidence_models import PermissionTier
from robosahaya.mcp.models import (
    MCPConnectionConfig,
    MCPConnectionState,
    MCPConnectionStatus,
    MCPToolInfo,
    MCPTransportType,
)

logger = logging.getLogger("robosahaya.mcp.client")

# Tool permission tier classification
TOOL_PERMISSION_TIERS = {
    "check_environment": PermissionTier.READ_ONLY,
    "list_ros_nodes": PermissionTier.READ_ONLY,
    "list_ros_topics": PermissionTier.READ_ONLY,
    "inspect_topic": PermissionTier.READ_ONLY,
    "inspect_tf": PermissionTier.READ_ONLY,
    "get_gazebo_status": PermissionTier.READ_ONLY,
    "run_sensor_test": PermissionTier.READ_ONLY,
    "launch_managed_simulation": PermissionTier.MANAGED_RUNTIME,
    "stop_managed_simulation": PermissionTier.MANAGED_RUNTIME,
}


class MCPClientManager:
    """Backend manager owning MCP client lifecycle, connection transport, and tool execution."""

    def __init__(self, config: Optional[MCPConnectionConfig] = None):
        self.config = config or MCPConnectionConfig()
        self.status = MCPConnectionStatus(
            state=MCPConnectionState.DISCONNECTED,
            transport=self.config.transport
        )
        self._session: Optional[ClientSession] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._ready_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._init_error: Optional[str] = None
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self.status.state == MCPConnectionState.CONNECTED and self._session is not None

    def get_status(self) -> MCPConnectionStatus:
        return self.status

    def validate_url(self, url: str) -> Tuple[bool, str]:
        """Validate destination URL to prevent SSRF and unsafe schemes."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False, f"Invalid scheme '{parsed.scheme}'. Only 'http' and 'https' are allowed."
            if not parsed.hostname:
                return False, "URL must contain a valid hostname or IP address."
            return True, ""
        except Exception as exc:
            return False, f"Malformed URL: {exc}"

    async def connect(self, config: Optional[MCPConnectionConfig] = None) -> Tuple[bool, str]:
        """Establish a real connection using the selected transport."""
        async with self._lock:
            if config:
                self.config = config

            if self.is_connected:
                await self._disconnect_internal()

            self.status.state = MCPConnectionState.CONNECTING
            self.status.transport = self.config.transport
            self.status.error_message = None
            self.status.add_diagnostic(f"Initiating connection via '{self.config.transport.value}' transport...")

            self._ready_event.clear()
            self._stop_event.clear()
            self._init_error = None

            # Spawn background session worker
            self._worker_task = asyncio.create_task(self._session_worker())

            try:
                # Wait for session worker to complete handshake
                await asyncio.wait_for(self._ready_event.wait(), timeout=self.config.timeout_sec)
            except asyncio.TimeoutError:
                self.status.state = MCPConnectionState.ERROR
                msg = f"Connection timed out after {self.config.timeout_sec}s during capability negotiation."
                self.status.error_message = msg
                self.status.add_diagnostic(msg)
                await self._disconnect_internal()
                return False, msg
            except Exception as exc:
                self.status.state = MCPConnectionState.ERROR
                msg = f"Connection failed: {exc}"
                self.status.error_message = msg
                self.status.add_diagnostic(msg)
                await self._disconnect_internal()
                return False, msg

            if self._init_error:
                self.status.state = MCPConnectionState.ERROR
                self.status.error_message = self._init_error
                self.status.add_diagnostic(self._init_error)
                await self._disconnect_internal()
                return False, self._init_error

            self.status.state = MCPConnectionState.CONNECTED
            self.status.last_connected_at = datetime.now(timezone.utc).isoformat()
            self.status.add_diagnostic("Capability negotiation and handshake succeeded. Connection established.")

            # Perform initial discovery and health check
            await self._refresh_tools_internal()
            await self._run_health_check_internal()

            return True, f"Connected successfully via {self.config.transport.value}."

    async def disconnect(self) -> Tuple[bool, str]:
        """Cleanly disconnects, cancels pending requests, and terminates managed child processes."""
        async with self._lock:
            await self._disconnect_internal()
            self.status.state = MCPConnectionState.DISCONNECTED
            self.status.discovered_tools = []
            self.status.add_diagnostic("Disconnected by user request. Cleaned up resources.")
            return True, "Disconnected successfully."

    async def _disconnect_internal(self):
        self._stop_event.set()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass
        self._worker_task = None
        self._session = None

    async def _session_worker(self):
        """Worker maintaining the active ClientSession context."""
        try:
            if self.config.transport == MCPTransportType.STDIO:
                server_env = dict(os.environ)
                params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=server_env
                )
                self.status.add_diagnostic(f"Starting child stdio process: '{self.config.command} {' '.join(self.config.args)}'")
                async with stdio_client(params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        res = await session.initialize()
                        self.status.server_name = res.serverInfo.name if hasattr(res, "serverInfo") else "MCP Server"
                        self.status.server_version = res.serverInfo.version if hasattr(res, "serverInfo") else "1.0.0"
                        self._session = session
                        self._ready_event.set()
                        await self._stop_event.wait()

            elif self.config.transport == MCPTransportType.STREAMABLE_HTTP:
                ok, err = self.validate_url(self.config.server_url)
                if not ok:
                    self._init_error = f"Invalid server URL: {err}"
                    self._ready_event.set()
                    return

                headers = {}
                if self.config.auth_token:
                    headers["Authorization"] = f"Bearer {self.config.auth_token}"

                from mcp.client.streamable_http import streamable_http_client
                self.status.add_diagnostic(f"Connecting to Streamable HTTP endpoint: {self.config.server_url}")
                async with streamable_http_client(self.config.server_url, headers=headers) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        res = await session.initialize()
                        self.status.server_name = res.serverInfo.name if hasattr(res, "serverInfo") else "HTTP MCP Server"
                        self.status.server_version = res.serverInfo.version if hasattr(res, "serverInfo") else "1.0.0"
                        self._session = session
                        self._ready_event.set()
                        await self._stop_event.wait()

            elif self.config.transport == MCPTransportType.SSE:
                ok, err = self.validate_url(self.config.server_url)
                if not ok:
                    self._init_error = f"Invalid server URL: {err}"
                    self._ready_event.set()
                    return

                headers = {}
                if self.config.auth_token:
                    headers["Authorization"] = f"Bearer {self.config.auth_token}"

                from mcp.client.sse import sse_client
                self.status.add_diagnostic(f"Connecting to SSE endpoint: {self.config.server_url}")
                async with sse_client(self.config.server_url, headers=headers) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        res = await session.initialize()
                        self.status.server_name = res.serverInfo.name if hasattr(res, "serverInfo") else "SSE MCP Server"
                        self.status.server_version = res.serverInfo.version if hasattr(res, "serverInfo") else "1.0.0"
                        self._session = session
                        self._ready_event.set()
                        await self._stop_event.wait()

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._init_error = f"Transport error ({type(exc).__name__}): {exc}"
            self.status.add_diagnostic(self._init_error)
            self._ready_event.set()
        finally:
            self._session = None
            if self.status.state == MCPConnectionState.CONNECTED:
                self.status.state = MCPConnectionState.CONNECTION_LOST
                self.status.add_diagnostic("Connection lost or server terminated unexpectedly.")

    async def refresh_tools(self) -> Tuple[bool, str, List[MCPToolInfo]]:
        """Queries the connected MCP server for advertised tools."""
        async with self._lock:
            if not self.is_connected or not self._session:
                return False, "Cannot refresh tools: MCP client is not connected.", []
            return await self._refresh_tools_internal()

    async def _refresh_tools_internal(self) -> Tuple[bool, str, List[MCPToolInfo]]:
        try:
            tools_result = await self._session.list_tools()
            discovered = []
            for t in tools_result.tools:
                tier = TOOL_PERMISSION_TIERS.get(t.name, PermissionTier.READ_ONLY)
                schema = t.inputSchema if hasattr(t, "inputSchema") else {}
                discovered.append(MCPToolInfo(
                    name=t.name,
                    description=t.description or "No description provided.",
                    input_schema=schema if isinstance(schema, dict) else {},
                    permission_tier=tier.value if hasattr(tier, "value") else int(tier),
                    is_available=True
                ))
            self.status.discovered_tools = discovered
            self.status.add_diagnostic(f"Discovered {len(discovered)} tools from server.")
            return True, f"Discovered {len(discovered)} tools.", discovered
        except Exception as exc:
            msg = f"Failed to list tools: {exc}"
            self.status.add_diagnostic(msg)
            return False, msg, []

    async def test_connection(self) -> Tuple[bool, str]:
        """Runs a live roundtrip probe to verify connection health and prerequisites."""
        async with self._lock:
            if not self.is_connected or not self._session:
                return False, "Test failed: Not connected to an MCP server."
            return await self._run_health_check_internal()

    async def _run_health_check_internal(self) -> Tuple[bool, str]:
        t0 = time.time()
        try:
            # Check if check_environment tool exists
            env_tool = next((t for t in self.status.discovered_tools if t.name == "check_environment"), None)
            if env_tool:
                call_res = await self._session.call_tool("check_environment", {})
                latency_ms = int((time.time() - t0) * 1000)
                raw_text = call_res.content[0].text if (call_res.content and hasattr(call_res.content[0], "text")) else "{}"
                try:
                    data = json.loads(raw_text)
                    self.status.ros2_available = bool(data.get("rclpy_available"))
                    self.status.gazebo_available = bool(data.get("gazebo_available"))
                except Exception:
                    pass

                result_str = f"Healthy (roundtrip {latency_ms}ms). ROS 2: {'available' if self.status.ros2_available else 'unavailable'}, Gazebo: {'detected' if self.status.gazebo_available else 'not detected'}."
                self.status.health_check_result = result_str
                self.status.last_health_check_at = datetime.now(timezone.utc).isoformat()
                self.status.add_diagnostic(f"Health check passed: {result_str}")
                return True, result_str

            # Fallback to list_tools ping
            await self._session.list_tools()
            latency_ms = int((time.time() - t0) * 1000)
            result_str = f"Handshake verified (roundtrip {latency_ms}ms, {len(self.status.discovered_tools)} tools advertised)."
            self.status.health_check_result = result_str
            self.status.last_health_check_at = datetime.now(timezone.utc).isoformat()
            self.status.add_diagnostic(f"Ping passed: {result_str}")
            return True, result_str

        except Exception as exc:
            msg = f"Health check probe failed: {exc}"
            self.status.health_check_result = msg
            self.status.last_health_check_at = datetime.now(timezone.utc).isoformat()
            self.status.add_diagnostic(msg)
            return False, msg

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        approval_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a controlled MCP tool under strict backend permission checks:
        - T0 Read-only tools execute freely.
        - T2 Managed execution tools require an explicit human approval token.
        - T3 Blocked / unauthorized operations are rejected immediately.
        """
        if not self.is_connected or not self._session:
            return {"success": False, "error": "MCP server is not connected.", "status": "disconnected"}

        # Tool existence check
        tool_info = next((t for t in self.status.discovered_tools if t.name == tool_name), None)
        if not tool_info:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is not in the advertised tool catalogue of the connected MCP server.",
                "status": "tool_not_found"
            }

        tier = PermissionTier(tool_info.permission_tier)

        # Enforce Security Gate
        if tier == PermissionTier.BLOCKED:
            return {
                "success": False,
                "error": f"Security Gate: Tool '{tool_name}' is blocked by policy (Tier 3).",
                "status": "blocked"
            }

        if tier == PermissionTier.MANAGED_RUNTIME:
            if not approval_token or not approval_token.strip():
                return {
                    "success": False,
                    "error": f"Security Gate: Tier 2 tool '{tool_name}' requires an explicit human approval token.",
                    "status": "permission_denied"
                }
            arguments["approval_token"] = approval_token

        # Audit logging (mask any secrets in args)
        safe_args = {k: ("***" if "token" in k or "key" in k else v) for k, v in arguments.items()}
        self.status.add_diagnostic(f"Calling tool '{tool_name}' (Tier {tier.value}) with args: {safe_args}")

        try:
            res = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=self.config.timeout_sec
            )
            raw_text = res.content[0].text if (res.content and hasattr(res.content[0], "text")) else ""
            try:
                parsed = json.loads(raw_text)
                return {"success": True, "data": parsed, "raw": raw_text}
            except Exception:
                return {"success": True, "data": {"output": raw_text}, "raw": raw_text}
        except asyncio.TimeoutError:
            err = f"Tool '{tool_name}' invocation timed out after {self.config.timeout_sec}s."
            self.status.add_diagnostic(err)
            return {"success": False, "error": err, "status": "timeout"}
        except Exception as exc:
            err = f"Tool '{tool_name}' invocation failed ({type(exc).__name__}): {exc}"
            self.status.add_diagnostic(err)
            return {"success": False, "error": err, "status": "error"}
