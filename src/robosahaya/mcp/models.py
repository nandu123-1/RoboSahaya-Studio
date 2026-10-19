import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MCPConnectionState(str, Enum):
    NOT_CONFIGURED = "not_configured"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CONNECTION_LOST = "connection_lost"


class MCPTransportType(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"
    SSE = "sse"


class MCPConnectionConfig(BaseModel):
    transport: MCPTransportType = MCPTransportType.STDIO
    server_url: str = "http://127.0.0.1:8000/mcp"
    command: str = sys.executable
    args: List[str] = Field(default_factory=lambda: ["-m", "robosahaya.mcp.server"])
    auth_token: Optional[str] = None
    timeout_sec: float = 15.0

    def to_safe_dict(self) -> Dict[str, Any]:
        """Returns config dictionary with credentials safely masked."""
        data = self.model_dump()
        if self.auth_token:
            data["auth_token"] = "***" + self.auth_token[-4:] if len(self.auth_token) >= 4 else "***"
        return data


class MCPToolInfo(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    permission_tier: int = 0  # 0: Read-Only, 1: Workspace, 2: Managed sim/patch, 3: Blocked
    is_available: bool = True


class MCPConnectionStatus(BaseModel):
    state: MCPConnectionState = MCPConnectionState.DISCONNECTED
    transport: MCPTransportType = MCPTransportType.STDIO
    server_name: Optional[str] = None
    server_version: Optional[str] = None
    discovered_tools: List[MCPToolInfo] = Field(default_factory=list)
    last_connected_at: Optional[str] = None
    last_health_check_at: Optional[str] = None
    health_check_result: Optional[str] = None
    error_message: Optional[str] = None
    diagnostics_log: List[str] = Field(default_factory=list)
    ros2_available: Optional[bool] = None
    gazebo_available: Optional[bool] = None

    def add_diagnostic(self, message: str):
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.diagnostics_log.append(f"[{now_str}] {message}")
        if len(self.diagnostics_log) > 50:
            self.diagnostics_log.pop(0)
