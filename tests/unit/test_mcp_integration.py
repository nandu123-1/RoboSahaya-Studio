import asyncio
from pathlib import Path

from robosahaya.evidence_models import PermissionTier
from robosahaya.mcp.client_manager import MCPClientManager
from robosahaya.mcp.models import (
    MCPConnectionConfig,
    MCPConnectionState,
    MCPTransportType,
)
from robosahaya.orchestrator import TaskOrchestrator
from robosahaya.task_store import TaskStore
from robosahaya.workspace_manager import WorkspaceManager


def test_mcp_initial_disconnected_state():
    mgr = MCPClientManager()
    assert mgr.is_connected is False
    assert mgr.status.state == MCPConnectionState.DISCONNECTED
    assert len(mgr.status.discovered_tools) == 0


def test_mcp_secret_masking():
    cfg = MCPConnectionConfig(
        transport=MCPTransportType.STREAMABLE_HTTP,
        server_url="http://127.0.0.1:8000/mcp",
        auth_token="super_secret_bearer_token_12345"
    )
    safe = cfg.to_safe_dict()
    assert "super_secret_bearer_token_12345" not in safe["auth_token"]
    assert "***2345" in safe["auth_token"]


def test_mcp_stdio_connection_and_discovery():
    async def _test():
        mgr = MCPClientManager()
        ok, msg = await mgr.connect()
        assert ok is True
        assert mgr.is_connected is True
        assert mgr.status.state == MCPConnectionState.CONNECTED
        assert mgr.status.server_name is not None

        # Discover tools
        tool_names = [t.name for t in mgr.status.discovered_tools]
        assert "check_environment" in tool_names
        assert "list_ros_nodes" in tool_names
        assert "list_ros_topics" in tool_names
        assert "inspect_topic" in tool_names
        assert "get_gazebo_status" in tool_names
        assert "launch_managed_simulation" in tool_names
        assert "run_sensor_test" in tool_names
        assert "stop_managed_simulation" in tool_names

        # Test connection health check
        health_ok, health_msg = await mgr.test_connection()
        assert health_ok is True
        assert mgr.status.health_check_result is not None

        # Verify tool invocation: check_environment (Tier 0 Read-Only)
        env_res = await mgr.call_tool("check_environment", {})
        assert env_res["success"] is True

        # Verify Tier 2 permission check (must reject without approval token)
        t2_res = await mgr.call_tool(
            "launch_managed_simulation",
            {"template_id": "differential_drive", "world_id": "obstacle_world"}
        )
        assert t2_res["success"] is False
        assert t2_res["status"] == "permission_denied"

        # Verify Tier 2 tool with approval token
        t2_approved = await mgr.call_tool(
            "launch_managed_simulation",
            {"template_id": "differential_drive", "world_id": "obstacle_world"},
            approval_token="valid_human_approval_token_xyz"
        )
        assert t2_approved["success"] is True

        # Disconnect cleanly
        disc_ok, disc_msg = await mgr.disconnect()
        assert disc_ok is True
        assert mgr.is_connected is False
        assert mgr.status.state == MCPConnectionState.DISCONNECTED

    asyncio.run(_test())


def test_mcp_aware_planner_connected_vs_offline(tmp_path):
    async def _test():
        mgr = MCPClientManager()
        repo_root = Path(__file__).resolve().parent.parent.parent
        templates_dir = repo_root / "simulation" / "templates"
        store = TaskStore(tmp_path / "tasks")
        ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)
        orch = TaskOrchestrator(task_store=store, workspace_manager=ws_mgr, mode="mock")

        # 1. Offline planning (MCP disconnected)
        plan_offline = orch.formulate_plan(
            "task_offline",
            "Create a differential drive robot.",
            mcp_client=mgr
        )
        assert plan_offline.mcp_verified is False
        assert any("offline mode" in a for a in plan_offline.assumptions)

        # 2. Live MCP verified planning (MCP connected)
        await mgr.connect()
        try:
            plan_live = orch.formulate_plan(
                "task_live",
                "Create a differential drive robot.",
                mcp_client=mgr
            )
            assert plan_live.mcp_verified is True
            assert len(plan_live.mcp_verified_facts) > 0
            assert any("MCP Server" in f for f in plan_live.mcp_verified_facts)
        finally:
            await mgr.disconnect()

    asyncio.run(_test())


def test_mcp_invalid_url_safety():
    async def _test():
        mgr = MCPClientManager()
        cfg = MCPConnectionConfig(
            transport=MCPTransportType.STREAMABLE_HTTP,
            server_url="ftp://malicious.host/exploit"
        )
        ok, msg = await mgr.connect(cfg)
        assert ok is False
        assert "Invalid scheme" in msg

    asyncio.run(_test())
