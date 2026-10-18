import asyncio
from pathlib import Path
from robosahaya.orchestrator import TaskOrchestrator
from robosahaya.task_store import TaskStore
from robosahaya.workspace_manager import WorkspaceManager
from robosahaya.evidence_models import TaskStage
from robosahaya.report_generator import ReportGenerator

def test_studio_autonomous_repair_loop_e2e(tmp_path):
    async def _test():
        repo_root = Path(__file__).resolve().parent.parent.parent
        templates_dir = repo_root / "simulation" / "templates"
        
        task_store = TaskStore(tmp_path / "tasks")
        ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)
        orch = TaskOrchestrator(task_store=task_store, workspace_manager=ws_mgr, mode="mock")

        task_id = "studio_e2e_run_01"
        query = "నా robot launch ayyindi kani laser scan data ravatledu, test chesi fix chey."
        events = []

        # Phase 1: Launch autonomous loop with injected bridge fault
        res1 = await orch.execute_studio_pipeline(
            task_id=task_id,
            query=query,
            inject_fault=True,
            on_event_cb=lambda e: events.append(e)
        )

        assert res1["stage"] == TaskStage.AWAITING_APPROVAL
        assert "patch_proposal" in res1
        proposal = res1["patch_proposal"]
        assert proposal.patch_id == "patch_enable_laser_bridge"
        token = res1["approval_token"]
        assert token.startswith(f"approved_{proposal.patch_id}_")

        # Phase 2: Approve patch & retest
        res2 = await orch.approve_and_retest(
            task_id=task_id,
            proposal=proposal,
            approval_token=token,
            workspace_dir=res1["workspace_dir"],
            on_event_cb=lambda e: events.append(e)
        )

        assert res2["stage"] == TaskStage.COMPLETED
        assert res2["retest_result"].passed is True
        assert res2["patch_applied"] is True

        # Phase 3: Verify report generation
        md_report = ReportGenerator.generate_markdown_report(
            task_id=task_id,
            query=query,
            events=events,
            initial_test=res1["test_result"],
            diagnosis=res1["diagnosis"],
            patch_proposal=proposal,
            patch_approved=True,
            retest_result=res2["retest_result"],
            mode="simulation"
        )

        assert "# 🤖 RoboSahaya Studio" in md_report
        assert "RESOLVED & VERIFIED" in md_report
        assert "patch_enable_laser_bridge" in md_report

        # Phase 4: Verify task store persistence
        stored_events = task_store.get_events(task_id)
        assert len(stored_events) >= 10
        assert stored_events[-1].stage == TaskStage.COMPLETED
        assert task_store.get_current_stage(task_id) == TaskStage.COMPLETED

    asyncio.run(_test())
