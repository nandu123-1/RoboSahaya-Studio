import json
from robosahaya.report_generator import ReportGenerator
from robosahaya.evidence_models import TaskStage, TaskEvent, PatchProposal, TestResult

def test_report_generator_markdown_and_json():
    events = [
        TaskEvent(
            task_id="task_001",
            event_id="event_001",
            timestamp="2026-10-19T02:00:00Z",
            stage=TaskStage.RECEIVED,
            severity="info",
            user_facing_message="Request received"
        ),
        TaskEvent(
            task_id="task_001",
            event_id="event_002",
            timestamp="2026-10-19T02:01:00Z",
            stage=TaskStage.COMPLETED,
            severity="success",
            user_facing_message="Workflow completed"
        )
    ]

    diff = "--- bridge.yaml\n+++ bridge.yaml\n@@ -1 +1 @@\n-old\n+new\n"
    proposal = PatchProposal(
        patch_id="patch_001",
        target_file="bridge.yaml",
        unified_diff=diff,
        explanation="Add missing bridge topic",
        verification_test_id="test_bridge"
    )

    retest = TestResult(
        test_id="test_bridge",
        passed=True,
        measured_values={"messages": 10},
        expected_criteria={"min_messages": 1},
        log_excerpt="Nominal throughput"
    )

    md = ReportGenerator.generate_markdown_report(
        task_id="task_001",
        query="Fix bridge",
        events=events,
        patch_proposal=proposal,
        patch_approved=True,
        retest_result=retest,
        mode="simulation"
    )

    assert "RoboSahaya Studio" in md
    assert "task_001" in md
    assert "RESOLVED & VERIFIED" in md
    assert "patch_001" in md
    assert "Nominal throughput" in md
    assert "vach_" not in md

    audit = ReportGenerator.generate_json_audit(
        task_id="task_001",
        query="Fix bridge",
        events=events,
        patch_proposal=proposal,
        retest_result=retest
    )

    assert audit["task_id"] == "task_001"
    assert len(audit["events"]) == 2
    assert audit["patch_proposal"]["patch_id"] == "patch_001"
