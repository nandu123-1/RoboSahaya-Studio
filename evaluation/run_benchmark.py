import json
import time
from pathlib import Path
from robosahaya.grounding import EntityGrounder
from robosahaya.mock_inspector import MockInspector
from robosahaya.diagnostics.engine import DiagnosticEngine
from robosahaya.command_catalogue import COMMAND_DEFINITIONS

def run_benchmark():
    fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    labelled_path = fixtures_dir / "labelled_cases.json"

    with open(labelled_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print("=" * 70)
    print("🎯 RoboSahaya Benchmark Evaluation Suite")
    print(f"Loaded {len(cases)} labelled diagnostic benchmark cases.")
    print("=" * 70)

    grounder = EntityGrounder()
    correct_diagnoses = 0
    traceable_evidence_count = 0
    safe_commands_count = 0
    grounding_success_count = 0
    latencies = []

    for idx, c in enumerate(cases, 1):
        case_id = c["case_id"]
        query = c["query_te_en"]
        expected_case = c["expected_case"]
        expected_obs = c["expected_observations"]

        t0 = time.perf_counter()

        # 1. Grounding
        ground_res = grounder.ground(query)
        if ground_res.primary_target and ground_res.primary_target.canonical_name == c["target_entity"]:
            grounding_success_count += 1

        # 2. Inspector
        # Map scenario
        scenario_map = {
            "case_d_qos_reliability_mismatch": "qos_reliability_mismatch",
            "case_a_missing_publisher": "missing_laser_pub",
            "case_c_tf_transform_failure": "tf_broken_camera_link",
            "case_e_sim_time_divergence": "sim_time_divergence",
            "clean_healthy_system": "clean_healthy_system"
        }
        scenario_file = scenario_map.get(case_id, "clean_healthy_system")
        inspector = MockInspector(scenario_name=scenario_file)
        engine = DiagnosticEngine(inspector)

        # 3. Diagnosis
        diag = engine.analyze(
            target_entity=c["target_entity"],
            entity_type=c["target_type"]
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        # Check accuracy
        is_diag_correct = diag.case_id == expected_case
        if is_diag_correct:
            correct_diagnoses += 1

        # Check evidence traceability
        all_obs_valid = all(obs in diag.observations for obs in expected_obs)
        if all_obs_valid:
            traceable_evidence_count += 1

        # Check command safety
        all_cmds_safe = all(cmd.command_id in COMMAND_DEFINITIONS for cmd in diag.terminal_card)
        if all_cmds_safe:
            safe_commands_count += 1

        status_icon = "✅" if is_diag_correct and all_obs_valid else "❌"
        print(f"[{idx}/{len(cases)}] {status_icon} Case: {c['name']:<38} | Latency: {elapsed_ms:5.2f} ms")

    accuracy = (correct_diagnoses / len(cases)) * 100
    evidence_rate = (traceable_evidence_count / len(cases)) * 100
    safety_compliance = (safe_commands_count / len(cases)) * 100
    grounding_rate = (grounding_success_count / len(cases)) * 100
    p50_latency = sorted(latencies)[len(latencies) // 2]
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    print("\n" + "=" * 70)
    print("📊 Final Benchmark Metrics (Diagnostic Copilot)")
    print("=" * 70)
    print(f"Top-1 Diagnostic Accuracy:       {accuracy:6.1f}%  (Target: ≥85%)")
    print(f"Evidence Traceability Rate:      {evidence_rate:6.1f}%  (Target: ≥95%)")
    print(f"Command-Safety Compliance:       {safety_compliance:6.1f}%  (Target: 100%)")
    print(f"Code-Mixed Grounding Accuracy:   {grounding_rate:6.1f}%  (Target: ≥80%)")
    print(f"Latency P50:                     {p50_latency:6.2f} ms")
    print(f"Latency P95:                     {p95_latency:6.2f} ms")
    print("=" * 70)


def run_studio_benchmark():
    print("\nStudio runtime benchmark: NOT RUN")
    print("This command evaluates replay fixtures only. Live workspace build, Gazebo launch, patch approval, and sensor retest require an explicit live session.")
    return

    import asyncio
    import tempfile
    from robosahaya.orchestrator import TaskOrchestrator
    from robosahaya.task_store import TaskStore
    from robosahaya.workspace_manager import WorkspaceManager
    from robosahaya.policy_gate import PolicyGate
    from robosahaya.report_generator import ReportGenerator
    from robosahaya.evidence_models import TaskStage

    print("\n" + "=" * 70)
    print("🚀 RoboSahaya Studio Autonomous Engineering Evaluation")
    print("=" * 70)

    async def _run():
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = Path(__file__).resolve().parent.parent
            templates_dir = repo_root / "simulation" / "templates"
            
            store = TaskStore(tmp_path / "tasks")
            ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)
            gate = PolicyGate(tmp_path / "ws")
            orch = TaskOrchestrator(task_store=store, workspace_manager=ws_mgr, mode="mock")

            task_id = "bench_studio_01"
            query = "Create a small differential-drive robot with LiDAR in Gazebo, launch it, test sensors, and fix any issue."
            events = []

            t0 = time.perf_counter()

            # 1. Pipeline execution with fault
            res1 = await orch.execute_studio_pipeline(
                task_id=task_id,
                query=query,
                inject_fault=True,
                on_event_cb=lambda e: events.append(e)
            )
            t_plan_diag = (time.perf_counter() - t0) * 1000

            stage1_ok = res1["stage"] == TaskStage.AWAITING_APPROVAL
            patch = res1.get("patch_proposal")
            token = res1.get("approval_token")
            print(f"[1/4] {'✅' if stage1_ok else '❌'} Autonomous Plan, Launch & Diagnosis Loop | Latency: {t_plan_diag:5.2f} ms")

            # 2. Security & PolicyGate enforcement
            sec_checks_passed = 0
            # Test A: Path traversal rejection
            path_ok = gate.validate_workspace_path(Path("../../../etc/passwd"))
            if not path_ok:
                sec_checks_passed += 1
            
            # Test B: Diff SHA tampering rejection
            fake_token = f"approved_{patch.patch_id}_000000000000"
            dec_fake = gate.verify_patch_approval(patch, fake_token, patch.unified_diff)
            if not dec_fake.allowed:
                sec_checks_passed += 1

            # Test C: Valid token accepted
            dec_valid = gate.verify_patch_approval(patch, token, patch.unified_diff)
            if dec_valid.allowed:
                sec_checks_passed += 1

            print(f"[2/4] {'✅' if sec_checks_passed == 3 else '❌'} PolicyGate Tier 1 & 2 Security Enforcement (3/3 checks passed)")

            # 3. Patch application and retest
            t1 = time.perf_counter()
            res2 = await orch.approve_and_retest(
                task_id=task_id,
                proposal=patch,
                approval_token=token,
                workspace_dir=res1["workspace_dir"],
                on_event_cb=lambda e: events.append(e)
            )
            t_retest = (time.perf_counter() - t1) * 1000
            retest_ok = res2["stage"] == TaskStage.COMPLETED and res2["retest_result"].passed is True
            print(f"[3/4] {'✅' if retest_ok else '❌'} Supervised Patch Application & Payload Retest | Latency: {t_retest:5.2f} ms")

            # 4. Report generation & audit persistence
            md_rep = ReportGenerator.generate_markdown_report(
                task_id=task_id,
                query=query,
                events=events,
                initial_test=res1["test_result"],
                diagnosis=res1["diagnosis"],
                patch_proposal=patch,
                patch_approved=True,
                retest_result=res2["retest_result"]
            )
            audit_json = ReportGenerator.generate_json_audit(
                task_id=task_id,
                query=query,
                events=events,
                initial_test=res1["test_result"],
                diagnosis=res1["diagnosis"],
                patch_proposal=patch,
                retest_result=res2["retest_result"]
            )
            report_ok = "RESOLVED & VERIFIED" in md_rep and len(audit_json["events"]) >= 10
            print(f"[4/4] {'✅' if report_ok else '❌'} Full Audit Event Log & Markdown Report Generation")

            print("\n" + "=" * 70)
            print("📊 Studio Engineering Loop Metrics")
            print("=" * 70)
            print("End-to-End Workflow Success:    not measured in fixture replay")
            print("PolicyGate Safety Violations:   not measured in fixture replay")
            print("Total Workflow Latency:         not measured")
            print("Audit Trail Completeness:       not measured")
            print("=" * 70)

    asyncio.run(_run())


if __name__ == "__main__":
    run_benchmark()
    run_studio_benchmark()
