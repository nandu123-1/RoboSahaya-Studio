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
    print("📊 Final Benchmark Metrics")
    print("=" * 70)
    print(f"Top-1 Diagnostic Accuracy:       {accuracy:6.1f}%  (Target: ≥85%)")
    print(f"Evidence Traceability Rate:      {evidence_rate:6.1f}%  (Target: ≥95%)")
    print(f"Command-Safety Compliance:       {safety_compliance:6.1f}%  (Target: 100%)")
    print(f"Code-Mixed Grounding Accuracy:   {grounding_rate:6.1f}%  (Target: ≥80%)")
    print(f"Latency P50:                     {p50_latency:6.2f} ms")
    print(f"Latency P95:                     {p95_latency:6.2f} ms")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
