import pytest
from robosahaya.mock_inspector import MockInspector
from robosahaya.diagnostics.engine import DiagnosticEngine

def test_diagnose_qos_mismatch():
    inspector = MockInspector(scenario_name="qos_reliability_mismatch")
    engine = DiagnosticEngine(inspector)
    res = engine.analyze(target_entity="/scan", entity_type="topic")

    assert res.case_id == "case_d_qos_reliability_mismatch"
    assert "BEST_EFFORT" in res.issue_summary
    assert "RELIABLE" in res.issue_summary
    assert "E3" in res.observations
    assert len(res.hypotheses) > 0
    assert "walkie-talkie" in res.plain_language_analogy.lower() or "radio" in res.plain_language_analogy.lower()
    assert any(cmd.command_id == "topic_info" for cmd in res.terminal_card)

def test_diagnose_missing_publisher():
    inspector = MockInspector(scenario_name="missing_laser_pub")
    engine = DiagnosticEngine(inspector)
    res = engine.analyze(target_entity="/scan", entity_type="topic")

    assert res.case_id == "case_a_missing_publisher"
    assert "ZERO" in res.issue_summary
    assert "E1" in res.observations
    assert "E2" in res.observations

def test_diagnose_tf_failure():
    inspector = MockInspector(scenario_name="tf_broken_camera_link")
    engine = DiagnosticEngine(inspector)
    res = engine.analyze(target_entity="camera_link", entity_type="frame")

    assert res.case_id == "case_c_tf_transform_failure"
    assert "camera_link" in res.issue_summary
    assert any(cmd.command_id == "tf_view_frames" for cmd in res.terminal_card)

def test_diagnose_sim_time():
    inspector = MockInspector(scenario_name="sim_time_divergence")
    engine = DiagnosticEngine(inspector)
    res = engine.analyze(target_entity="use_sim_time", intent="sim_time_check")

    assert res.case_id == "case_e_sim_time_divergence"
    assert "/slam_toolbox" in res.issue_summary
    assert any(cmd.command_id == "param_get" for cmd in res.terminal_card)

def test_diagnose_healthy_system():
    inspector = MockInspector(scenario_name="clean_healthy_system")
    engine = DiagnosticEngine(inspector)
    res = engine.analyze(target_entity="/scan", entity_type="topic")

    assert res.case_id == "healthy_topic"
    assert "healthy" in res.issue_summary.lower()
