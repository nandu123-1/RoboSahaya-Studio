import pytest
from robosahaya.evon_adapter import EvonReasoningAdapter
from robosahaya.mock_inspector import MockInspector
from robosahaya.diagnostics.engine import DiagnosticEngine

def test_evon_ai_reasoning_qos_mismatch():
    inspector = MockInspector(scenario_name="qos_reliability_mismatch")
    engine = DiagnosticEngine(inspector)
    res = engine.analyze(target_entity="/scan", entity_type="topic", engine_mode="ai")

    assert res.case_id == "case_d_qos_reliability_mismatch"
    assert "Evon" in res.plain_language_analogy
    assert "Evon AI" in res.issue_summary
    assert len(res.hypotheses) > 0
    assert any(cmd.command_id == "topic_info" for cmd in res.terminal_card)

def test_evon_ai_reasoning_missing_publisher():
    inspector = MockInspector(scenario_name="missing_laser_pub")
    engine = DiagnosticEngine(inspector)
    res = engine.analyze(target_entity="/scan", entity_type="topic", engine_mode="ai")

    assert res.case_id == "case_a_missing_publisher"
    assert "Evon AI" in res.issue_summary
    assert "Evon" in res.plain_language_analogy

def test_evon_ai_reasoning_broken_tf():
    inspector = MockInspector(scenario_name="tf_broken_camera_link")
    engine = DiagnosticEngine(inspector)
    res = engine.analyze(target_entity="camera_link", entity_type="frame", engine_mode="ai")

    assert res.case_id == "case_c_tf_transform_failure"
    assert "Evon" in res.plain_language_analogy

def test_evon_prompt_formatting():
    adapter = EvonReasoningAdapter()
    user_prompt = adapter._build_user_prompt(
        target_entity="/scan",
        entity_type="topic",
        evidence=[],
        query_text="Why is scan data not arriving?"
    )
    assert "/scan" in user_prompt
    assert "evidence_observations" in user_prompt
