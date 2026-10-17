import pytest
from robosahaya.grounding import EntityGrounder
from robosahaya.mock_inspector import MockInspector
from robosahaya.diagnostics.engine import DiagnosticEngine
from robosahaya.response_builder import ResponseBuilder

def test_full_pipeline_telugu_english():
    # Spoken code-mixed complaint
    query = "నా robot tirugutondi kani laser scan data ravatledu, why?"
    
    # 1. Grounding
    grounder = EntityGrounder()
    ground_res = grounder.ground(query)
    assert ground_res.is_code_mixed is True
    assert ground_res.primary_target.canonical_name == "/scan"

    # 2. Inspector & Diagnosis
    inspector = MockInspector(scenario_name="qos_reliability_mismatch")
    engine = DiagnosticEngine(inspector)
    diag = engine.analyze(
        target_entity=ground_res.primary_target.canonical_name,
        entity_type=ground_res.primary_target.entity_type,
        intent=ground_res.inferred_intent
    )

    assert diag.case_id == "case_d_qos_reliability_mismatch"
    assert len(diag.observations) >= 3

    # 3. Response Building
    terminal_md = ResponseBuilder.format_terminal_markdown(diag)
    voice_script = ResponseBuilder.format_voice_script(diag)

    assert "Plain-Language Analogy" in terminal_md
    assert "ros2 topic info /scan --verbose" in terminal_md
    assert "QoS mismatch" in voice_script
    # Ensure raw CLI commands are omitted from voice script for Timbre
    assert "ros2 topic info" not in voice_script
