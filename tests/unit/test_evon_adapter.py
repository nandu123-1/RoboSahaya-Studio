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

def test_evon_local_model_verification():
    adapter = EvonReasoningAdapter()
    info = adapter.get_local_model_info()
    assert info["model_name"] == "gnani-evon-v3.3"
    assert info["expected_shards"] == 13
    assert info["shard_count"] == 13
    assert info["weights_present"] is True
    assert info["has_tokenizer"] is True
    assert info["vocab_size"] == 163072

def test_evon_local_tokenizer_multilingual():
    adapter = EvonReasoningAdapter()
    # English
    en_tokens = adapter.count_tokens("Why is the robot unable to receive /scan data?")
    assert en_tokens > 0
    # Telugu
    te_tokens = adapter.count_tokens("నా robot tirugutondi kani laser scan data ravatledu")
    assert te_tokens > 0

def test_evon_format_chatml():
    adapter = EvonReasoningAdapter()
    chatml = adapter.format_chatml_prompt("System prompt", "User query")
    assert "<|im_start|>system" in chatml
    assert "<|im_start|>user" in chatml
    assert "<|im_start|>assistant" in chatml
