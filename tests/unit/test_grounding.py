import pytest
from robosahaya.grounding import EntityGrounder

def test_phonetic_grounding_slash_can():
    grounder = EntityGrounder()
    res = grounder.ground("my robot is running but slash can data is missing")
    assert res.primary_target is not None
    assert res.primary_target.canonical_name == "/scan"
    assert res.primary_target.entity_type == "topic"

def test_phonetic_grounding_command_well():
    grounder = EntityGrounder()
    res = grounder.ground("are command well messages coming?")
    assert res.primary_target is not None
    assert res.primary_target.canonical_name == "/cmd_vel"

def test_telugu_code_mixed_detection():
    grounder = EntityGrounder()
    res = grounder.ground("నా robot tirugutondi kani laser scan data ravatledu, why?")
    assert res.is_code_mixed is True
    assert res.primary_target is not None
    assert res.primary_target.canonical_name == "/scan"

def test_tf_frame_grounding():
    grounder = EntityGrounder()
    res = grounder.ground("the base link frame transform is failing")
    assert res.primary_target is not None
    assert res.primary_target.canonical_name == "base_link"
    assert res.primary_target.entity_type == "frame"

def test_sim_time_intent():
    grounder = EntityGrounder()
    res = grounder.ground("there is a time mismatch error in navigation clock")
    assert res.inferred_intent == "sim_time_check"
