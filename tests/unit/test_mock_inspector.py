import pytest
from robosahaya.mock_inspector import MockInspector
from robosahaya.evidence_models import EvidenceStatus

def test_mock_inspector_scenario_loading():
    inspector = MockInspector(scenario_name="qos_reliability_mismatch")
    graph = inspector.inspect_graph()
    assert "/scan" in graph.topics
    assert graph.clock_active is True

    endpoints = inspector.inspect_topic_endpoints("/scan")
    assert endpoints.exists_in_graph is True
    assert len(endpoints.publishers) == 1
    assert endpoints.publishers[0].qos.reliability == "BEST_EFFORT"
    assert len(endpoints.subscriptions) == 1
    assert endpoints.subscriptions[0].qos.reliability == "RELIABLE"

def test_mock_inspector_tf_lookup_failure():
    inspector = MockInspector(scenario_name="tf_broken_camera_link")
    res = inspector.lookup_transform("base_link", "camera_link")
    assert res.transform_available is False
    assert res.status == EvidenceStatus.ERROR
    assert "camera_link" in res.error_message

def test_mock_inspector_sim_time():
    inspector = MockInspector(scenario_name="sim_time_divergence")
    p_slam = inspector.inspect_parameter("/slam_toolbox", "use_sim_time")
    assert p_slam.value is False

    p_gazebo = inspector.inspect_parameter("/gazebo", "use_sim_time")
    assert p_gazebo.value is True
