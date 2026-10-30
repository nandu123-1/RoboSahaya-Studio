"""Unit tests for the authoritative Robot Execution Matrix.

Verifies:
1. Complete catalogue enumeration (175 canonical model IDs)
2. Distinction between complete robots (164) and standalone components (11)
3. Summary statistics calculations
4. Correct attribution of verified skills vs honest blockers
"""
from robosahaya.execution_matrix import (
    ComponentType,
    ExecutionReadiness,
    RobotExecutionMatrix,
    get_execution_matrix,
)


def test_execution_matrix_covers_full_catalogue():
    matrix_mgr = get_execution_matrix()
    matrix = matrix_mgr.generate_full_matrix(verify_live=False)

    assert len(matrix) == 175
    assert "rsk" in matrix
    assert "franka_panda" in matrix
    assert "ur5e" in matrix
    assert "unitree_go2" in matrix
    assert "skydio_x2" in matrix
    assert "so_arm100" in matrix


def test_component_type_classification():
    matrix_mgr = get_execution_matrix()

    # Complete robots
    assert matrix_mgr.classify_component_type("franka_panda", "robotic arm") == ComponentType.COMPLETE_ROBOT
    assert matrix_mgr.classify_component_type("rsk", "wheeled/mobile") == ComponentType.COMPLETE_ROBOT
    assert matrix_mgr.classify_component_type("unitree_go2", "quadruped") == ComponentType.COMPLETE_ROBOT
    assert matrix_mgr.classify_component_type("skydio_x2", "drone/aerial") == ComponentType.COMPLETE_ROBOT

    # Standalone components
    assert matrix_mgr.classify_component_type("robotiq_2f85", "gripper/end effector") == ComponentType.STANDALONE_COMPONENT
    assert matrix_mgr.classify_component_type("ability_hand", "gripper/end effector") == ComponentType.STANDALONE_COMPONENT
    assert matrix_mgr.classify_component_type("gripper", "other") == ComponentType.STANDALONE_COMPONENT


def test_summary_statistics_counts():
    matrix_mgr = get_execution_matrix()
    stats = matrix_mgr.get_summary_statistics()

    assert stats["total_canonical_models"] == 175
    assert stats["by_component_type"]["complete_robot"] == 164
    assert stats["by_component_type"]["standalone_component"] == 11
    assert stats["ready_models_count"] >= 6
    assert stats["blocked_models_count"] > 160
    assert len(stats["top_blockers"]) > 0


def test_ready_robots_have_verified_skills():
    matrix_mgr = get_execution_matrix()
    matrix = matrix_mgr.generate_full_matrix(verify_live=False)

    # RSK
    rsk = matrix["rsk"]
    assert rsk.execution_readiness == ExecutionReadiness.READY.value
    assert "set_wheel_velocity" in rsk.verified_skills
    assert "drive_forward" in rsk.verified_skills

    # Franka Panda
    panda = matrix["franka_panda"]
    assert panda.execution_readiness == ExecutionReadiness.READY.value
    assert "move_joints" in panda.verified_skills
    assert "gripper_control" in panda.verified_skills


def test_blocked_robots_have_explicit_blockers():
    matrix_mgr = get_execution_matrix()
    matrix = matrix_mgr.generate_full_matrix(verify_live=False)

    # Go2 walking blocked for WBC
    go2 = matrix["unitree_go2"]
    assert go2.execution_readiness == ExecutionReadiness.BLOCKED.value
    assert any("wbc" in b.lower() or "locomotion" in b.lower() for b in go2.exact_blockers)
    assert len(go2.remediation_steps) > 0

    # Skydio X2 blocked for aerodynamics/plugins
    drone = matrix["skydio_x2"]
    assert drone.execution_readiness == ExecutionReadiness.BLOCKED.value
    assert any("aerodynamics" in b.lower() or "rotor" in b.lower() or "flight" in b.lower() for b in drone.exact_blockers)
