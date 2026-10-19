"""Unit tests for the Robot Identity Resolver."""
import pytest
from robosahaya.identity_resolver import IdentityResolver, IdentityResolutionResult
from robosahaya.robot_registry import RobotRegistry, get_robot_registry


@pytest.fixture
def resolver():
    return IdentityResolver()


def test_resolve_canonical_id(resolver):
    result = resolver.resolve("rsk")
    assert result.resolved_id == "rsk"
    assert result.confidence == 1.0
    assert result.match_type == "canonical"

    result_diff = resolver.resolve("differential_drive")
    assert result_diff.resolved_id == "differential_drive"


def test_resolve_aliases(resolver):
    result = resolver.resolve("robot soccer kit")
    assert result.resolved_id == "rsk"
    assert result.confidence >= 0.95

    result_so101 = resolver.resolve("so-101")
    assert result_so101.resolved_id == "so_arm101"

    result_panda = resolver.resolve("panda arm")
    assert result_panda.resolved_id == "franka_panda"


def test_resolve_natural_language_prompts(resolver):
    result_rsk = resolver.resolve("Please build the rsk robot in simulation")
    assert result_rsk.resolved_id == "rsk"
    assert result_rsk.confidence >= 0.9

    result_so101 = resolver.resolve("I want to simulate the so 101 arm in tabletop world")
    assert result_so101.resolved_id == "so_arm101"

    result_diff = resolver.resolve("Spawn a wheeled base robot with lidar")
    assert result_diff.resolved_id == "differential_drive"


def test_ambiguous_category_requires_clarification(resolver):
    result = resolver.resolve("Build an arm for tabletop manipulation")
    assert result.requires_clarification is True
    assert result.resolved_id is None
    assert result.match_type == "ambiguous"
    candidate_ids = [cid for cid, _ in result.candidates]
    assert "so_arm101" in candidate_ids or "franka_panda" in candidate_ids
    assert result.clarification_message is not None


def test_unknown_query_does_not_silently_substitute(resolver):
    result = resolver.resolve("Build a toaster oven with wheels")
    assert result.resolved_id is None
    assert result.confidence == 0.0
    assert result.match_type == "none"
