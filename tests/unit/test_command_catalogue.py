import pytest
from robosahaya.command_catalogue import CommandCatalogue

def test_valid_command_retrieval():
    cmd = CommandCatalogue.get_command("topic_info", topic="/scan")
    assert cmd.command_id == "topic_info"
    assert cmd.raw_command == "ros2 topic info /scan --verbose"
    assert "QoS" in cmd.purpose

def test_command_arg_sanitization():
    # Valid characters should pass
    clean = CommandCatalogue.sanitize_arg("/robot_1/scan")
    assert clean == "/robot_1/scan"

    # Shell injection attempts should raise ValueError
    with pytest.raises(ValueError):
        CommandCatalogue.sanitize_arg("/scan; rm -rf /")

    with pytest.raises(ValueError):
        CommandCatalogue.sanitize_arg("/scan && cat /etc/passwd")

def test_invalid_command_id():
    with pytest.raises(KeyError):
        CommandCatalogue.get_command("arbitrary_hacked_command")
