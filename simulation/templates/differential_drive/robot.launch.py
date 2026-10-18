import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_file = os.path.join(pkg_dir, "model.urdf")
    bridge_config = os.path.join(pkg_dir, "bridge.yaml")

    with open(urdf_file, "r") as f:
        robot_desc = f.read()

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # 1. Robot State Publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_desc,
            "use_sim_time": use_sim_time
        }]
    )

    # 2. ROS-GZ Bridge
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        parameters=[{
            "config_file": bridge_config,
            "use_sim_time": use_sim_time
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true", description="Use simulation clock"),
        robot_state_publisher_node,
        gz_bridge_node
    ])
