import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def launch_gz(context, *args, **kwargs):
    gz_share = get_package_share_directory("ros_gz_sim")
    headless_str = LaunchConfiguration("headless").perform(context).strip().lower()
    world_str = LaunchConfiguration("world").perform(context)
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if headless_str in ("true", "1", "yes") or not has_display:
        gz_args = f"-r -s {world_str}"
    else:
        gz_args = f"-r {world_str}"

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gz_sim.launch.py")),
            launch_arguments={"gz_args": gz_args}.items(),
        )
    ]

def generate_launch_description():
    pkg_dir = get_package_share_directory("robosahaya_simulation")
    urdf_file = os.path.join(pkg_dir, "urdf", "model.urdf")
    bridge_config = os.path.join(pkg_dir, "config", "bridge.yaml")
    world_file = os.path.join(pkg_dir, "worlds", "tabletop_world.sdf")
    if not os.path.exists(world_file):
        world_file = os.path.join(pkg_dir, "worlds", "obstacle_world.sdf")
    with open(urdf_file, "r", encoding="utf-8") as f:
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
        arguments=["--ros-args", "-p", f"config_file:={bridge_config}"],
        parameters=[{"use_sim_time": use_sim_time}]
    )

    # 3. Spawn SO-101 Robot on Tabletop directly from generated URDF file
    spawn_robot = Node(
        package="ros_gz_sim", executable="create", name="spawn_so101_arm",
        arguments=["-name", "so101_arm", "-file", urdf_file, "-x", "0.20", "-y", "0.0", "-z", "0.80"],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true", description="Use simulation clock"),
        DeclareLaunchArgument("world", default_value=world_file, description="SDF world file"),
        DeclareLaunchArgument("headless", default_value="false", description="Run Gazebo Sim in headless mode (-s)"),
        OpaqueFunction(function=launch_gz),
        robot_state_publisher_node,
        gz_bridge_node,
        TimerAction(period=2.0, actions=[spawn_robot]),
    ])
