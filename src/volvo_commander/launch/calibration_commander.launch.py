"""
calibration_commander.launch.py
Launches calibration_commander with kinematics.yaml loaded so that
MoveGroupInterface can perform IK for Cartesian pose targets.

Usage:
  ros2 launch volvo_commander calibration_commander.launch.py
  ros2 launch volvo_commander calibration_commander.launch.py hover_z_offset:=0.07
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # Load the full MoveIt config — this includes kinematics.yaml
    moveit_config = (
        MoveItConfigsBuilder("volvo_arm", package_name="volvo_moveit")
        .to_moveit_configs()
    )

    # Declare overrideable args
    hover_arg = DeclareLaunchArgument(
        "hover_z_offset", default_value="0.05",
        description="Height above paper surface for hover moves (m)")
    vel_arg = DeclareLaunchArgument(
        "vel_scale", default_value="0.15",
        description="Velocity scaling factor")
    paper_z_arg = DeclareLaunchArgument(
        "paper_z", default_value="-0.006",
        description="Paper surface Z in base_link (m)")
    pause_arg = DeclareLaunchArgument(
        "pause_between_s", default_value="2.0",
        description="Pause at each position (s)")

    commander_node = Node(
        package="volvo_commander",
        executable="calibration_commander",
        name="calibration_commander",
        output="screen",
        parameters=[
            # These three give the node access to robot model + kinematics
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,  # ← the key one
            # Runtime params
            {
                "hover_z_offset":  LaunchConfiguration("hover_z_offset"),
                "vel_scale":       LaunchConfiguration("vel_scale"),
                "paper_z":         LaunchConfiguration("paper_z"),
                "pause_between_s": LaunchConfiguration("pause_between_s"),
            },
        ],
    )

    return LaunchDescription([
        hover_arg,
        vel_arg,
        paper_z_arg,
        pause_arg,
        commander_node,
    ])
