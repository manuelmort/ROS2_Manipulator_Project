#!/usr/bin/env python3

from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Set package and file paths
    urdf_package = 'manipulatorws'
    urdf_filename = 'pa10.urdf.xacro'
    rviz_filename = 'manipulator.rviz'

    # Resolve full paths
    urdf_path = PathJoinSubstitution([FindPackageShare(urdf_package), 'urdf', urdf_filename])
    rviz_path = PathJoinSubstitution([FindPackageShare(urdf_package), 'rviz', rviz_filename])

    # Run xacro to generate robot_description
    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    # robot_state_publisher node
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # rviz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_path]
    )

    return LaunchDescription([rsp_node, rviz_node])

