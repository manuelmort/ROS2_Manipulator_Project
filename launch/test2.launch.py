from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                "config/control.yaml",        # Hardware interface (required)
            ],
            output="screen"
        )
    ])
