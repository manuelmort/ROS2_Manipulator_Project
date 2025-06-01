from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments for position
    declared_args = [
        DeclareLaunchArgument('x', default_value='0'),
        DeclareLaunchArgument('y', default_value='0'),
        DeclareLaunchArgument('z', default_value='1'),
        DeclareLaunchArgument('yaw', default_value='0'),
    ]

    # Paths
    pkg_share = FindPackageShare('manipulatorws')
    xacro_path = PathJoinSubstitution([pkg_share, 'urdf', 'pa10.urdf.xacro'])
    world_path = PathJoinSubstitution([pkg_share, 'worlds', 'pa10_world.world'])
    controller_yaml = PathJoinSubstitution([pkg_share, 'config', 'control.yaml'])

    # Robot Description
    robot_description = Command(['xacro ', xacro_path])

    # Node: Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    # Process: Launch Gazebo world
    gz_world = ExecuteProcess(
        cmd=['gz', 'sim', world_path],
        output='screen'
    )

    # Process: Spawn robot in Gazebo
    spawn_entity = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_sim', 'create',
            '--name', 'pa10',
            '--x', LaunchConfiguration('x'),
            '--y', LaunchConfiguration('y'),
            '--z', LaunchConfiguration('z'),
            '--Y', LaunchConfiguration('yaw'),
            '--topic', 'robot_description'
        ],
        output='screen'
    )

    # Node: Controller Manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[{'robot_description': robot_description}, controller_yaml],
        output='screen'
    )

    # Process: Spawner for joint_state_broadcaster
    spawner_jsb = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'controller_manager', 'spawner',
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager'
        ],
        output='screen'
    )

    # Process: Spawner for position_controller
    spawner_pos = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'controller_manager', 'spawner',
            'position_controller',
            '--controller-manager', '/controller_manager'
        ],
        output='screen'
    )

    return LaunchDescription(declared_args + [
        gz_world,
        robot_state_publisher,
        controller_manager,
        spawn_entity,
        spawner_jsb,
        spawner_pos
    ])
