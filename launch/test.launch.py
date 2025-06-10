from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments for robot spawn position
    declared_args = [
        DeclareLaunchArgument('x', default_value='0'),
        DeclareLaunchArgument('y', default_value='0'),
        DeclareLaunchArgument('z', default_value='1'),
        DeclareLaunchArgument('yaw', default_value='0'),
    ]

    # File paths
    pkg_share = FindPackageShare('manipulatorws')
    xacro_file = PathJoinSubstitution([pkg_share, 'urdf', 'pa10_gazebo.urdf.xacro'])
    yaml_file = PathJoinSubstitution([pkg_share, 'config', 'control.yaml'])
    world_path = PathJoinSubstitution([pkg_share, 'worlds', 'pa10_world.world'])

    # Generate robot_description from xacro
    robot_description = Command(['xacro', ' ', xacro_file])

    # Set GZ plugin path so gz_ros2_control-system can be found
    set_gz_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/home/mmorteo/gz_ros2_control_ws/install/gz_ros2_control/lib'
    )

    # Node: Publish robot_description
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    # Gazebo simulator
    gz_world = ExecuteProcess(
        cmd=['gz', 'sim', world_path],
        output='screen'
    )

    # Controller Manager (ros2_control_node)
    controller_manager_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='controller_manager',
                executable='ros2_control_node',
                parameters=[
                    {'robot_description': robot_description},
                    yaml_file
                ],
                output='screen'
            )
        ]
    )

    # Spawn robot into Gazebo
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

    # Spawner: Joint State Broadcaster
    spawner_jsb = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'controller_manager', 'spawner',
                    'joint_state_broadcaster',
                    '--controller-manager', '/controller_manager'
                ],
                output='screen'
            )
        ]
    )

    # Spawner: Position Controller
    spawner_pos = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'controller_manager', 'spawner',
                    'position_controller',
                    '--controller-manager', '/controller_manager'
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription(
        declared_args + [
            set_gz_plugin_path,
            robot_state_publisher_node,
            controller_manager_node,
            spawn_entity,
            spawner_jsb,
            spawner_pos,
            gz_world
        ]
    )
