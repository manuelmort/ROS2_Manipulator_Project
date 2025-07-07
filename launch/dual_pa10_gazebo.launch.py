from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command

def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument('x', default_value='0'),
        DeclareLaunchArgument('y', default_value='0'),
        DeclareLaunchArgument('z', default_value='0.01'),
        DeclareLaunchArgument('yaw', default_value='0'),
    ]

    pkg_share = FindPackageShare('manipulatorws')
    xacro_path = PathJoinSubstitution([pkg_share, 'urdf', 'dual_pa10.urdf.xacro'])
    yaml_file = PathJoinSubstitution([pkg_share, 'config', 'control.yaml'])
    world_path = PathJoinSubstitution([pkg_share, 'worlds', 'pa10_world.world'])
    robot_description = Command(['xacro ', ' ', xacro_path])
    lnn_script_path = PathJoinSubstitution([pkg_share,'nodes','lnn_control_node.py'])


    gz_world = ExecuteProcess(
        cmd=['gz', 'sim','-r', world_path],
        output='screen'
    )

    set_gz_plugin_env = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib'
    )

    set_ld_library_path = SetEnvironmentVariable(
        name='LD_LIBRARY_PATH',
        value='/opt/ros/jazzy/lib'
    )


    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}, {'use_sim_time': True}], 
        output='screen'
    )

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
    spawner_jsb = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster'],
        output='screen'
    )

    spawner_arm = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'arm_controller'],
        output='screen'
    )

    spawner_gripper = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'gripper_action_controller'],
        output='screen'
    )
    



    lnn_control_node = Node(
        package='manipulatorws',
        executable='lnn_control_node.py',  # This matches your script name
        name='lnn_control_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    load_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[spawner_jsb],
        )
    )

    load_arm = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawner_jsb,
            on_exit=[spawner_arm],
        )
    )

    load_gripper = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawner_arm,
            on_exit=[spawner_gripper],
        )
    )
    '''
    launch_lnn_node = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawner_arm,
            on_exit=[lnn_control_node],
        )
    )
    '''
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen'
    )

    return LaunchDescription(declared_args + [
        set_gz_plugin_env,
        gz_world,
        robot_state_publisher,
        spawn_entity,
        load_jsb,
        load_arm,
        #load_gripper,
        #lnn_control_node,
        bridge
    ])
