cmake_minimum_required(VERSION 3.8)
project(manipulatorws)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(controller_manager REQUIRED)
find_package(gz_ros2_control REQUIRED)
find_package(ros_gz REQUIRED)
find_package(ros_gz_sim REQUIRED)
find_package(ros2_control REQUIRED)
find_package(xacro REQUIRED)

install(DIRECTORY meshes urdf rviz launch worlds config
  DESTINATION share/${PROJECT_NAME}
)


install(
  PROGRAMS
    nodes/lnn_control_node.py
    nodes/list_traj.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS
    nodes/LNNrviz.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
