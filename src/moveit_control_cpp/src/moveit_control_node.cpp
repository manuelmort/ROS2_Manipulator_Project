#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>



class MoveItControlNode : public rclcpp::Node 
{
 public:
   MoveItControlNode() : Node("moveit_control_node"){

    RCLCPP_INFO(this->get_logger(), "Starting MoveIt Control Node...");
  }

};

int main(int argc, char* argv[]){
  rclcpp::init(argc, argv);
  
  //Creating Node
  auto node = std::make_shared<MoveItControlNode>();

  //spin node to keep it alive 
  rclcpp::spin(node);

  rclcpp::shutdown();


 return 0;
}
