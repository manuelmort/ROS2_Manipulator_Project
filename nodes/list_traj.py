import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import numpy as np

class TrajectoryManagerNode(Node):
    def __init__(self):
        super().__init__('trajectory_manager_node')

        # List of joint target configurations (angles in radians)
        self.target_list = [
            #np.array([0.1, -1.0, 0.0, 2.0, 0.0, -1.57, 0.0]),
            #np.array([-0.017, 0.487, 0.049, 1.977, 3.14072, 0.855 , 0.0]),
            np.array([-0.017, 0.753, 0.049, 1.098, 0.0072, -0.264 , 0.0]),
            np.array([-0.017, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),
            np.array([-0.017, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),
            #np.array([-0.017, 0.487, 0.049, 1.977, 3.14072, 0.855 , 0.0]),
            #np.array([-0.017, 0.753, 0.049, 1.098, -3.14072, 0.24 , 0.0]),
            #np.array([-0.017, 0.984, 0.049, 1.098, 3.1472, 0.545 , 0.0]),
            #np.array([-0.017, 0.753, 0.049, 1.098, 0.0072, -0.264 , 0.0]),
            #np.array([-0.017, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),
            #np.array([-0.017, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),

        ]

        self.sent_current_goal = False
        self.velocity_threshold = 0.05  # More lenient for simulation noise

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.target_pub = self.create_publisher(
            Float64MultiArray,
            '/next_target',
            10
        )
        self.status_sub = self.create_subscription(
            String,
            '/arm_controller/complete',
            self.status_callback,
            10
        )


        # Optional delayed start timer to kick off the first goal after 2 seconds
        self.start_timer = self.create_timer(2.0, self.delayed_start)
        self.has_sent_start = False

        self.get_logger().info("🚀 Trajectory Manager Node has started.")

    def delayed_start(self):
        if self.target_list and not self.sent_current_goal and not self.has_sent_start:
            next_target = self.target_list.pop(0)
            self.send_target(next_target)
            self.sent_current_goal = True
            self.has_sent_start = True
            self.get_logger().info("📦 Sent first target via startup timer.")
        # One-shot timer logic
        self.start_timer.cancel()

    def joint_state_callback(self, msg: JointState):
        # Log current velocities for debugging
        self.get_logger().debug(f"Joint velocities: {msg.velocity}")

        if self.robot_is_stopped(msg.velocity):
            if not self.sent_current_goal and self.target_list:
                next_target = self.target_list.pop(0)
                self.send_target(next_target)
                self.sent_current_goal = True
        else:
            self.sent_current_goal = False  # Reset flag to allow next goal

    def robot_is_stopped(self, velocities):
        return all(abs(v) < self.velocity_threshold for v in velocities)

    def send_target(self, joint_angles):
        msg = Float64MultiArray()
        msg.data = joint_angles
        self.target_pub.publish(msg)
        self.get_logger().info(f"🎯 Published next target: {joint_angles}")

    # new callback:
    def status_callback(self, msg: String):
        if msg.data == "Complete" and self.target_list:
            next_target = self.target_list.pop(0)
            self.send_target(next_target)
            self.sent_current_goal = True
            self.get_logger().info("✅ Received 'Complete'. Sent next target.")
def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
