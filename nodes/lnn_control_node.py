#!/usr/bin/env python3
# OFFICIAL LNN CONTROL NODE FOR PA10 IN GAZEBO

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray, String

from roboticstoolbox import DHRobot, RevoluteDH
from scipy.integrate import solve_ivp

import numpy as np
import time

# ========================
# Robot Definition: PA10
# ========================
pa10 = DHRobot([
    RevoluteDH(d=0.317, a=0.0, alpha=-np.pi/2, qlim=[-3.089, 3.089]),
    RevoluteDH(d=0.0,   a=0.0, alpha=np.pi/2,  qlim=[-1.64, 1.64]),
    RevoluteDH(d=0.45,  a=0.0, alpha=-np.pi/2, qlim=[-3.036, 3.036]),
    RevoluteDH(d=0.0,   a=0.0, alpha=np.pi/2,  qlim=[-2.39, 2.39]),
    RevoluteDH(d=0.48,  a=0.0, alpha=-np.pi/2, qlim=[-4.45, 4.45]),
    RevoluteDH(d=0.0,   a=0.0, alpha=np.pi/2,  qlim=[-2.878, 2.878]),
    RevoluteDH(d=0.07,  a=0.0, alpha=0,        qlim=[-2.878, 2.878]),
], name='PA10')

# ====================================
# Dynamics Definition & Simulation
# ====================================
n = 7  # joints
m = 6  # task space DOF

C1 = 2e-3 * np.eye(n)
C2 = 2e-3 * np.eye(m)
W = np.eye(n)

theta_list = [
    #np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0]),

    #np.array([0.1, -1.0, 0.0, 2.0, 0.0, -1.57, 0.0]),
    #np.array([-0.017, 0.487, 0.049, 1.977, 3.14072, 0.855 , 0.0]),

    np.array([-0.017, 0.753, 0.049, 1.098, 0.0072, -0.264 , 0.0]),
    #np.array([-0.017, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),
    #np.array([-0.017, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),

    #np.array([-0.017, 0.487, 0.049, 1.977, 3.14072, 0.855 , 0.0]),
    #np.array([-0.017, 0.753, 0.049, 1.098, -3.14072, 0.24 , 0.0]),
    #np.array([-0.017, 0.984, 0.049, 1.098, 3.1472, 0.545 , 0.0]),
    #np.array([-0.017, 0.753, 0.049, 1.098, 0.0072, -0.264 , 0.0]),
    #np.array([-0.017, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),
    #np.array([-0.017, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),

]

def dynamics_loop(theta_goal):
    def dynamics(t, y):
        theta = y[0:n]
        v = y[n:2*n]
        u = y[2*n:2*n + m]

        J = pa10.jacob0(theta)
        r_d_dot = J @ (theta_goal - theta)

        theta_dot = v
        J_pinv = np.linalg.pinv(J)
        N = np.eye(n) - J_pinv @ J
        null_term = -N @ (theta - theta_goal)

        v_dot = np.linalg.solve(C1, -W @ v - J.T @ u + null_term)
        u_dot = np.linalg.solve(C2, J @ v - r_d_dot)

        return np.concatenate([theta_dot, v_dot, u_dot])

    def reached_goal(t, y):
        theta = y[0:n]
        error = np.linalg.norm(theta - theta_goal)
        return error - 0.5

    reached_goal.terminal = True
    reached_goal.direction = -1
    return dynamics, reached_goal


from std_msgs.msg import Float64MultiArray

class LNNControlNode(Node):
    def __init__(self):
        super().__init__('lnn_control_node')

        self.trajectory_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.target_sub = self.create_subscription(Float64MultiArray, '/next_target', self.new_target_callback, 10)
        self.complete_pub = self.create_publisher(String, '/arm_controller/complete', 10)

        self.current_trajectory = None
        self.step = 0
        self.timer = self.create_timer(0.01, self.send_command)
        self.executing = False

    def new_target_callback(self, msg: Float64MultiArray):
        theta_goal = np.array(msg.data)
        self.get_logger().info(f"🎯 New target received: {theta_goal.tolist()}")

        y_current = np.zeros(2 * n + m)
        t_span = (0, 10)
        t_eval = np.linspace(*t_span, 500)

        dynamics_fn, stop_event = dynamics_loop(theta_goal)
        sol = solve_ivp(
            dynamics_fn, t_span, y_current, t_eval=t_eval,
            events=stop_event, method='RK45',
            rtol=1e-6, atol=1e-9
        )

        theta_segment = sol.y[0:n, :].T
        self.current_trajectory = theta_segment
        self.step = 0
        self.executing = True
        self.get_logger().info("🚀 Trajectory planning complete.")

    def send_command(self):
        if not self.executing or self.current_trajectory is None:
            return

        if self.step >= len(self.current_trajectory):
            self.executing = False
            self.get_logger().info("Trajectory execution complete.")

            msg = String()
            msg.data = "Complete"
            self.complete_pub.publish(msg)
            self.get_logger().info(f"Asking for next trajectory....")
            return

        msg = JointTrajectory()
        msg.joint_names = ['S1', 'S2', 'S3', 'E1', 'E2', 'W1', 'W2', 'finger_1_joint', 'finger_2_joint']
        point = JointTrajectoryPoint()
        point.positions = self.current_trajectory[self.step].tolist() + [0.0, 0.0]
        point.time_from_start = rclpy.duration.Duration(seconds=0.01 * self.step).to_msg()
        msg.points = [point]

        self.trajectory_pub.publish(msg)
        self.step += 1


def main(args=None):
    rclpy.init(args=args)
    node = LNNControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
