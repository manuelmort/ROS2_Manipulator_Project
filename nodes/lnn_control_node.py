#!/usr/bin/env python3
# OFFICIAL LNN CONTROL NODE FOR PA10 IN GAZEBO

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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
    np.array([0.1, -1.0, 0.0, 2.0, 0.0, 1.57, 0.0]),
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
        return error - 0.05

    reached_goal.terminal = True
    reached_goal.direction = -1
    return dynamics, reached_goal


class LNNControlNode(Node):
    def __init__(self):
        super().__init__('lnn_control_node')

        # ✅ Correct message type
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )
        self.timer_period = 0.01  # 100 Hz

        # Solve trajectories
        y_current = np.zeros(2 * n + m)
        t_span = (0, 10)
        t_eval = np.linspace(*t_span, 800)
        theta_traj_list = []

        for i, theta_goal in enumerate(theta_list):
            self.get_logger().info(f"🧭 Solving trajectory {i+1}/{len(theta_list)}")

            dynamics_fn, stop_event = dynamics_loop(theta_goal)
            sol = solve_ivp(
                dynamics_fn,
                t_span,
                y_current,
                t_eval=t_eval,
                events=stop_event,
                method='RK45',
                rtol=1e-6,
                atol=1e-9
            )

            theta_segment = sol.y[0:n, :].T
            theta_traj_list.append(theta_segment)
            y_current = sol.y[:, -1]

            time.sleep(1)

        self.theta_traj = np.vstack(theta_traj_list)
        self.step = 0
        self.timer = self.create_timer(self.timer_period, self.send_command)

    def send_command(self):
        if self.step >= len(self.theta_traj):
            self.get_logger().info("✅ LNN Simulation complete.")
            self.timer.cancel()
            return

        msg = JointTrajectory()
        msg.joint_names = [
            'S1', 'S2', 'S3', 'E1', 'E2', 'W1', 'W2',
            'finger_1_joint', 'finger_2_joint'
        ]

        point = JointTrajectoryPoint()
        point.positions = self.theta_traj[self.step].tolist() + [0.0, 0.0]  # pad finger joints
        point.time_from_start = rclpy.duration.Duration(seconds=0.01 * self.step).to_msg()

        msg.points = [point]
        self.publisher_.publish(msg)
        self.step += 1


def main(args=None):
    rclpy.init(args=args)
    node = LNNControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
