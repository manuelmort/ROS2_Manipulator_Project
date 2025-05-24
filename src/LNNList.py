#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import numpy as np

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

n = 7  # joints
m = 6  # task space DOF

C1 = 2e-3 * np.eye(n)
C2 = 2e-3 * np.eye(m)
W = np.eye(n)

theta_list = [
    np.array([0.1, -1.0, 0, 2.0, 0, 3.8, 0.0]),
   np.array([ 0.21353103, -1.22695623,  0.03454151,  2.14213994,  0.04101,    -0.9157795,
  0.27342854])
]
theta_goal = theta_list[0]  # Will be updated dynamically by the node

def dynamics(t, y):
    global theta_goal
    theta = y[0:n]
    v = y[n:2*n]
    u = y[2*n:2*n + m]

    J = pa10.jacob0(theta)
    r_d_dot = J @ (theta_goal - theta)
    theta_dot = v

    J_pinv = np.linalg.pinv(J)
    N = np.eye(n) - J_pinv @ J
    null_term = -N @ (theta - theta_goal)

    v_dot = np.zeros(n)
    for i in range(n):
        damping = -W[i, i] * v[i]
        jacobian_term = sum(J[j, i] * u[j] for j in range(m))
        v_dot[i] = (damping - jacobian_term + null_term[i]) / C1[i, i]

    u_dot = np.zeros(m)
    for j in range(m):
        task_velocity = sum(J[j, i] * v[i] for i in range(n))
        u_dot[j] = (task_velocity - r_d_dot[j]) / C2[j, j]

    return np.concatenate([theta_dot, v_dot, u_dot])

# =====================
# Run Simulation Once
# =====================
y0 = np.zeros(2 * n + m)
t_span = (0, 5)
t_eval = np.linspace(*t_span, 500)
sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

class LNNNode(Node):
    def __init__(self):
        super().__init__('lnn_sim_node')
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        self.timer_period = 0.01  # 100 Hz
        self.joint_names = ["S1", "S2", "S3", "E1", "E2", "W1", "W2"]

        self.theta_traj = sol.y[0:n, :].T  # shape (timesteps, 7)
        self.t_eval = t_eval
        self.step = 0

        # New variables for controlling goal switching
        self.theta_list = theta_list
        self.goal_index = 0
        self.goal_update_period = 2.5  # seconds
        self.last_goal_update_time = self.get_clock().now().nanoseconds / 1e9

        self.timer = self.create_timer(self.timer_period, self.publish_joint_state)

    def publish_joint_state(self):
        global theta_goal

        if self.step >= len(self.theta_traj):
            self.get_logger().info("✅ Simulation complete.")
            self.timer.cancel()
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_goal_update_time > self.goal_update_period:
            self.goal_index = min(self.goal_index + 1, len(self.theta_list) - 1)
            theta_goal = self.theta_list[self.goal_index]
            self.last_goal_update_time = current_time
            self.get_logger().info(f"🔄 Switched to theta_goal[{self.goal_index}]: {theta_goal}")

        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names
        joint_state.position = self.theta_traj[self.step].tolist()
        self.publisher_.publish(joint_state)

        self.step += 1

def main(args=None):
    rclpy.init(args=args)
    node = LNNNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()