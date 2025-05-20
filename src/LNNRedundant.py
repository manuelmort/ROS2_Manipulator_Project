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

# ====================================
# Dynamics Definition & Simulation
# ====================================
n = 7  # joints
m = 6  # task space DOF

C1 = 2e-3 * np.eye(n)
C2 = 2e-3 * np.eye(m)
W = np.eye(n)

theta_goal = np.array([0.1 , -1.0, 0, 1.57, 0, 4.0, 0.0])

def dynamics(t, y):
    theta = y[0:n]
    v = y[n:2*n]
    u = y[2*n:2*n + m]

    J = pa10.jacob0(theta)
    r_d_dot = J @ (theta_goal - theta)

    theta_dot = v

    # Null-space projection matrix
    J_pinv = np.linalg.pinv(J)
    N = np.eye(n) - J_pinv @ J

    # Add null-space stabilization toward theta_goal
    null_term = -N @ (theta - theta_goal)

    v_dot = np.linalg.solve(C1, -W @ v - J.T @ u + null_term)
    u_dot = np.linalg.solve(C2, J @ v - r_d_dot)

    return np.concatenate([theta_dot, v_dot, u_dot])

# =====================
# Run Simulation Once
# =====================
y0 = np.zeros(2 * n + m)
t_span = (0, 5)
t_eval = np.linspace(*t_span, 500)

sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

# ========================
# Plot Joint Angle Errors
# ========================
theta_traj = sol.y[0:n, :].T  # shape: (timesteps, 7)
errors = theta_goal - theta_traj

plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(t_eval, errors[:, i], label=f'Joint {i+1}')
plt.xlabel("Time [s]")
plt.ylabel("Joint Angle Error [rad]")
plt.title("Joint Angle Errors Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



class LNNNode(Node):
    def __init__(self):
        super().__init__('lnn_sim_node')
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        self.timer_period = 0.01  # 100 Hz
        self.joint_names = ["S1", "S2", "S3", "E1", "E2", "W1", "W2"]

        self.theta_traj = sol.y[0:n, :].T  # shape (timesteps, 7)
        self.t_eval = t_eval
        self.step = 0

        self.timer = self.create_timer(self.timer_period, self.publish_joint_state)

    def publish_joint_state(self):
        if self.step >= len(self.theta_traj):
            self.get_logger().info("✅ Simulation complete.")
            self.timer.cancel()
            return

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
