#!/usr/bin/env python3
#OFFICIAL LNN CODE WE CAN USE ATM
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
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

theta_list_left = [

    np.array([-0.017, 0.541, 0.049, 1.82, 0.024, -0.91 , 0.0]),    
 
]
theta_list_right = [
    
    np.array([0.017, -0.683, 0.049, -1.828, 0.024, 0.918, 0.0]),    


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
        return error - 0.05  # Stop integration when error < 0.05 radians

    reached_goal.terminal = True  # Stop when this event triggers
    reached_goal.direction = -1   # Trigger when value is decreasing through 0

    return dynamics, reached_goal


# =====================
# Run Simulation Once
# =====================



'''

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

'''

class LNNNode(Node):
    def __init__(self):
        super().__init__('lnn_sim_node')
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        self.timer_period = 0.01  # 100 Hz

        self.joint_names = [
            'left_S1', 'left_S2', 'left_S3',
            'left_E1', 'left_E2', 'left_W1', 'left_W2',
            'right_S1', 'right_S2', 'right_S3',
            'right_E1', 'right_E2', 'right_W1', 'right_W2',
            'left_finger_1_joint', 'left_finger_2_joint',
            'right_finger_1_joint', 'right_finger_2_joint'
        ]

        y_current_left = np.zeros(2 * n + m)
        y_current_right = np.zeros(2 * n + m)

        t_span = (0, 10)
        t_eval = np.linspace(*t_span, 1000)

        left_traj_segments = []
        right_traj_segments = []

        # Simulate left arm
        for i, theta_goal in enumerate(theta_list_left):
            self.get_logger().info(f"🧭 Solving LEFT trajectory {i+1}/{len(theta_list_left)}")
            dynamics_fn, stop_event = dynamics_loop(theta_goal)

            sol = solve_ivp(
                dynamics_fn,
                t_span,
                y_current_left,
                t_eval=t_eval,
                events=stop_event,
                method='RK45',
                rtol=1e-6,
                atol=1e-9
            )
            theta_segment = sol.y[0:n, :].T
            left_traj_segments.append(theta_segment)
            y_current_left = sol.y[:, -1]
            time.sleep(0.5)

        # Simulate right arm
        for i, theta_goal in enumerate(theta_list_right):
            self.get_logger().info(f"🧭 Solving RIGHT trajectory {i+1}/{len(theta_list_right)}")
            dynamics_fn, stop_event = dynamics_loop(theta_goal)

            sol = solve_ivp(
                dynamics_fn,
                t_span,
                y_current_right,
                t_eval=t_eval,
                events=stop_event,
                method='RK45',
                rtol=1e-6,
                atol=1e-9
            )
            theta_segment = sol.y[0:n, :].T
            right_traj_segments.append(theta_segment)
            y_current_right = sol.y[:, -1]
            time.sleep(0.5)

        # Stack both into one trajectory: [left | right]
        left_arm = np.vstack(left_traj_segments)
        right_arm = np.vstack(right_traj_segments)

        self.theta_traj_left = left_arm
        self.theta_traj_right = right_arm


        self.step = 0
        self.timer = self.create_timer(self.timer_period, self.publish_joint_state)

    def publish_joint_state(self):
        if self.step >= min(len(self.theta_traj_left), len(self.theta_traj_right)):
            self.get_logger().info("✅ Simulation complete.")
            self.timer.cancel()
            return


        claw_angle = np.sin(time.time()) * 0.02

        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()

        left_angles = self.theta_traj_left[self.step]
        right_angles = self.theta_traj_right[self.step]

        joint_state.name = self.joint_names
        joint_state.position = list(left_angles) + list(right_angles) + [0.0, 0.0, 0.0, 0.0]
        
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