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
#theta_goal = np.array([0.1 , -1.0, 0, 2.0, 0, 3.8, 0.0])
#theta_goal = np.array([-1.9, -1.10, -0.10, 2.14, -0.110, 3.8, 2.59270569])
theta_list = [
    
    #np.array([0.0, 0, 0, 0, 0, 0, 0.0]),
    np.array([0.1, -1.0, 0, 2.0, 0, 3.8, 0.0]),
    np.array([-0.017, 0.487, 0.049, 1.977, 3.14072, 0.855 , 0.0]),
    np.array([-0.017, 0.753, 0.049, 1.098, -3.14072, 0.24 , 0.0]),
    np.array([-0.017, 0.984, 0.049, 1.098, 3.1472, 0.545 , 0.0]),
    np.array([-0.017, 0.753, 0.049, 1.098, 0.0072, -0.264 , 0.0]),
    np.array([-0.017, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),
    np.array([-0.017, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),



    #np.array([1.57, 0.753, 0.049, 1.098, 0.0072, -0.264 , 0.0]),
    #np.array([1.57, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),
    #np.array([1.57, 0.487, 0.049, 1.977, 0.0072, -0.855 , 0.0]),
    #np.array([0.001, 0.001, 0.001, 0.001, 0.001, -0.001 , 0.001])

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
theta_traj_list = []


class LNNNode(Node):
    def __init__(self):
        super().__init__('lnn_sim_node')
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        self.timer_period = 0.01  # 100 Hz

        self.joint_names = [
            'S1', 'S2', 'S3', 'E1', 'E2', 'W1', 'W2',  # existing joints
            'finger_1_joint', 'finger_2_joint'        # new finger joints
        ]
        y_current = np.zeros(2 * n + m)
        t_span = (0, 10)  # Give enough time to converge
        t_eval = np.linspace(*t_span, 800)

        theta_traj_list = []

        import time
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

            y_current = sol.y[:, -1]  # New initial state for next goal

            time.sleep(1)  # ⏸️ Optional pause

        # Combine into full trajectory
        self.theta_traj = np.vstack(theta_traj_list)
        self.step = 0
        self.timer = self.create_timer(self.timer_period, self.publish_joint_state)

    def publish_joint_state(self):
        if self.step >= len(self.theta_traj):
            self.get_logger().info("✅ Simulation complete.")
            self.timer.cancel()
            return
        
        
        # Example: add static claw positions (e.g. open)
        claw_1_angle = np.sin(time.time()) * 0.02  # just an example animation
        claw_2_angle = np.sin(time.time()) * 0.02

        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names
        joint_state.position = self.theta_traj[self.step].tolist() + [claw_1_angle, claw_2_angle]
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