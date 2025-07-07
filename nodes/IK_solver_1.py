#IK Solver used in Chapter 3 eq. (3.20) of "Complex Robotic Systems (Lecture Notes in Control and -- Dr Pasquale Chiacchio, Dr Stefano Chiaverini"
# Code implemented by Manuel Morteo 

import numpy as np

from roboticstoolbox import DHRobot, RevoluteDH
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt


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


q_not = np.zeros(n) # Initial theta
theta_d = np.array([-0.017, 0.753, 0.049, 1.098, 0.0072, -0.264 , 0.0]) # Desired theta

K = np.eye(n)
I = np.eye(n)
J = pa10.jacob0(q_not)
J_pinv = np.linalg.pinv(J)

print(np.size(J))

def ik_dynamics(t, q):

    J = pa10.jacob0(q)
    J_pinv = np.linalg.pinv(J)
    e = theta_d - q

    q0_dot = np.zeros(n)  # You can change this for null-space behavior
    q_dot = J_pinv @ (K @ e) + (I - J_pinv @ J) @ q0_dot # Eq. (3.20)

    return q_dot


# Integrator
"""
sol = solve_ivp(
    fun=ik_dynamics,
    t_span=(0, 5),        # simulate from t=0 to t=5 seconds
    y0=q_not,             # initial condition
    t_eval=np.linspace(0, 5, 500)  # sampling for evaluation
)


for i in range(n):
    plt.plot(sol.t, sol.y[i], label=f'q{i+1}')
plt.legend()
plt.title("Joint trajectories")
plt.xlabel("Time [s]")
plt.ylabel("Joint angle [rad]")
plt.grid()
plt.show()
"""