from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import numpy as np

# -------------------------------
# 1. Define the PA10 robot (7 DOF)
# -------------------------------
pa10 = DHRobot([
    RevoluteDH(d=0.317, a=0.0, alpha=-np.pi/2, qlim=[-3.089, 3.089]),
    RevoluteDH(d=0.0,   a=0.0, alpha=np.pi/2,  qlim=[-1.64, 1.64]),
    RevoluteDH(d=0.45,  a=0.0, alpha=-np.pi/2, qlim=[-3.036, 3.036]),
    RevoluteDH(d=0.0,   a=0.0, alpha=np.pi/2,  qlim=[-2.39, 2.39]),
    RevoluteDH(d=0.48,  a=0.0, alpha=-np.pi/2, qlim=[-4.45, 4.45]),
    RevoluteDH(d=0.0,   a=0.0, alpha=np.pi/2,  qlim=[-2.878, 2.878]),
    RevoluteDH(d=0.07,  a=0.0, alpha=0,        qlim=[-2.878, 2.878]),
], name='PA10')

# -------------------------------
# 2. Define a desired end-effector pose
# Position (x, y, z) and orientation (30° rotation about Z-axis)
# -------------------------------
T_goal = SE3(-0.04523, 0.00446, 0.8311) * SE3.Rz(np.radians(30))
print("Desired pose (T_goal):\n", T_goal)

# -------------------------------
# 3. Solve IK using ikine_min (supports redundancy)
# -------------------------------
solution = pa10.ikine_QP(T_goal)

# -------------------------------
# 4. Evaluate the solution
# -------------------------------
if solution.success:
    q_sol = solution.q
    print("\n✅ IK Solution Found!")
    print("Joint angles (rad):", q_sol)

    # Compute forward kinematics
    T_fk = pa10.fkine(q_sol)
    print("\nResulting FK Pose:\n", T_fk)

    # -------------------------------
    # 5. Error Analysis
    # -------------------------------
    pos_error = np.linalg.norm(T_goal.t - T_fk.t)
    R_error = T_goal.R @ T_fk.R.T
    angle_error = np.arccos((np.trace(R_error) - 1) / 2)

    print("\nPosition error (m):", pos_error)
    print("Orientation error (rad):", angle_error)

else:
    print("\n❌ IK failed to converge")
