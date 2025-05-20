from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import numpy as np

# Define your PA10 robot again if needed
pa10 = DHRobot([
    RevoluteDH(d=0.317, a=0, alpha=0),
    RevoluteDH(d=0, a=0, alpha=-np.pi/2),
    RevoluteDH(d=0.45, a=0, alpha=np.pi/2),
    RevoluteDH(d=0, a=0, alpha=-np.pi/2),
    RevoluteDH(d=0.48, a=0, alpha=np.pi/2),
    RevoluteDH(d=0, a=0, alpha=-np.pi/2),
    RevoluteDH(d=0.07, a=0, alpha=np.pi/2)
], name='PA10')

# Desired end-effector pose
T_goal = SE3(-0.8878, 0.3286, 0.169) * SE3.Rz(np.radians(30))

# Solve IK
solution = pa10.ikine_LM(T_goal)

# Check solution
if solution.success:
    print("\n\n IK joint solution:", solution.q)
else:
    print("\nIK failed to converge")



