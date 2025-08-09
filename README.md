# 🤖 PA-10 Robotic Arm Simulation with Dynamic Control and RViz Visualization

[![ROS 2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blueviolet)](https://docs.ros.org/en/jazzy/)
[![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-E95420?logo=ubuntu)](https://releases.ubuntu.com/24.04/)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-blue.svg)](./LICENSE)

This repository simulates a Mitsubishi PA-10 robotic arm in **ROS 2 Jazzy** using custom dynamic control based on operational space and null-space projection. The system visualizes joint trajectories in **RViz** and **Gazebo** and can be extended to simulate object pickup via markers.

---

## PA-10 Rviz Control Simulation
![PA-10 Demonstration](assets/pa10example.gif)



## 📦 Features

- Symbolic + numerical dynamic model of the 7-DOF PA-10 arm using `roboticstoolbox` 
- Multi-goal joint trajectory execution using `solve_ivp`
- Joint state publishing for real-time RViz visualization
- Modular design with clear separation between dynamics and ROS node
- Marker-based extension support to simulate object grasping in RViz

---

## 🚀 Requirements

- ROS 2 Jazzy and Gazebo
- Ubuntu 24.04 with Python 3.12
- Python 3.11 (Virtual env) `python3.11 -m venv ~/venvs/my_venv` to run nodes that include roboticstoolbox-python, from here we can use pip to install our dependencies.

- Dependencies:
  - `rclpy`
  - `roboticstoolbox-python`
  - `spatialmath-python`
  - `scipy`
  - `matplotlib`
  - `numpy`

Install with:

```bash
pip install roboticstoolbox-python spatialmath scipy matplotlib numpy<2"
```

Notes: 

roboticstoolbox-python should be replaced in the future with better DH solvers out there if software doesn't get updated to be compatible with python 3.12.

## Installing Move-it
- sudo apt install ros-<your-distro>-moveit ros-<your-distro>-moveit-common

# Coming Soon
- Containerization using Docker
- Dual Manipulation 
### 
