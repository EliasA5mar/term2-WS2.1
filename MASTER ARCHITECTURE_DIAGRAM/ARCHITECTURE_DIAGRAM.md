# UR10e Bottle Flip - System Architecture & Workflow

##  COMPLETE SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              WINDOWS HOST MACHINE                                       │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              WSL2 (Ubuntu 22.04)                                   │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                         DOCKER CONTAINER                                     │  │ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │  │ │
│  │  │  │   ROS2 Humble   │  │  MoveIt2        │  │  UR Robot       │               │  │ │
│  │  │  │   + RVIZ2       │  │  Motion Planner │  │  Driver         │               │  │ │
│  │  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘               │  │ │
│  │  │           │                    │                    │                        │  │ │
│  │  │           └────────────────────┼────────────────────┘                        │  │ │
│  │  │                                │                                             │  │ │
│  │  │  ┌─────────────────────────────▼─────────────────────────────────┐           │  │ │
│  │  │  │              SIMULATION ENVIRONMENT                           │           │  │ │
│  │  │  │  ┌─────────────────┐        ┌─────────────────┐               │           │  │ │
│  │  │  │  │   PyBullet      │◄──────►│   PyTorch       │               │           │  │ │
│  │  │  │  │   Physics Sim   │        │   RL Training   │               │           │  │ │
│  │  │  │  │   (Bottle+Robot)│        │   (Policy Net)  │               │           │  │ │
│  │  │  │  └─────────────────┘        └─────────────────┘               │           │  │ │
│  │  │  └───────────────────────────────────────────────────────────────┘           │  │ │
│  │  │                                │                                             │  │ │
│  │  └────────────────────────────────┼─────────────────────────────────────────────┘  │ │
│  │                                   │ Docker Network Bridge                          │ │
│  └───────────────────────────────────┼────────────────────────────────────────────────┘ │
│                                      │ WSL2 Virtual Network                             │
└──────────────────────────────────────┼──────────────────────────────────────────────────┘
                                       │
                                       │ Ethernet / USB
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         REAL HARDWARE                │
                    │  ┌────────────────────────────────┐  │
                    │  │   UR10e Robot Controller       │  │
                    │  │   IP: 192.168.1.xxx            │  │
                    │  │   Port: 50002 (RTDE)           │  │
                    │  └────────────────────────────────┘  │
                    │  ┌────────────────────────────────┐  │
                    │  │   Intel RealSense Camera       │  │
                    │  │   (Bottle Detection)           │  │
                    │  └────────────────────────────────┘  │
                    │  ┌────────────────────────────────┐  │
                    │  │   Robotiq 2F-85 Gripper        │  │
                    │  │   (Bottle Grasping)            │  │
                    │  └────────────────────────────────┘  │
                    └──────────────────────────────────────┘
```

---

##  DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   DATA FLOW                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
    │   CAMERA     │         │  PERCEPTION  │         │   GRASPING   │
    │  (RealSense) │────────►│   MODULE     │────────►│   MODULE     │
    │              │  RGB-D  │ (Detection)  │  Pose   │  (Planning)  │
    └──────────────┘  Image  └──────────────┘ Estimate└──────────────┘
                                                              │
                                                              │ Grasp
                                                              │ Config
                                                              ▼
    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
    │   UR10e      │         │   MOTION     │         │   PHYSICS    │
    │   ROBOT      │◄────────│   GENERATION │◄────────│   MODULE     │
    │              │  Joint  │   MODULE     │ Release │  (PyBullet)  │
    └──────────────┘  Traj   └──────────────┘ Params  └──────────────┘
          │                                                   ▲
          │                                                   │
          │ Joint States                                      │ Sim
          ▼                                                   │ Results
    ┌──────────────┐         ┌──────────────┐                 │
    │   RVIZ2      │         │  EVALUATION  │─────────────────┘
    │  (Visualize) │◄────────│   & LEARNING │
    │              │ Display │   (PyTorch)  │
    └──────────────┘         └──────────────┘
```

---

##  HARDWARE REQUIREMENTS

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              HARDWARE REQUIREMENTS                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  DEVELOPMENT MACHINE (Windows PC with WSL2)                                             │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  MINIMUM SPECS:                          │  RECOMMENDED SPECS:                          │
│  • CPU: Intel i5 / AMD Ryzen 5           │  • CPU: Intel i7 / AMD Ryzen 7              │
│  • RAM: 16 GB                            │  • RAM: 32 GB                               │
│  • GPU: NVIDIA GTX 1060 (6GB VRAM)       │  • GPU: NVIDIA RTX 3070+ (8GB+ VRAM)        │
│  • Storage: 50 GB SSD                    │  • Storage: 100 GB NVMe SSD                 │
│  • OS: Windows 10/11 with WSL2           │  • OS: Windows 11 with WSL2                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  ROBOT HARDWARE                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  ESSENTIAL:                              │  OPTIONAL:                                   │
│  • UR10e Robotic Arm                     │  • Force/Torque Sensor (FT300)              │
│  • UR Control Box (CB-Series)            │  • External PC for real-time control        │
│  • Ethernet Cable (Cat6)                 │  • Safety Enclosure                         │
│  • 24V Power Supply                      │  • Emergency Stop Button (external)         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  GRIPPER OPTIONS:                        │  CAMERA OPTIONS:                            │
│  • Robotiq 2F-85 (Recommended)           │  • Intel RealSense D435i (Recommended)      │
│  • Robotiq 2F-140                        │  • Intel RealSense D455                     │
│  • OnRobot RG2/RG6                       │  • ZED 2 Stereo Camera                      │
│  • Custom pneumatic gripper              │  • Standard USB Webcam (basic)              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  NETWORK SETUP                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   [Windows PC]                    [Network Switch]                    [UR10e Robot]    │
│   192.168.1.100  ◄──────────────► (Gigabit)       ◄──────────────►  192.168.1.101     │
│                                        │                                               │
│                                        ▼                                               │
│                                 [RealSense Camera]                                      │
│                                 (USB 3.0 to PC)                                        │
│                                                                                         │
│   PORTS USED:                                                                          │
│   • 50001: Primary Interface                                                           │
│   • 50002: Secondary Interface (RTDE)                                                  │
│   • 50003: Real-time Interface                                                         │
│   • 29999: Dashboard Server                                                            │
│   • 30001-30003: Script commands                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

##  SOFTWARE STACK LAYERS

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              SOFTWARE STACK                                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

    Layer 7: APPLICATION
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  bottle_flip_ur10e.py  │  blue_bottle_detection.py  │  training_policy.py      │
    │  (Motion Control)      │  (Computer Vision)         │  (RL Learning)           │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                            │
    Layer 6: FRAMEWORKS                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  PyTorch 2.0+  │  OpenCV 4.x  │  NumPy  │  SciPy  │  Matplotlib              │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                            │
    Layer 5: SIMULATION                     ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  PyBullet        │  Gazebo (optional)  │  Isaac Sim (optional)                 │
    │  (Physics Sim)   │  (Full Sim)         │  (GPU Accelerated)                    │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                            │
    Layer 4: ROS2 ECOSYSTEM                 ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  MoveIt2         │  RVIZ2            │  ros2_control    │  tf2                 │
    │  (Motion Plan)   │  (Visualization)  │  (Controllers)   │  (Transforms)        │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                            │
    Layer 3: ROS2 CORE                      ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  ROS2 Humble    │  rclpy/rclcpp  │  std_msgs  │  sensor_msgs  │  geometry_msgs │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                            │
    Layer 2: ROBOT DRIVERS                  ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  ur_robot_driver  │  robotiq_driver  │  realsense2_camera  │  joint_state_pub  │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                            │
    Layer 1: CONTAINER/OS                   ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Docker Container  │  Ubuntu 22.04  │  WSL2 Kernel  │  NVIDIA Container Toolkit │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                            │
    Layer 0: HOST                           ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Windows 11  │  WSL2  │  NVIDIA GPU Driver  │  Docker Desktop                   │
    └─────────────────────────────────────────────────────────────────────────────────┘
```

---

##  COMPLETE WORKFLOW PIPELINE

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           DEVELOPMENT WORKFLOW PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════════════════╗
║  PHASE 1: ENVIRONMENT SETUP                                                            ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                        ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            ║
║  │  Install    │───►│  Install    │───►│  Install    │───►│  Build      │            ║
║  │  WSL2       │    │  Docker     │    │  NVIDIA     │    │  Docker     │            ║
║  │  + Ubuntu   │    │  Desktop    │    │  Container  │    │  Image      │            ║
║  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘            ║
║                                                                                        ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
                                         │
                                         ▼
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║  PHASE 2: SIMULATION DEVELOPMENT                                                       ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                        ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            ║
║  │  Start      │───►│  Load UR10e │───►│  Test       │───►│  Visualize  │            ║
║  │  Docker     │    │  in         │    │  Basic      │    │  in         │            ║
║  │  Container  │    │  PyBullet   │    │  Motions    │    │  RVIZ2      │            ║
║  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘            ║
║                                                                                        ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
                                         │
                                         ▼
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║  PHASE 3: TRAINING (RL LOOP)                                                           ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                        ║
║  ┌───────────────────────────────────────────────────────────────────────────────┐   ║
║  │                                                                               │   ║
║  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │   ║
║  │  │  PyBullet   │───►│  Execute    │───►│  Evaluate   │───►│  Update     │   │   ║
║  │  │  Reset      │    │  Flip       │    │  Success    │    │  Policy     │   │   ║
║  │  │  Episode    │    │  Action     │    │  Reward     │    │  (PyTorch)  │   │   ║
║  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │   ║
║  │        ▲                                                         │           │   ║
║  │        └─────────────────────────────────────────────────────────┘           │   ║
║  │                              Training Loop                                    │   ║
║  └───────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                        ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
                                         │
                                         ▼
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║  PHASE 4: VALIDATION IN RVIZ                                                           ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                        ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            ║
║  │  Load       │───►│  Play       │───►│  Verify     │───►│  Fine-tune  │            ║
║  │  Trained    │    │  Trajectory │    │  Motion     │    │  Parameters │            ║
║  │  Policy     │    │  in RVIZ    │    │  Safety     │    │  if needed  │            ║
║  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘            ║
║                                                                                        ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
                                         │
                                         ▼
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║  PHASE 5: REAL ROBOT DEPLOYMENT                                                        ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                        ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            ║
║  │  Connect    │───►│  Test at    │───►│  Gradual    │───►│  Full       │            ║
║  │  to UR10e   │    │  10% Speed  │    │  Speed      │    │  Speed      │            ║
║  │  via RTDE   │    │  (Safety)   │    │  Increase   │    │  Execution  │            ║
║  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘            ║
║                                                                                        ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
```

---

##  DOCKER CONTAINER ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           DOCKER CONTAINER STRUCTURE                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  bottle-flip-ros2:latest                                                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  BASE IMAGE: nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04                                 │
│                     │                                                                   │
│                     ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │  ROS2 Humble Desktop Full                                                        │  │
│  │  • rclpy, rclcpp, std_msgs, sensor_msgs, geometry_msgs                          │  │
│  │  • rviz2, ros2_control, tf2                                                      │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                     │                                                                   │
│                     ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │  UR Robot Packages                                                               │  │
│  │  • ur_robot_driver, ur_description, ur_moveit_config                            │  │
│  │  • ur_controllers, ur_bringup                                                    │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                     │                                                                   │
│                     ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │  MoveIt2                                                                         │  │
│  │  • moveit_core, moveit_ros_planning, moveit_ros_move_group                      │  │
│  │  • moveit_servo (real-time control)                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                     │                                                                   │
│                     ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Python ML/Simulation Stack                                                      │  │
│  │  • PyTorch 2.0 (CUDA enabled)                                                   │  │
│  │  • PyBullet 3.x                                                                  │  │
│  │  • OpenCV 4.x, NumPy, SciPy                                                     │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                     │                                                                   │
│                     ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Workspace: /home/user/ros2_ws/src/bottle_flip/                                 │  │
│  │  • bottle_flip_ur10e.py                                                         │  │
│  │  • pybullet_simulation.py                                                       │  │
│  │  • training_policy.py                                                            │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  EXPOSED PORTS:                                                                         │
│  • 11311 (ROS Master - legacy)    • 6080 (NoVNC Web GUI)                              │
│  • 11345 (Gazebo)                 • 5900 (VNC)                                         │
│                                                                                         │
│  VOLUMES:                                                                               │
│  • /home/user/ros2_ws  ◄──► ./ros2_ws (host)                                          │
│  • /tmp/.X11-unix      ◄──► /tmp/.X11-unix (X11 forwarding)                           │
│  • /dev/video*         ◄──► Camera devices                                             │
│                                                                                         │
│  GPU ACCESS:                                                                            │
│  • --gpus all (NVIDIA Container Toolkit)                                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

##  MODULE INTERACTION DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           MODULE INTERACTION                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         MAIN CONTROLLER             │
                    │     (bottle_flip_ur10e.py)          │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│    PERCEPTION    │    │     GRASPING     │    │     PHYSICS      │
│     MODULE       │    │      MODULE      │    │     MODULE       │
│                  │    │                  │    │                  │
│ • Camera Input   │    │ • Grasp Planning │    │ • PyBullet Sim   │
│ • Bottle Detect  │    │ • Gripper Ctrl   │    │ • Trajectory Opt │
│ • Pose Estimate  │    │ • Force Control  │    │ • Release Calc   │
│                  │    │                  │    │                  │
│ Owner: Heleri    │    │ Owner: Arthur    │    │ Owner: Elias     │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         │  bottle_pose          │  grasp_config        │  release_params
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────────┐
                    │       MOTION GENERATION             │
                    │          MODULE                     │
                    │                                     │
                    │  • Trajectory Planning              │
                    │  • MoveIt2 Interface                │
                    │  • Joint Control                    │
                    │                                     │
                    │  Owner: Amaro                       │
                    └──────────────┬──────────────────────┘
                                   │
                                   │  joint_trajectory
                                   ▼
                    ┌─────────────────────────────────────┐
                    │       EVALUATION & LEARNING         │
                    │          MODULE                     │
                    │                                     │
                    │  • Success Detection                │
                    │  • Reward Calculation               │
                    │  • Policy Update (PyTorch)          │
                    │                                     │
                    │  Owner: Team                        │
                    └─────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  ROS2 TOPICS & SERVICES                                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  TOPICS:                                     SERVICES:                                  │
│  ├── /joint_states                          ├── /compute_ik                            │
│  ├── /joint_trajectory                      ├── /compute_fk                            │
│  ├── /bottle_pose                           ├── /plan_kinematic_path                   │
│  ├── /gripper_state                         ├── /execute_trajectory                    │
│  ├── /camera/color/image_raw                └── /gripper_command                       │
│  ├── /camera/depth/image_raw                                                           │
│  ├── /tf                                    ACTIONS:                                   │
│  └── /visualization_marker                  ├── /follow_joint_trajectory               │
│                                             └── /move_group                            │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

##  TRAINING LOOP PSEUDOCODE

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           REINFORCEMENT LEARNING LOOP                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

ALGORITHM: PPO (Proximal Policy Optimization) for Bottle Flip

╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                         ║
║  INITIALIZE:                                                                            ║
║      policy_network = PolicyNet(state_dim=18, action_dim=6)                            ║
║      value_network = ValueNet(state_dim=18)                                            ║
║      pybullet_env = BottleFlipEnv()                                                    ║
║      replay_buffer = RolloutBuffer()                                                    ║
║                                                                                         ║
║  FOR episode = 1 TO max_episodes:                                                       ║
║      │                                                                                  ║
║      │  # Reset environment                                                             ║
║      │  state = pybullet_env.reset()                                                   ║
║      │  episode_reward = 0                                                              ║
║      │                                                                                  ║
║      │  FOR step = 1 TO max_steps:                                                     ║
║      │      │                                                                           ║
║      │      │  # Get action from policy                                                ║
║      │      │  action, log_prob = policy_network.sample_action(state)                  ║
║      │      │                                                                           ║
║      │      │  # Execute action in simulation                                          ║
║      │      │  next_state, reward, done = pybullet_env.step(action)                   ║
║      │      │                                                                           ║
║      │      │  # Store transition                                                       ║
║      │      │  replay_buffer.store(state, action, reward, next_state, log_prob)       ║
║      │      │                                                                           ║
║      │      │  # Update state                                                          ║
║      │      │  state = next_state                                                      ║
║      │      │  episode_reward += reward                                                ║
║      │      │                                                                           ║
║      │      │  IF done:                                                                ║
║      │      │      BREAK                                                               ║
║      │      │                                                                           ║
║      │  END FOR                                                                         ║
║      │                                                                                  ║
║      │  # Update policy every N episodes                                               ║
║      │  IF episode % update_frequency == 0:                                            ║
║      │      │                                                                           ║
║      │      │  # Compute advantages                                                    ║
║      │      │  advantages = compute_gae(replay_buffer, value_network)                  ║
║      │      │                                                                           ║
║      │      │  # PPO update                                                            ║
║      │      │  FOR _ IN range(ppo_epochs):                                             ║
║      │      │      policy_loss = ppo_policy_loss(advantages, clip_ratio)              ║
║      │      │      value_loss = mse_loss(value_network, returns)                      ║
║      │      │      optimizer.step(policy_loss + value_loss)                           ║
║      │      │                                                                           ║
║      │      │  replay_buffer.clear()                                                   ║
║      │      │                                                                           ║
║      │  END IF                                                                          ║
║      │                                                                                  ║
║      │  # Log progress                                                                  ║
║      │  log(episode, episode_reward, success_rate)                                     ║
║      │                                                                                  ║
║  END FOR                                                                                ║
║                                                                                         ║
║  SAVE policy_network.state_dict() → "trained_policy.pt"                                ║
║                                                                                         ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝


STATE SPACE (18 dimensions):
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  [0:6]   Robot joint positions (6 DOF)                                                 │
│  [6:12]  Robot joint velocities (6 DOF)                                                │
│  [12:15] Bottle position (x, y, z)                                                     │
│  [15:18] Bottle orientation (roll, pitch, yaw)                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

ACTION SPACE (6 dimensions):
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  [0:6]   Target joint velocities or position deltas for each joint                     │
│          Normalized to [-1, 1] and scaled to joint limits                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

REWARD FUNCTION:
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  reward = 0                                                                            │
│                                                                                         │
│  # Flip success (bottle lands upright)                                                 │
│  IF bottle_upright AND bottle_stationary:                                              │
│      reward += 100.0                                                                   │
│                                                                                         │
│  # Partial credit for rotation                                                         │
│  reward += 10.0 * (rotation_achieved / target_rotation)                                │
│                                                                                         │
│  # Height bonus                                                                         │
│  reward += 5.0 * max_height_reached                                                    │
│                                                                                         │
│  # Penalties                                                                            │
│  reward -= 0.1 * joint_effort  # Energy efficiency                                     │
│  reward -= 10.0 * collision_detected                                                   │
│  reward -= 5.0 * out_of_bounds                                                         │
│                                                                                         │
│  RETURN reward                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

##  DEPLOYMENT SEQUENCE

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT TO REAL ROBOT                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘

SIMULATION ───────────────────────────────────────────────────────────► REAL ROBOT

Step 1: VERIFY IN PYBULLET
┌─────────────────────────────────────────┐
│  • Load trained policy                  │
│  • Run 100 simulation episodes          │
│  • Verify >80% success rate             │
│  • Check joint limits respected         │
│  • Verify no self-collisions            │
└─────────────────────────────────────────┘
                    │
                    ▼
Step 2: VALIDATE IN RVIZ
┌─────────────────────────────────────────┐
│  • Play trajectory visualization        │
│  • Verify smooth motion                 │
│  • Check workspace boundaries           │
│  • Review velocity profiles             │
│  • Get team approval                    │
└─────────────────────────────────────────┘
                    │
                    ▼
Step 3: CONNECT TO ROBOT (TEACH MODE)
┌─────────────────────────────────────────┐
│  • Enable teach mode on UR10e           │
│  • Manually guide through keypoints     │
│  • Verify no obstacles in path          │
│  • Record safety boundaries             │
└─────────────────────────────────────────┘
                    │
                    ▼
Step 4: TEST AT REDUCED SPEED (10%)
┌─────────────────────────────────────────┐
│  • Set velocity_scaling = 0.1           │
│  • Execute single trajectory            │
│  • Emergency stop ready                 │
│  • Observe and verify motion            │
└─────────────────────────────────────────┘
                    │
                    ▼
Step 5: GRADUAL SPEED INCREASE
┌─────────────────────────────────────────┐
│  • 10% → 25% → 50% → 75% → 100%        │
│  • Multiple tests at each level         │
│  • Monitor joint torques                │
│  • Verify consistent behavior           │
└─────────────────────────────────────────┘
                    │
                    ▼
Step 6: FULL DEPLOYMENT
┌─────────────────────────────────────────┐
│  • Full speed execution                 │
│  • Real bottle flipping!                │
│  • Record success/failure data          │
│  • Fine-tune parameters                 │
└─────────────────────────────────────────┘
```
