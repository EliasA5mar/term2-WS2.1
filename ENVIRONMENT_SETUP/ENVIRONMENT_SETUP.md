# Complete Environment Setup Guide
## WSL2 → Docker → PyBullet/PyTorch → RVIZ → UR10e

---

##  SETUP OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           SETUP SEQUENCE                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘

   STEP 1          STEP 2          STEP 3          STEP 4          STEP 5
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  WSL2    │───►│  Docker  │───►│  NVIDIA  │───►│  ROS2    │───►│  UR10e   │
│  Ubuntu  │    │  Desktop │    │  Toolkit │    │  Stack   │    │  Connect │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
   ~10min         ~10min          ~10min          ~30min          ~10min
```

---

##  STEP 1: WSL2 Setup (Windows)

### 1.1 Enable WSL2

Open **PowerShell as Administrator**:

```powershell
# Enable WSL
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart your computer
Restart-Computer
```

### 1.2 Install WSL2 and Ubuntu

After restart, in PowerShell (Admin):

```powershell
# Set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Or list available distros
wsl --list --online
```

### 1.3 Configure WSL2 Resources

Create/edit `C:\Users\<YourUsername>\.wslconfig`:

```ini
[wsl2]
memory=16GB
processors=8
swap=4GB
localhostForwarding=true

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
```

### 1.4 Launch Ubuntu and Initial Setup

```bash
# Launch Ubuntu from Start Menu or:
wsl -d Ubuntu-22.04

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    net-tools \
    x11-apps
```

### 1.5 Configure X11 Display (for GUI apps)

```bash
# Add to ~/.bashrc
echo 'export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '\''{print $2}'\''):0' >> ~/.bashrc
echo 'export LIBGL_ALWAYS_INDIRECT=0' >> ~/.bashrc
source ~/.bashrc
```

On Windows, install **VcXsrv** or **X410** for X11 server.

---

##  STEP 2: Docker Desktop Setup

### 2.1 Install Docker Desktop on Windows

1. Download from: https://www.docker.com/products/docker-desktop
2. Run installer
3. Enable **WSL2 backend** during installation
4. Restart computer if prompted

### 2.2 Configure Docker for WSL2

In Docker Desktop:
1. Go to **Settings** → **General**
2. Check **Use the WSL 2 based engine**
3. Go to **Settings** → **Resources** → **WSL Integration**
4. Enable integration with **Ubuntu-22.04**

### 2.3 Verify Docker in WSL2

```bash
# In WSL2 Ubuntu terminal
docker --version
docker run hello-world
```

---

##  STEP 3: NVIDIA Container Toolkit (GPU Support)

### 3.1 Install NVIDIA Drivers on Windows

1. Download latest drivers from: https://www.nvidia.com/drivers
2. Install the **Game Ready** or **Studio** driver
3. Restart computer

### 3.2 Install NVIDIA Container Toolkit in WSL2

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3.3 Verify GPU Access

```bash
# Test NVIDIA GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

---

##  STEP 4: Build ROS2 + Simulation Docker Image

### 4.1 Create Dockerfile

Create `~/bottle_flip_docker/Dockerfile`:

```dockerfile
# Dockerfile for UR10e Bottle Flip Project
# Base: NVIDIA CUDA + Ubuntu 22.04

FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Set locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    vim \
    gnupg2 \
    lsb-release \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    x11-apps \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ========== Install ROS2 Humble ==========
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y \
    ros-humble-desktop-full \
    ros-humble-ros-base \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# ========== Install UR Robot Packages ==========
RUN apt-get update && apt-get install -y \
    ros-humble-ur \
    ros-humble-ur-robot-driver \
    ros-humble-ur-description \
    ros-humble-ur-moveit-config \
    ros-humble-ur-controllers \
    && rm -rf /var/lib/apt/lists/*

# ========== Install MoveIt2 ==========
RUN apt-get update && apt-get install -y \
    ros-humble-moveit \
    ros-humble-moveit-resources \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-servo \
    && rm -rf /var/lib/apt/lists/*

# ========== Install ROS2 Control ==========
RUN apt-get update && apt-get install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*

# ========== Install Python ML/Simulation Stack ==========
RUN pip3 install --upgrade pip && pip3 install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install \
    pybullet \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    opencv-contrib-python \
    gymnasium \
    stable-baselines3 \
    tensorboard \
    pyyaml \
    tqdm

# ========== Install Additional Tools ==========
RUN apt-get update && apt-get install -y \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rviz2 \
    ros-humble-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true && rosdep update

# Create workspace
RUN mkdir -p /home/user/ros2_ws/src
WORKDIR /home/user/ros2_ws

# Setup entrypoint
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /home/user/ros2_ws/install/setup.bash 2>/dev/null || true" >> /root/.bashrc

# Set display for GUI
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

CMD ["/bin/bash"]
```

### 4.2 Create Docker Compose File

Create `~/bottle_flip_docker/docker-compose.yml`:

```yaml
version: '3.8'

services:
  bottle-flip-ros2:
    build:
      context: .
      dockerfile: Dockerfile
    image: bottle-flip-ros2:latest
    container_name: bottle_flip_container

    # Enable GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    # Environment
    environment:
      - DISPLAY=${DISPLAY}
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - QT_X11_NO_MITSHM=1
      - ROS_DOMAIN_ID=0

    # Volumes
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./ros2_ws:/home/user/ros2_ws:rw
      - ./trained_models:/home/user/trained_models:rw
      - /dev:/dev:rw

    # Network
    network_mode: host
    privileged: true

    # Keep container running
    stdin_open: true
    tty: true

    # Working directory
    working_dir: /home/user/ros2_ws

  # Optional: Separate container for simulation only
  pybullet-sim:
    build:
      context: .
      dockerfile: Dockerfile
    image: bottle-flip-ros2:latest
    container_name: pybullet_sim_container

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    environment:
      - DISPLAY=${DISPLAY}
      - NVIDIA_VISIBLE_DEVICES=all

    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./simulation:/home/user/simulation:rw
      - ./trained_models:/home/user/trained_models:rw

    network_mode: host
    stdin_open: true
    tty: true

    working_dir: /home/user/simulation
    command: python3 pybullet_training.py
```

### 4.3 Build and Run Docker

```bash
# Navigate to docker directory
cd ~/bottle_flip_docker

# Build the image (takes ~20-30 minutes first time)
docker-compose build

# Allow X11 connections (run on host WSL2)
xhost +local:docker

# Start the container
docker-compose up -d bottle-flip-ros2

# Enter the container
docker exec -it bottle_flip_container bash
```

---

##  STEP 5: PyBullet + PyTorch Simulation Setup

### 5.1 PyBullet Simulation Script

Create `~/bottle_flip_docker/simulation/pybullet_bottle_flip.py`:

```python
#!/usr/bin/env python3
"""
PyBullet Simulation Environment for UR10e Bottle Flip
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
from math import pi


class BottleFlipSimEnv:
    """PyBullet simulation environment for bottle flipping."""

    def __init__(self, gui=True):
        # Connect to physics server
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Set simulation parameters
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)

        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")

        # Load UR10e robot
        self.robot_id = self._load_ur10e()

        # Load bottle
        self.bottle_id = self._create_bottle()

        # Joint info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(6))  # UR10e has 6 joints

        # Joint limits
        self.joint_lower = [-2*pi, -2*pi, -pi, -2*pi, -2*pi, -2*pi]
        self.joint_upper = [2*pi, 2*pi, pi, 2*pi, 2*pi, 2*pi]

        # State
        self.step_count = 0
        self.max_steps = 500

    def _load_ur10e(self):
        """Load UR10e robot URDF."""
        # You can download UR10e URDF from Universal Robots GitHub
        # For now, use a placeholder
        robot_start_pos = [0, 0, 0]
        robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])

        # Try to load UR10e URDF
        try:
            robot_id = p.loadURDF(
                "ur_e_description/urdf/ur10e.urdf",
                robot_start_pos,
                robot_start_orn,
                useFixedBase=True
            )
        except:
            # Fallback: use Kuka as placeholder
            robot_id = p.loadURDF(
                "kuka_iiwa/model.urdf",
                robot_start_pos,
                robot_start_orn,
                useFixedBase=True
            )
            print("Warning: Using Kuka as placeholder. Install UR10e URDF for accurate simulation.")

        return robot_id

    def _create_bottle(self):
        """Create a simple bottle using collision shapes."""
        # Bottle dimensions (approximate water bottle)
        bottle_height = 0.25  # 25cm
        bottle_radius = 0.035  # 3.5cm radius

        # Create collision shape (cylinder)
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=bottle_radius,
            height=bottle_height
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=bottle_radius,
            length=bottle_height,
            rgbaColor=[0.2, 0.5, 0.8, 1.0]  # Blue color
        )

        # Create bottle body
        bottle_mass = 0.5  # 500g (half-filled)
        bottle_start_pos = [0.5, 0, bottle_height/2 + 0.01]
        bottle_start_orn = p.getQuaternionFromEuler([0, 0, 0])

        bottle_id = p.createMultiBody(
            baseMass=bottle_mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=bottle_start_pos,
            baseOrientation=bottle_start_orn
        )

        # Set friction and restitution
        p.changeDynamics(
            bottle_id, -1,
            lateralFriction=0.8,
            restitution=0.3,
            linearDamping=0.1,
            angularDamping=0.1
        )

        return bottle_id

    def reset(self):
        """Reset the environment."""
        self.step_count = 0

        # Reset robot to home position
        home_position = [0, -pi/2, 0, -pi/2, 0, 0]
        for i, pos in enumerate(home_position):
            p.resetJointState(self.robot_id, i, pos)

        # Reset bottle position
        bottle_pos = [0.5, 0, 0.125 + 0.01]
        bottle_orn = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.bottle_id, bottle_pos, bottle_orn)
        p.resetBaseVelocity(self.bottle_id, [0, 0, 0], [0, 0, 0])

        return self._get_observation()

    def step(self, action):
        """Execute action and return new state."""
        self.step_count += 1

        # Apply action (joint velocities or position targets)
        action = np.clip(action, -1, 1)  # Normalize

        # Scale to joint velocity limits
        max_vel = 2.0  # rad/s
        target_velocities = action * max_vel

        # Apply velocity control
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.VELOCITY_CONTROL,
            targetVelocities=target_velocities,
            forces=[150] * 6
        )

        # Step simulation
        for _ in range(4):  # Sub-stepping for stability
            p.stepSimulation()

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if done
        done = self._is_done()

        return observation, reward, done, {}

    def _get_observation(self):
        """Get current observation (state)."""
        # Robot joint states
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        # Bottle state
        bottle_pos, bottle_orn = p.getBasePositionAndOrientation(self.bottle_id)
        bottle_vel, bottle_ang_vel = p.getBaseVelocity(self.bottle_id)
        bottle_euler = p.getEulerFromQuaternion(bottle_orn)

        # Combine into observation vector
        observation = np.array(
            joint_positions + joint_velocities +
            list(bottle_pos) + list(bottle_euler)
        )

        return observation

    def _calculate_reward(self):
        """Calculate reward for current state."""
        reward = 0.0

        # Get bottle state
        bottle_pos, bottle_orn = p.getBasePositionAndOrientation(self.bottle_id)
        bottle_euler = p.getEulerFromQuaternion(bottle_orn)
        bottle_vel, bottle_ang_vel = p.getBaseVelocity(self.bottle_id)

        # Check if bottle is upright (pitch and roll near 0)
        is_upright = abs(bottle_euler[0]) < 0.2 and abs(bottle_euler[1]) < 0.2

        # Check if bottle has landed (low velocity and on ground)
        is_stationary = np.linalg.norm(bottle_vel) < 0.1 and np.linalg.norm(bottle_ang_vel) < 0.1
        is_on_ground = bottle_pos[2] < 0.2

        # Successful flip!
        if is_upright and is_stationary and is_on_ground and self.step_count > 50:
            reward += 100.0

        # Reward for height achieved
        reward += bottle_pos[2] * 5.0

        # Reward for rotation (flip)
        total_rotation = abs(bottle_euler[0]) + abs(bottle_euler[1])
        reward += total_rotation * 2.0

        # Penalty for bottle falling off table (out of bounds)
        if bottle_pos[2] < 0:
            reward -= 20.0

        return reward

    def _is_done(self):
        """Check if episode is done."""
        # Max steps reached
        if self.step_count >= self.max_steps:
            return True

        # Bottle fell off
        bottle_pos, _ = p.getBasePositionAndOrientation(self.bottle_id)
        if bottle_pos[2] < -0.1:
            return True

        return False

    def render(self):
        """Render is automatic in GUI mode."""
        pass

    def close(self):
        """Disconnect from physics server."""
        p.disconnect()


def test_simulation():
    """Test the simulation environment."""
    env = BottleFlipSimEnv(gui=True)

    print("Testing BottleFlipSimEnv...")
    print("Press Ctrl+C to exit")

    obs = env.reset()
    print(f"Observation shape: {obs.shape}")

    try:
        for episode in range(3):
            obs = env.reset()
            total_reward = 0

            for step in range(env.max_steps):
                # Random action for testing
                action = np.random.uniform(-1, 1, 6)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

                time.sleep(1/60)  # Slow down for visualization

                if done:
                    break

            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\nExiting...")

    env.close()


if __name__ == "__main__":
    test_simulation()
```

### 5.2 PyTorch Training Script

Create `~/bottle_flip_docker/simulation/train_policy.py`:

```python
#!/usr/bin/env python3
"""
PyTorch PPO Training for Bottle Flip Policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from pybullet_bottle_flip import BottleFlipSimEnv


class PolicyNetwork(nn.Module):
    """Actor network for PPO."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        features = self.network(state)
        mean = torch.tanh(self.mean(features))
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob


class ValueNetwork(nn.Module):
    """Critic network for PPO."""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state)


class PPOTrainer:
    """PPO trainer for bottle flip policy."""

    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device

        # Dimensions
        self.state_dim = 18  # 6 joint pos + 6 joint vel + 3 bottle pos + 3 bottle orient
        self.action_dim = 6

        # Networks
        self.policy = PolicyNetwork(self.state_dim, self.action_dim).to(device)
        self.value = ValueNetwork(self.state_dim).to(device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-3)

        # Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.epochs = 10
        self.batch_size = 64

        # Logging
        self.episode_rewards = []
        self.success_count = 0

    def collect_rollout(self, num_steps=2048):
        """Collect experience from environment."""
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        state = self.env.reset()

        for _ in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob = self.policy.sample_action(state_tensor)
                value = self.value(state_tensor)

            action_np = action.cpu().numpy().flatten()
            next_state, reward, done, _ = self.env.step(action_np)

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state

            if done:
                state = self.env.reset()

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'log_probs': np.array(log_probs),
            'values': np.array(values)
        }

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """Update policy and value networks."""
        states = torch.FloatTensor(rollout['states']).to(self.device)
        actions = torch.FloatTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)

        advantages, returns = self.compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones']
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update for multiple epochs
        for _ in range(self.epochs):
            # Shuffle indices
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Policy loss
                mean, std = self.policy(batch_states)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()

                # Value loss
                values = self.value(batch_states).squeeze()
                value_loss = nn.MSELoss()(values, batch_returns)

                # Update policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Update value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

    def train(self, num_iterations=1000, rollout_steps=2048):
        """Main training loop."""
        print("Starting training...")

        for iteration in range(num_iterations):
            # Collect rollout
            rollout = self.collect_rollout(rollout_steps)

            # Update networks
            self.update(rollout)

            # Calculate episode rewards
            episode_reward = np.sum(rollout['rewards'])
            self.episode_rewards.append(episode_reward)

            # Log progress
            if iteration % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Iteration {iteration}: Avg Reward = {avg_reward:.2f}")

            # Save checkpoint
            if iteration % 100 == 0:
                self.save_model(f'checkpoint_{iteration}.pt')

        self.save_model('trained_policy.pt')
        self.plot_training()

    def save_model(self, path):
        """Save trained model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load trained model."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        print(f"Model loaded from {path}")

    def plot_training(self):
        """Plot training curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.xlabel('Iteration')
        plt.ylabel('Episode Reward')
        plt.title('Training Progress')
        plt.savefig('training_curves.png')
        plt.close()
        print("Training curves saved to training_curves.png")


def main():
    # Create environment (headless for training)
    env = BottleFlipSimEnv(gui=False)

    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    trainer = PPOTrainer(env, device)

    # Train
    trainer.train(num_iterations=500)

    # Clean up
    env.close()


if __name__ == "__main__":
    main()
```

---

##  STEP 6: RVIZ2 Visualization Pipeline

### 6.1 Connect PyBullet to RVIZ via ROS2

Create `~/bottle_flip_docker/ros2_ws/src/bottle_flip/pybullet_ros2_bridge.py`:

```python
#!/usr/bin/env python3
"""
Bridge between PyBullet simulation and ROS2/RVIZ
Publishes joint states from PyBullet to ROS2 for visualization in RVIZ
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import pybullet as p
import pybullet_data
import numpy as np
from math import pi
import time


class PyBulletROS2Bridge(Node):
    """Bridges PyBullet simulation to ROS2."""

    def __init__(self):
        super().__init__('pybullet_ros2_bridge')

        self.get_logger().info('Starting PyBullet-ROS2 Bridge...')

        # Joint names for UR10e
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Publisher for joint states
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load robot
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",  # Replace with UR10e URDF
            [0, 0, 0],
            useFixedBase=True
        )

        # Timer for publishing
        self.timer = self.create_timer(0.02, self.publish_joint_states)  # 50Hz

        self.get_logger().info('Bridge initialized!')

    def publish_joint_states(self):
        """Publish current joint states to ROS2."""
        # Get joint states from PyBullet
        joint_states = []
        for i in range(6):
            state = p.getJointState(self.robot_id, i)
            joint_states.append(state)

        # Create JointState message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [state[0] for state in joint_states]
        msg.velocity = [state[1] for state in joint_states]
        msg.effort = [state[3] for state in joint_states]

        # Publish
        self.joint_state_pub.publish(msg)

        # Step simulation
        p.stepSimulation()


def main(args=None):
    rclpy.init(args=args)
    bridge = PyBulletROS2Bridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## 🔌 STEP 7: Real Robot Connection

### 7.1 Network Configuration

```bash
# On your Ubuntu/WSL2 machine
# Set static IP (adjust for your network)
sudo ip addr add 192.168.1.100/24 dev eth0

# Test connection to robot
ping 192.168.1.101  # UR10e default IP

# Check RTDE port
nc -zv 192.168.1.101 50002
```

### 7.2 Launch Real Robot Driver

```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Launch UR driver
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur10e \
    robot_ip:=192.168.1.101 \
    launch_rviz:=true
```

---

##  COMPLETE COMMANDS CHEAT SHEET

```bash
# ============= SETUP =============
# Build Docker image
cd ~/bottle_flip_docker
docker-compose build

# Start container
docker-compose up -d bottle-flip-ros2
docker exec -it bottle_flip_container bash

# ============= SIMULATION =============
# Run PyBullet simulation (standalone)
python3 pybullet_bottle_flip.py

# Train policy
python3 train_policy.py

# ============= RVIZ =============
# Launch RVIZ with UR10e (inside container)
ros2 launch ur_description view_ur.launch.py ur_type:=ur10e

# Run bottle flip controller
ros2 run bottle_flip bottle_flip_simple

# ============= REAL ROBOT =============
# Launch real robot driver
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur10e \
    robot_ip:=192.168.1.101

# Run trained policy on real robot
ros2 run bottle_flip bottle_flip_ur10e --ros-args -p use_sim:=false
```

---

##  USEFUL LINKS

- WSL2 Documentation: https://docs.microsoft.com/en-us/windows/wsl/
- Docker Desktop: https://www.docker.com/products/docker-desktop
- ROS2 Humble: https://docs.ros.org/en/humble/
- Universal Robots ROS2: https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver
- PyBullet: https://pybullet.org/
- PyTorch: https://pytorch.org/
- MoveIt2: https://moveit.picknik.ai/

---

This guide covers the complete pipeline from Windows → WSL2 → Docker → Simulation → Real Robot!
