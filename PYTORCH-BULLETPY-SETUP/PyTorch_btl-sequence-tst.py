"""
================================================================================
UR10e BOTTLE FLIP - Machine Learning Pipeline
================================================================================
Single gripper, single bottle shape, variable water weight (50-500g)

Project Structure:
├── phase1_environment_setup.py
├── phase2_simulation.py
├── phase3_training.py
├── phase4_sim_to_real.py
├── phase5_real_robot_finetuning.py
├── phase6_deployment.py
└── utils/
    ├── robot_controller.py
    ├── sensors.py
    └── visualization.py

Author: For UR10e Bottle Flip Demo
Date: 2026
================================================================================
"""

# ============================================================================
# PHASE 1: ENVIRONMENT SETUP & PROBLEM DEFINITION
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from collections import deque

@dataclass
class BottleConfig:
    """Configuration for bottle parameters"""
    height: float = 0.20  # meters (20cm bottle)
    diameter: float = 0.06  # meters (6cm diameter)
    empty_weight: float = 0.030  # kg (30g empty bottle)
    water_weight_min: float = 0.050  # kg (50g min water)
    water_weight_max: float = 0.500  # kg (500g max water)
    
    @property
    def total_weight_range(self):
        return (
            self.empty_weight + self.water_weight_min,
            self.empty_weight + self.water_weight_max
        )

@dataclass
class RobotConfig:
    """UR10e Robot Configuration"""
    num_joints: int = 6
    max_joint_velocity: float = 3.14  # rad/s
    max_joint_acceleration: float = 10.0  # rad/s²
    workspace_limits: Tuple = (
        [-0.85, 0.85],  # X range (meters)
        [-0.85, 0.85],  # Y range
        [0.0, 1.30]     # Z range
    )
    gripper_force_range: Tuple[float, float] = (20.0, 100.0)  # Newtons

class StateSpace:
    """
    Defines the observation space for the RL agent
    
    State Vector Composition (Total: 21 dimensions):
    ┌─────────────────────────────────────────┐
    │ [0-5]   : Joint positions (6D)          │
    │ [6-11]  : Joint velocities (6D)         │
    │ [12]    : Bottle total weight (1D)      │
    │ [13]    : Gripper force (1D)            │
    │ [14-16] : Bottle position XYZ (3D)      │
    │ [17-20] : Bottle orientation quat (4D)  │
    └─────────────────────────────────────────┘
    """
    def __init__(self, robot_config: RobotConfig, bottle_config: BottleConfig):
        self.robot_config = robot_config
        self.bottle_config = bottle_config
        self.dimension = 21
        
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1] range for better neural network training"""
        normalized = state.copy()
        
        # Normalize joint positions to [-1, 1] (assuming -π to π range)
        normalized[0:6] = state[0:6] / np.pi
        
        # Normalize joint velocities
        normalized[6:12] = state[6:12] / self.robot_config.max_joint_velocity
        
        # Normalize bottle weight
        min_w, max_w = self.bottle_config.total_weight_range
        normalized[12] = (state[12] - min_w) / (max_w - min_w) * 2 - 1
        
        # Normalize gripper force
        min_f, max_f = self.robot_config.gripper_force_range
        normalized[13] = (state[13] - min_f) / (max_f - min_f) * 2 - 1
        
        # Normalize position (workspace limits)
        for i, (min_val, max_val) in enumerate(self.robot_config.workspace_limits):
            normalized[14+i] = (state[14+i] - min_val) / (max_val - min_val) * 2 - 1
        
        # Quaternion is already normalized
        normalized[17:21] = state[17:21]
        
        return normalized

class ActionSpace:
    """
    Defines the action space for the RL agent
    
    Action Vector Composition (Total: 10 dimensions):
    ┌──────────────────────────────────────────────────┐
    │ [0-5]  : Target joint velocities (6D)            │
    │ [6]    : Flip trajectory height (1D)             │
    │ [7]    : Flip trajectory duration (1D)           │
    │ [8]    : Release timing offset (1D)              │
    │ [9]    : Initial gripper force multiplier (1D)   │
    └──────────────────────────────────────────────────┘
    """
    def __init__(self, robot_config: RobotConfig):
        self.robot_config = robot_config
        self.dimension = 10
        
        # Action bounds
        self.bounds = {
            'joint_velocities': (-robot_config.max_joint_velocity, 
                                robot_config.max_joint_velocity),
            'trajectory_height': (0.3, 0.8),  # meters
            'trajectory_duration': (0.2, 0.6),  # seconds
            'release_timing': (-0.1, 0.1),  # seconds offset
            'force_multiplier': (0.5, 1.5)  # grip force multiplier
        }
    
    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip actions to valid ranges"""
        clipped = action.copy()
        
        # Clip joint velocities
        clipped[0:6] = np.clip(
            action[0:6],
            self.bounds['joint_velocities'][0],
            self.bounds['joint_velocities'][1]
        )
        
        # Clip trajectory parameters
        clipped[6] = np.clip(action[6], *self.bounds['trajectory_height'])
        clipped[7] = np.clip(action[7], *self.bounds['trajectory_duration'])
        clipped[8] = np.clip(action[8], *self.bounds['release_timing'])
        clipped[9] = np.clip(action[9], *self.bounds['force_multiplier'])
        
        return clipped

class BottleFlipEnvironment:
    """
    Main environment class for bottle flipping task
    
    Workflow per episode:
    1. Reset: Place bottle with random water weight
    2. Agent observes state
    3. Agent selects action (trajectory parameters)
    4. Execute flip motion
    5. Evaluate landing
    6. Return reward
    """
    def __init__(self, mode='simulation'):
        self.mode = mode
        self.robot_config = RobotConfig()
        self.bottle_config = BottleConfig()
        self.state_space = StateSpace(self.robot_config, self.bottle_config)
        self.action_space = ActionSpace(self.robot_config)
        
        # Episode tracking
        self.current_step = 0
        self.max_steps_per_episode = 1
        self.episode_count = 0
        
        # Success tracking
        self.success_history = deque(maxlen=100)
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║          UR10e BOTTLE FLIP ENVIRONMENT INITIALIZED           ║
╠══════════════════════════════════════════════════════════════╣
║ Mode: {mode:50s} ║
║ State Dimension: {self.state_space.dimension:43d} ║
║ Action Dimension: {self.action_space.dimension:42d} ║
║ Bottle Weight Range: {self.bottle_config.total_weight_range[0]:.3f} - {self.bottle_config.total_weight_range[1]:.3f} kg      ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def reset(self, bottle_weight: Optional[float] = None) -> np.ndarray:
        """
        Reset environment for new episode
        
        Returns:
            Initial state observation
        """
        self.current_step = 0
        self.episode_count += 1
        
        # Random water weight if not specified
        if bottle_weight is None:
            water_weight = np.random.uniform(
                self.bottle_config.water_weight_min,
                self.bottle_config.water_weight_max
            )
            bottle_weight = self.bottle_config.empty_weight + water_weight
        
        self.current_bottle_weight = bottle_weight
        
        # Initial state: robot at home position, bottle in gripper
        state = np.zeros(self.state_space.dimension)
        state[0:6] = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])  # Home pose
        state[12] = bottle_weight  # Bottle weight
        state[13] = 50.0  # Initial gripper force
        state[14:17] = np.array([0.5, 0.0, 0.3])  # Bottle position
        state[17:21] = np.array([0, 0, 0, 1])  # Upright orientation (quaternion)
        
        print(f"\n[Episode {self.episode_count}] Reset: Bottle weight = {bottle_weight*1000:.1f}g")
        
        return self.state_space.normalize_state(state)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action vector from agent
            
        Returns:
            next_state: Next observation
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information
        """
        self.current_step += 1
        
        # Clip action to valid range
        action = self.action_space.clip_action(action)
        
        # Execute flip motion (in simulation or real robot)
        flip_result = self._execute_flip(action)
        
        # Calculate reward
        reward = self._calculate_reward(flip_result)
        
        # Episode ends after one flip attempt
        done = True
        
        # Track success
        success = flip_result['landed_upright']
        self.success_history.append(1.0 if success else 0.0)
        
        info = {
            'bottle_weight': self.current_bottle_weight,
            'success': success,
            'flip_height': flip_result['max_height'],
            'landing_angle': flip_result['landing_angle'],
            'success_rate': np.mean(self.success_history) if self.success_history else 0.0
        }
        
        # Next state is final bottle state
        next_state = flip_result['final_state']
        
        return self.state_space.normalize_state(next_state), reward, done, info
    
    def _execute_flip(self, action: np.ndarray) -> Dict:
        """
        Execute the flip motion (to be implemented in simulation/real robot)
        
        This is a placeholder - will be replaced by actual physics simulation
        or real robot control
        """
        # Extract action components
        joint_velocities = action[0:6]
        traj_height = action[6]
        traj_duration = action[7]
        release_timing = action[8]
        force_mult = action[9]
        
        # Placeholder: simulate flip success based on action quality
        # Real implementation will use physics engine or robot
        
        # Simple heuristic for demo: success depends on trajectory params
        optimal_height = 0.5
        optimal_duration = 0.4
        
        height_error = abs(traj_height - optimal_height)
        duration_error = abs(traj_duration - optimal_duration)
        timing_error = abs(release_timing)
        
        # Weight-dependent success probability
        weight_factor = 1.0 - abs(self.current_bottle_weight - 0.3) / 0.5
        
        success_prob = max(0, 1.0 - height_error - duration_error - timing_error)
        success_prob *= weight_factor
        
        landed_upright = np.random.random() < success_prob
        
        # Generate flip results
        max_height = traj_height + np.random.normal(0, 0.05)
        landing_angle = 0 if landed_upright else np.random.uniform(30, 180)
        
        # Final state
        final_state = np.zeros(self.state_space.dimension)
        final_state[12] = self.current_bottle_weight
        final_state[14:17] = np.array([0.5, 0.0, 0.1])  # Landed position
        
        if landed_upright:
            final_state[17:21] = np.array([0, 0, 0, 1])  # Upright
        else:
            # Random fallen orientation
            angle = np.random.uniform(0, 2*np.pi)
            final_state[17:21] = np.array([
                np.sin(angle/2), 0, 0, np.cos(angle/2)
            ])
        
        return {
            'landed_upright': landed_upright,
            'max_height': max_height,
            'landing_angle': landing_angle,
            'final_state': final_state,
            'trajectory_duration': traj_duration
        }
    
    def _calculate_reward(self, flip_result: Dict) -> float:
        """
        Calculate reward based on flip outcome
        
        Reward Structure:
        ┌────────────────────────────────────────┐
        │ Landed upright:        +100.0          │
        │ Partial flip:          +50.0 - angle   │
        │ No flip:               -10.0           │
        │ Height bonus:          +0 to +10       │
        │ Smoothness penalty:    -5 to 0         │
        └────────────────────────────────────────┘
        """
        reward = 0.0
        
        if flip_result['landed_upright']:
            # Success!
            reward = 100.0
            
            # Bonus for optimal height
            optimal_height = 0.5
            height_bonus = 10.0 * (1.0 - abs(flip_result['max_height'] - optimal_height))
            reward += height_bonus
            
        else:
            # Partial credit for attempting flip
            landing_angle = flip_result['landing_angle']
            
            if landing_angle < 90:
                # Bottle flipped but fell over
                reward = 50.0 - (landing_angle / 90.0) * 40.0
            else:
                # Didn't flip much
                reward = -10.0
        
        # Small penalty for extreme trajectory duration
        duration = flip_result['trajectory_duration']
        if duration < 0.25 or duration > 0.55:
            reward -= 5.0
        
        return reward
    
    def render(self, mode='human'):
        """Visualize current state (for debugging)"""
        if mode == 'human':
            success_rate = np.mean(self.success_history) if self.success_history else 0.0
            print(f"Episode: {self.episode_count} | Success Rate: {success_rate:.2%}")


# ============================================================================
# PHASE 2: SIMULATION SETUP (PyBullet)
# ============================================================================

class BottleFlipSimulation:
    """
    Physics-based simulation using PyBullet
    
    Components:
    - UR10e robot model
    - Bottle with variable mass
    - Ground plane
    - Physics engine
    
    Note: This requires PyBullet installation
          pip install pybullet
    """
    def __init__(self, gui=True):
        try:
            import pybullet as p
            import pybullet_data
            self.p = p
        except ImportError:
            print("⚠️  PyBullet not installed. Using placeholder simulation.")
            self.p = None
            return
        
        self.gui = gui
        
        # Connect to physics server
        if gui:
            self.physics_client = self.p.connect(self.p.GUI)
        else:
            self.physics_client = self.p.connect(self.p.DIRECT)
        
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)
        
        # Load models
        self.plane_id = self.p.loadURDF("plane.urdf")
        
        print("""
╔══════════════════════════════════════════════════════════════╗
║              PYBULLET SIMULATION INITIALIZED                 ║
╠══════════════════════════════════════════════════════════════╣
║ Physics Engine: PyBullet                                     ║
║ GUI Mode: {:49s} ║
║ Gravity: -9.81 m/s²                                          ║
╚══════════════════════════════════════════════════════════════╝
        """.format(str(gui)))
        
        # Robot and bottle will be loaded in reset()
        self.robot_id = None
        self.bottle_id = None
        
    def load_robot(self):
        """Load UR10e robot model"""
        if self.p is None:
            return None
            
        # In real implementation, load actual UR10e URDF
        # For demo, we'll create a simple placeholder
        print("📦 Loading UR10e robot model...")
        
        # Placeholder: In production, use actual URDF
        # self.robot_id = self.p.loadURDF("ur10e.urdf", basePosition=[0, 0, 0])
        
        return self.robot_id
    
    def load_bottle(self, weight_kg: float):
        """Load bottle with specified mass"""
        if self.p is None:
            return None
            
        print(f"🍾 Loading bottle: {weight_kg*1000:.1f}g")
        
        # Create bottle collision shape
        bottle_height = 0.20
        bottle_radius = 0.03
        
        collision_shape = self.p.createCollisionShape(
            self.p.GEOM_CYLINDER,
            radius=bottle_radius,
            height=bottle_height
        )
        
        visual_shape = self.p.createVisualShape(
            self.p.GEOM_CYLINDER,
            radius=bottle_radius,
            length=bottle_height,
            rgbaColor=[0.3, 0.5, 0.8, 1.0]
        )
        
        self.bottle_id = self.p.createMultiBody(
            baseMass=weight_kg,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0.5, 0, 0.3]
        )
        
        # Adjust friction for realistic behavior
        self.p.changeDynamics(
            self.bottle_id,
            -1,
            lateralFriction=0.6,
            spinningFriction=0.1,
            rollingFriction=0.01
        )
        
        return self.bottle_id
    
    def execute_flip_trajectory(self, action: np.ndarray) -> Dict:
        """
        Execute flip motion in simulation
        
        Steps:
        1. Move robot to flip starting position
        2. Execute upward trajectory
        3. Release bottle at specified time
        4. Simulate flight and landing
        5. Check final orientation
        """
        if self.p is None:
            # Fallback to placeholder
            return {
                'landed_upright': False,
                'max_height': 0.5,
                'landing_angle': 90.0,
                'final_state': np.zeros(21),
                'trajectory_duration': 0.4
            }
        
        # Extract trajectory parameters
        traj_height = action[6]
        traj_duration = action[7]
        release_timing = action[8]
        
        # Simulate trajectory (simplified)
        steps = int(traj_duration * 240)  # 240 Hz simulation
        
        for step in range(steps):
            # Apply forces to bottle (simplified throw motion)
            if step < steps * 0.3:  # Acceleration phase
                force = [0, 0, traj_height * 50]  # Upward force
                self.p.applyExternalForce(
                    self.bottle_id, -1, force,
                    [0, 0, 0], self.p.LINK_FRAME
                )
            
            self.p.stepSimulation()
            
            if self.gui:
                import time
                time.sleep(1./240.)
        
        # Wait for bottle to settle
        for _ in range(240):  # 1 second
            self.p.stepSimulation()
            if self.gui:
                import time
                time.sleep(1./240.)
        
        # Check final orientation
        bottle_pos, bottle_orn = self.p.getBasePositionAndOrientation(self.bottle_id)
        
        # Convert quaternion to euler angles
        euler = self.p.getEulerFromQuaternion(bottle_orn)
        tilt_angle = np.degrees(abs(euler[0]) + abs(euler[1]))
        
        landed_upright = tilt_angle < 15  # Within 15 degrees of vertical
        
        return {
            'landed_upright': landed_upright,
            'max_height': max(bottle_pos[2], traj_height),
            'landing_angle': tilt_angle,
            'final_state': np.concatenate([np.zeros(14), bottle_pos, bottle_orn]),
            'trajectory_duration': traj_duration
        }
    
    def reset_simulation(self, bottle_weight: float):
        """Reset simulation state"""
        if self.p is None:
            return
            
        # Remove old bottle
        if self.bottle_id is not None:
            self.p.removeBody(self.bottle_id)
        
        # Load new bottle
        self.load_bottle(bottle_weight)
    
    def close(self):
        """Clean up simulation"""
        if self.p is not None:
            self.p.disconnect()


# ============================================================================
# PHASE 3: NEURAL NETWORK POLICY
# ============================================================================

class BottleFlipPolicyNetwork(nn.Module):
    """
    Neural network for bottle flip policy
    
    Architecture:
    ┌─────────────────────────────────────┐
    │ Input: State (21D)                  │
    ├─────────────────────────────────────┤
    │ Dense(256) + ReLU + LayerNorm       │
    │ Dense(256) + ReLU + Dropout(0.2)    │
    │ Dense(128) + ReLU + LayerNorm       │
    ├─────────────────────────────────────┤
    │ ┌─────────────┐   ┌──────────────┐ │
    │ │ Action Head │   │ Value Head   │ │
    │ │ (10D)       │   │ (1D)         │ │
    │ └─────────────┘   └──────────────┘ │
    └─────────────────────────────────────┘
    
    Output: 
    - Action mean and log_std (for continuous actions)
    - State value (for PPO/A2C)
    """
    def __init__(self, state_dim=21, action_dim=10, hidden_dim=256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.action_mean = nn.Sequential(
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Log standard deviation (learnable, per action dimension)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║            POLICY NETWORK ARCHITECTURE                       ║
╠══════════════════════════════════════════════════════════════╣
║ Input Dimension:      {state_dim:38d} ║
║ Action Dimension:     {action_dim:38d} ║
║ Hidden Dimension:     {hidden_dim:38d} ║
║ Total Parameters:     {sum(p.numel() for p in self.parameters()):38d} ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def _init_weights(self, module):
        """Xavier initialization for better training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: (batch_size, state_dim) tensor
            
        Returns:
            action_mean: (batch_size, action_dim)
            action_log_std: (action_dim,)
            value: (batch_size, 1)
        """
        features = self.feature_extractor(state)
        
        action_mean = self.action_mean(features)
        value = self.value_head(features)
        
        return action_mean, self.action_log_std, value
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy
        
        Args:
            state: (state_dim,) numpy array
            deterministic: If True, return mean action (no noise)
            
        Returns:
            action: (action_dim,) numpy array
            log_prob: log probability of action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, action_log_std, value = self.forward(state_tensor)
        
        if deterministic:
            action = action_mean
        else:
            # Sample from Gaussian distribution
            action_std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
        
        # Calculate log probability (for training)
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0).numpy(), log_prob.item()
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions for training (used in PPO)
        
        Returns:
            values: state values
            log_probs: log probabilities of actions
            entropy: entropy of action distribution
        """
        action_mean, action_log_std, values = self.forward(states)
        
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return values, log_probs, entropy


class TrainingBuffer:
    """
    Experience replay buffer for training
    
    Stores: (state, action, reward, next_state, done, log_prob)
    """
    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, log_prob):
        """Add experience to buffer"""
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        
        self.buffer[self.position] = (
            state, action, reward, next_state, done, log_prob
        )
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size):
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones, log_probs = zip(
            *[self.buffer[i] for i in indices]
        )
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(log_probs)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# PHASE 4: TRAINING ALGORITHM (PPO - Proximal Policy Optimization)
# ============================================================================

class PPOTrainer:
    """
    PPO Training Algorithm
    
    PPO is chosen because:
    - Stable training for continuous actions
    - Sample efficient
    - Works well with sparse rewards
    - Industry standard for robotics
    """
    def __init__(
        self,
        policy: BottleFlipPolicyNetwork,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        
        # PPO hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Clipping parameter
        self.value_coef = value_coef  # Value loss coefficient
        self.entropy_coef = entropy_coef  # Entropy bonus coefficient
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  PPO TRAINER INITIALIZED                     ║
╠══════════════════════════════════════════════════════════════╣
║ Learning Rate:        {learning_rate:38.6f} ║
║ Gamma (γ):            {gamma:38.2f} ║
║ Clip Epsilon (ε):     {epsilon:38.2f} ║
║ Value Coefficient:    {value_coef:38.2f} ║
║ Entropy Coefficient:  {entropy_coef:38.2f} ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def compute_returns(self, rewards, dones, values, next_values):
        """
        Compute discounted returns and advantages (GAE)
        
        GAE (Generalized Advantage Estimation):
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        returns = []
        advantages = []
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae  # λ = 0.95
            
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
        
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        PPO update step
        
        Objective:
        L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
        
        where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        """
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Get current policy predictions
        values, log_probs, entropy = self.policy.evaluate_actions(states, actions)
        
        # Ratio for clipping
        ratio = torch.exp(log_probs - old_log_probs.unsqueeze(1))
        
        # Clipped surrogate objective
        surr1 = ratio * advantages.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.unsqueeze(1)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns.unsqueeze(1))
        
        # Entropy bonus (encourages exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # Gradient clipping
        self.optimizer.step()
        
        # Track statistics
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['value_loss'].append(value_loss.item())
        self.training_stats['entropy'].append(-entropy_loss.item())
        self.training_stats['total_loss'].append(loss.item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'total_loss': loss.item()
        }


# ============================================================================
# PHASE 5: CURRICULUM LEARNING
# ============================================================================

class CurriculumScheduler:
    """
    Curriculum learning: gradually increase task difficulty
    
    Stages:
    1. Fixed easy weight (200g) - Learn basic flip
    2. Fixed medium weight (300g) - Adapt to heavier bottles
    3. Fixed light weight (150g) - Handle lighter bottles
    4. Random weights (50-500g) - Generalize to all weights
    5. Adversarial weights - Handle worst-case scenarios
    """
    def __init__(self):
        self.stages = [
            {'name': 'Stage 1: Light Weight', 'weight': 0.150, 'episodes': 500},
            {'name': 'Stage 2: Medium Weight', 'weight': 0.300, 'episodes': 500},
            {'name': 'Stage 3: Heavy Weight', 'weight': 0.450, 'episodes': 500},
            {'name': 'Stage 4: Random Weights', 'weight': 'random', 'episodes': 1000},
            {'name': 'Stage 5: Adversarial', 'weight': 'adversarial', 'episodes': 500}
        ]
        
        self.current_stage = 0
        self.episodes_in_stage = 0
        
        print("""
╔══════════════════════════════════════════════════════════════╗
║              CURRICULUM LEARNING SCHEDULE                    ║
╠══════════════════════════════════════════════════════════════╣""")
        for i, stage in enumerate(self.stages):
            weight_str = f"{stage['weight']*1000:.0f}g" if isinstance(stage['weight'], float) else stage['weight']
            print(f"║ {i+1}. {stage['name']:42s} │ {weight_str:8s} ║")
        print("╚══════════════════════════════════════════════════════════════╝")
    
    def get_bottle_weight(self, success_history):
        """
        Get next bottle weight based on curriculum stage
        
        Advances to next stage when success rate > 70%
        """
        stage = self.stages[self.current_stage]
        
        # Check if should advance to next stage
        if len(success_history) >= 50:
            recent_success = np.mean(list(success_history)[-50:])
            
            if recent_success > 0.70 and self.episodes_in_stage >= 100:
                if self.current_stage < len(self.stages) - 1:
                    self.current_stage += 1
                    self.episodes_in_stage = 0
                    stage = self.stages[self.current_stage]
                    print(f"\n🎓 ADVANCING TO {stage['name']} (Success: {recent_success:.1%})")
        
        self.episodes_in_stage += 1
        
        # Return weight based on stage type
        if isinstance(stage['weight'], float):
            return stage['weight']
        elif stage['weight'] == 'random':
            return np.random.uniform(0.080, 0.530)  # 80g to 530g
        elif stage['weight'] == 'adversarial':
            # Focus on difficult weights (extremes)
            if np.random.random() < 0.5:
                return np.random.uniform(0.080, 0.150)  # Very light
            else:
                return np.random.uniform(0.450, 0.530)  # Very heavy
        
        return 0.300  # Default


# ============================================================================
# PHASE 6: MAIN TRAINING LOOP
# ============================================================================

class TrainingManager:
    """
    Manages the complete training pipeline
    
    Workflow:
    1. Initialize environment, policy, trainer
    2. Run episodes with curriculum learning
    3. Collect experiences
    4. Update policy with PPO
    5. Log metrics and save checkpoints
    6. Visualize progress
    """
    def __init__(
        self,
        env: BottleFlipEnvironment,
        policy: BottleFlipPolicyNetwork,
        trainer: PPOTrainer,
        total_episodes=3000,
        batch_size=64,
        update_frequency=10
    ):
        self.env = env
        self.policy = policy
        self.trainer = trainer
        
        self.total_episodes = total_episodes
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        self.curriculum = CurriculumScheduler()
        self.buffer = TrainingBuffer(buffer_size=10000)
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_successes = []
        self.training_history = {
            'episode': [],
            'reward': [],
            'success_rate': [],
            'policy_loss': [],
            'value_loss': []
        }
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║               TRAINING MANAGER INITIALIZED                   ║
╠══════════════════════════════════════════════════════════════╣
║ Total Episodes:       {total_episodes:38d} ║
║ Batch Size:           {batch_size:38d} ║
║ Update Frequency:     {update_frequency:38d} ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*64)
        print(" "*20 + "TRAINING START")
        print("="*64 + "\n")
        
        for episode in range(self.total_episodes):
            # Get bottle weight from curriculum
            bottle_weight = self.curriculum.get_bottle_weight(
                self.env.success_history
            )
            
            # Reset environment
            state = self.env.reset(bottle_weight=bottle_weight)
            
            episode_reward = 0
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_log_probs = []
            episode_values = []
            
            # Run episode
            done = False
            while not done:
                # Get action from policy
                action, log_prob = self.policy.get_action(state, deterministic=False)
                
                # Get value estimate
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    _, _, value = self.policy(state_tensor)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)
                episode_values.append(value.item())
                
                episode_reward += reward
                state = next_state
            
            # Store episode data
            for i in range(len(episode_states)):
                if i == len(episode_states) - 1:
                    next_value = 0  # Terminal state
                else:
                    next_value = episode_values[i + 1]
                
                self.buffer.push(
                    episode_states[i],
                    episode_actions[i],
                    episode_rewards[i],
                    next_state if i == len(episode_states) - 1 else episode_states[i + 1],
                    1 if i == len(episode_states) - 1 else 0,
                    episode_log_probs[i]
                )
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_successes.append(1.0 if info['success'] else 0.0)
            
            # Update policy
            if (episode + 1) % self.update_frequency == 0 and len(self.buffer) >= self.batch_size:
                self._update_policy()
            
            # Logging
            if (episode + 1) % 10 == 0:
                self._log_progress(episode + 1, info)
            
            # Save checkpoint
            if (episode + 1) % 500 == 0:
                self._save_checkpoint(episode + 1)
        
        print("\n" + "="*64)
        print(" "*20 + "TRAINING COMPLETE")
        print("="*64 + "\n")
        
        # Final visualization
        self._plot_training_curves()
    
    def _update_policy(self):
        """Update policy using PPO"""
        # Sample batch from buffer
        states, actions, rewards, next_states, dones, log_probs = self.buffer.sample(
            self.batch_size
        )
        
        # Get value estimates
        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        
        with torch.no_grad():
            _, _, values = self.policy(states_tensor)
            _, _, next_values = self.policy(next_states_tensor)
        
        values = values.squeeze().numpy()
        next_values = next_values.squeeze().numpy()
        
        # Compute returns and advantages
        returns, advantages = self.trainer.compute_returns(
            rewards, dones, values, next_values
        )
        
        # PPO update
        update_stats = self.trainer.update(
            states, actions, log_probs, returns, advantages
        )
        
        # Store training stats
        self.training_history['policy_loss'].append(update_stats['policy_loss'])
        self.training_history['value_loss'].append(update_stats['value_loss'])
    
    def _log_progress(self, episode, info):
        """Log training progress"""
        recent_rewards = self.episode_rewards[-10:]
        recent_successes = self.episode_successes[-100:]
        
        avg_reward = np.mean(recent_rewards)
        success_rate = np.mean(recent_successes) if recent_successes else 0.0
        
        self.training_history['episode'].append(episode)
        self.training_history['reward'].append(avg_reward)
        self.training_history['success_rate'].append(success_rate)
        
        print(f"""
[Episode {episode:4d}] Reward: {avg_reward:7.2f} | Success: {success_rate:5.1%} | Weight: {info['bottle_weight']*1000:5.1f}g | Height: {info['flip_height']:.2f}m
        """.strip())
    
    def _save_checkpoint(self, episode):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'training_history': self.training_history,
            'success_rate': np.mean(self.episode_successes[-100:]) if self.episode_successes else 0.0
        }
        
        filename = f"checkpoint_episode_{episode}.pt"
        torch.save(checkpoint, filename)
        print(f"💾 Checkpoint saved: {filename}")
    
    def _plot_training_curves(self):
        """Visualize training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success rate
        axes[0, 0].plot(self.training_history['episode'], 
                       self.training_history['success_rate'], 
                       linewidth=2, color='green')
        axes[0, 0].set_title('Success Rate Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # Average reward
        axes[0, 1].plot(self.training_history['episode'], 
                       self.training_history['reward'], 
                       linewidth=2, color='blue')
        axes[0, 1].set_title('Average Reward', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Policy loss
        if self.training_history['policy_loss']:
            axes[1, 0].plot(self.training_history['policy_loss'], 
                           linewidth=2, color='red', alpha=0.7)
            axes[1, 0].set_title('Policy Loss', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Value loss
        if self.training_history['value_loss']:
            axes[1, 1].plot(self.training_history['value_loss'], 
                           linewidth=2, color='orange', alpha=0.7)
            axes[1, 1].set_title('Value Loss', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("📊 Training curves saved: training_curves.png")


# ============================================================================
# PHASE 7: DEPLOYMENT & REAL ROBOT INTERFACE
# ============================================================================

class RealRobotInterface:
    """
    Interface for deploying to real UR10e robot
    
    Note: Requires urx library
          pip install urx
    """
    def __init__(self, robot_ip="192.168.1.100", use_real_robot=False):
        self.robot_ip = robot_ip
        self.use_real_robot = use_real_robot
        
        if use_real_robot:
            try:
                import urx
                self.robot = urx.Robot(robot_ip)
                print(f"✅ Connected to UR10e at {robot_ip}")
            except Exception as e:
                print(f"⚠️  Could not connect to robot: {e}")
                print("   Running in simulation mode")
                self.use_real_robot = False
        else:
            print("🔧 Running in simulation mode (no real robot)")
        
    def execute_flip(self, action: np.ndarray, bottle_weight: float):
        """Execute flip on real robot"""
        if not self.use_real_robot:
            print(f"[SIM] Executing flip with weight {bottle_weight*1000:.1f}g")
            # Simulate execution
            import time
            time.sleep(0.5)
            success = np.random.random() < 0.7  # 70% success rate
            return success
        
        # Real robot execution
        try:
            # Extract trajectory parameters
            joint_velocities = action[0:6]
            traj_height = action[6]
            traj_duration = action[7]
            release_timing = action[8]
            force_mult = action[9]
            
            # Move to starting position
            print("Moving to start position...")
            self.robot.movej([0, -1.57, 1.57, -1.57, -1.57, 0], vel=0.5, acc=0.5)
            
            # Execute flip trajectory
            print(f"Executing flip (height: {traj_height:.2f}m, duration: {traj_duration:.2f}s)...")
            
            # This is simplified - real implementation would use speedj or servoj
            # for smooth trajectory control
            
            # Close gripper (adjust force based on weight)
            grip_force = 50 * force_mult
            print(f"Gripping with force: {grip_force:.1f}N")
            
            # Execute upward motion
            import time
            start_time = time.time()
            
            # Simple trajectory - in production, use proper motion planning
            while time.time() - start_time < traj_duration:
                self.robot.speedj(joint_velocities.tolist(), acc=2.0, t=0.1)
                time.sleep(0.1)
            
            # Release at specified time
            time.sleep(release_timing)
            print("Releasing bottle...")
            
            # Open gripper
            # self.robot.set_digital_out(0, True)  # Example gripper control
            
            # Wait for bottle to land
            time.sleep(2.0)
            
            # Check success (would use camera/vision system)
            print("Checking landing...")
            success = self._check_landing_success()
            
            return success
            
        except Exception as e:
            print(f"❌ Execution error: {e}")
            return False
    
    def _check_landing_success(self):
        """
        Check if bottle landed upright
        
        In production, would use:
        - Camera/vision system
        - Capacitive sensors
        - Force sensors on table
        """
        # Placeholder - manual verification or vision system
        response = input("Did the bottle land upright? (y/n): ")
        return response.lower() == 'y'
    
    def close(self):
        """Disconnect from robot"""
        if self.use_real_robot and hasattr(self, 'robot'):
            self.robot.close()
            print("🔌 Disconnected from robot")


class DeploymentManager:
    """
    Manages deployment to real robot with safety checks
    """
    def __init__(self, policy_path: str, robot_ip: str = "192.168.1.100"):
        # Load trained policy
        self.policy = BottleFlipPolicyNetwork()
        checkpoint = torch.load(policy_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              DEPLOYMENT MANAGER INITIALIZED                  ║
╠══════════════════════════════════════════════════════════════╣
║ Policy Loaded: {policy_path:42s} ║
║ Success Rate:  {checkpoint['success_rate']:42.1%} ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Connect to robot
        self.robot = RealRobotInterface(robot_ip, use_real_robot=False)
        
        # Safety limits
        self.max_attempts_per_weight = 3
        self.emergency_stop = False
    
    def run_flip(self, bottle_weight: float):
        """Run a single flip attempt"""
        if self.emergency_stop:
            print("⛔ Emergency stop activated. Reset required.")
            return False
        
        # Create state observation
        state = np.zeros(21)
        state[12] = bottle_weight  # Set bottle weight
        
        # Normalize state
        state_space = StateSpace(RobotConfig(), BottleConfig())
        state = state_space.normalize_state(state)
        
        # Get action from policy (deterministic for deployment)
        action, _ = self.policy.get_action(state, deterministic=True)
        
        # Scale action to real robot limits
        action_space = ActionSpace(RobotConfig())
        action = action_space.clip_action(action)
        
        print(f"\n{'='*64}")
        print(f"FLIP ATTEMPT - Bottle Weight: {bottle_weight*1000:.1f}g")
        print(f"{'='*64}")
        print(f"Trajectory Height: {action[6]:.3f}m")
        print(f"Trajectory Duration: {action[7]:.3f}s")
        print(f"Release Timing: {action[8]:+.3f}s")
        print(f"Force Multiplier: {action[9]:.2f}x")
        print(f"{'='*64}\n")
        
        # Execute on robot
        success = self.robot.execute_flip(action, bottle_weight)
        
        if success:
            print("✅ SUCCESS - Bottle landed upright!")
        else:
            print("❌ FAILED - Bottle did not land upright")
        
        return success
    
    def run_test_suite(self):
        """Run comprehensive test across weight range"""
        test_weights = np.linspace(0.080, 0.530, 10)  # 10 test weights
        
        results = []
        
        print("\n" + "="*64)
        print(" "*18 + "RUNNING TEST SUITE")
        print("="*64 + "\n")
        
        for i, weight in enumerate(test_weights):
            print(f"\n[Test {i+1}/10] Weight: {weight*1000:.1f}g")
            
            successes = 0
            for attempt in range(self.max_attempts_per_weight):
                success = self.run_flip(weight)
                if success:
                    successes += 1
                
                if self.emergency_stop:
                    break
            
            success_rate = successes / self.max_attempts_per_weight
            results.append({
                'weight': weight,
                'successes': successes,
                'attempts': self.max_attempts_per_weight,
                'success_rate': success_rate
            })
            
            print(f"Weight {weight*1000:.1f}g: {successes}/{self.max_attempts_per_weight} successful ({success_rate:.1%})")
            
            if self.emergency_stop:
                break
        
        # Summary
        print("\n" + "="*64)
        print(" "*20 + "TEST RESULTS")
        print("="*64)
        
        for result in results:
            print(f"Weight {result['weight']*1000:5.1f}g: {result['success_rate']:6.1%} "
                  f"({result['successes']}/{result['attempts']})")
        
        overall_success = sum(r['successes'] for r in results) / sum(r['attempts'] for r in results)
        print(f"\nOverall Success Rate: {overall_success:.1%}")
        print("="*64 + "\n")
        
        return results
    
    def emergency_stop_handler(self):
        """Activate emergency stop"""
        self.emergency_stop = True
        print("\n⛔⛔⛔ EMERGENCY STOP ACTIVATED ⛔⛔⛔\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - demonstrates complete workflow
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        UR10e BOTTLE FLIP - MACHINE LEARNING PIPELINE             ║
    ║                                                                  ║
    ║  Complete workflow from simulation training to deployment        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ========================================================================
    # PHASE 1: Setup
    # ========================================================================
    print("\n" + "█"*64)
    print("  PHASE 1: ENVIRONMENT & NETWORK SETUP")
    print("█"*64 + "\n")
    
    # Create environment
    env = BottleFlipEnvironment(mode='simulation')
    
    # Create policy network
    policy = BottleFlipPolicyNetwork(
        state_dim=21,
        action_dim=10,
        hidden_dim=256
    )
    
    # Create trainer
    trainer = PPOTrainer(
        policy=policy,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=0.2
    )
    
    # ========================================================================
    # PHASE 2: Training
    # ========================================================================
    print("\n" + "█"*64)
    print("  PHASE 2: TRAINING IN SIMULATION")
    print("█"*64 + "\n")
    
    # Create training manager
    training_manager = TrainingManager(
        env=env,
        policy=policy,
        trainer=trainer,
        total_episodes=100,  # Reduced for demo - use 3000 for real training
        batch_size=64,
        update_frequency=10
    )
    
    # Train the policy
    training_manager.train()
    
    # ========================================================================
    # PHASE 3: Testing
    # ========================================================================
    print("\n" + "█"*64)
    print("  PHASE 3: TESTING TRAINED POLICY")
    print("█"*64 + "\n")
    
    # Test on random weights
    test_weights = [0.100, 0.200, 0.300, 0.400, 0.500]
    test_results = []
    
    for weight in test_weights:
        state = env.reset(bottle_weight=weight)
        action, _ = policy.get_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        
        test_results.append({
            'weight': weight,
            'success': info['success'],
            'reward': reward
        })
        
        status = "✅" if info['success'] else "❌"
        print(f"{status} Weight: {weight*1000:5.1f}g | Reward: {reward:7.2f} | Success: {info['success']}")
    
    # ========================================================================
    # PHASE 4: Deployment (Optional)
    # ========================================================================
    print("\n" + "█"*64)
    print("  PHASE 4: DEPLOYMENT TO REAL ROBOT")
    print("█"*64 + "\n")
    
    deploy_to_real = input("Deploy to real robot? (y/n): ").lower() == 'y'
    
    if deploy_to_real:
        # Save trained model
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'success_rate': np.mean([r['success'] for r in test_results])
        }, 'trained_policy.pt')
        
        # Create deployment manager
        deployment = DeploymentManager(
            policy_path='trained_policy.pt',
            robot_ip="192.168.1.100"
        )
        
        # Run test suite
        deployment.run_test_suite()
    else:
        print("Skipping real robot deployment.")
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║                    PIPELINE COMPLETE! 🎉                         ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    # Run the complete pipeline
    main()