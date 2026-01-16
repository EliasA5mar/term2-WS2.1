# ROBOTIC BOTTLE FLIP 
### Group 1
#### Team Members 
> - **Arthur Rotstien**
> - **Amaro Bravo**
> - **Heleri Koltsin**
> - **Elias Asmar**
![alt text](ezgif-479d2f691f2488f2.gif)

# Robotic Bottle Flipping

An autonomous robotic system that performs bottle flipping using a robotic arm and gripper. The system integrates perception, manipulation, motion planning, and machine learning to achieve consistent bottle flips.

## Overview

This project demonstrates a complete robotic manipulation pipeline for the challenging task of bottle flipping. The system perceives the bottle state, plans and executes throwing motions, and learns from outcomes to improve performance over time.

## System Architecture

The system is composed of five interconnected modules:

### 1. Perception Module (Heleri)
Detects the bottle's position, orientation, and fill level using computer vision and sensor fusion.

**Key Functions:**
- Bottle localization in 3D space
- Orientation estimation
- Fill level detection
- Real-time state monitoring

### 2. Grasping Module (Arthur)
Determines optimal grip position and force to ensure stable yet releasable control of the bottle.

**Key Functions:**
- Grasp point computation
- Force profile calculation
- Grip stability verification
- Release timing coordination

### 3. Motion Generation Module (Amaro)
Produces a throwing trajectory defined by velocity, angle, and release timing parameters.

**Key Functions:**
- Trajectory planning
- Velocity profile generation
- Release angle optimization
- Motion parameter tuning

### 4. Physics Interaction Phase (Elias)
After release, the bottle follows passive dynamics governed by gravity and internal fluid motion.

**Key Functions:**
- Ballistic trajectory simulation
- Fluid dynamics modeling
- Landing prediction
- Outcome state tracking

### 5. Evaluation & Learning Module
Assesses flip outcomes and updates future motion parameters through reinforcement learning.

**Key Functions:**
- Success/failure classification
- Performance metrics computation
- Parameter optimization
- Continuous improvement loop

## Features

- Real-time bottle detection and tracking
- Adaptive grasping based on bottle properties
- Parameterized motion generation
- Physics-based trajectory prediction
- Self-improving through learning

## Requirements

### Hardware
- Robotic arm (6-DOF or higher recommended)
- Gripper with force feedback
- RGB-D camera or vision system
- Computing platform (GPU recommended for vision processing)

### Software
- Python 3.8+
- ROS (Robot Operating System)
- OpenCV for vision processing
- PyTorch or TensorFlow for learning module
- Robot-specific control libraries

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/robotic-bottle-flipping.git
cd robotic-bottle-flipping

# Install dependencies
pip install -r requirements.txt

# Build ROS packages (if using ROS)
catkin_make
source devel/setup.bash
```

## Usage

```bash
# Launch the complete system
python main.py

# Run individual modules
python -m modules.perception
python -m modules.grasping
python -m modules.motion_generation
python -m modules.evaluation
```

## Configuration

Modify `config/system_config.yaml` to adjust:
- Camera calibration parameters
- Robot kinematic limits
- Grasp force thresholds
- Motion generation parameters
- Learning rates and rewards

## Pipeline Flow

```
┌─────────────┐     ┌──────────┐     ┌───────────────┐
│ Perception  │────▶│ Grasping │────▶│    Motion     │
│   (Heleri)  │     │ (Arthur) │     │  Generation   │
└─────────────┘     └──────────┘     │   (Amaro)     │
                                     └───────┬───────┘
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │   Physics     │
                                     │ Interaction   │
                                     │   (Elias)     │
                                     └───────┬───────┘
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │  Evaluation   │
                                     │  & Learning   │
                                     └───────────────┘
```

## Results

The system achieves successful bottle flips through iterative learning, improving flip success rate over time by adjusting motion parameters based on previous attempts.

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Module development team: Heleri, Arthur, Amaro, Elias
- Robotics and AI research community
- Open-source libraries and frameworks

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{robotic_bottle_flipping,
  title={Robotic Bottle Flipping: An Integrated Approach},
  author={group 1},
  year={2026},
  url={https://github.com/yourusername/robotic-bottle-flipping}
}
```

## Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.
