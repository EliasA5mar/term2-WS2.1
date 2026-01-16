# Hardware Specifications for UR10e Bottle Flip Project
## Cameras, Sensors, and Equipment Guide

---

##  CAMERA COMPARISON & RECOMMENDATIONS

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           CAMERA TYPE COMPARISON                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                  │   RGB-D CAMERA   │  STEREO CAMERA   │     LiDAR        │
│                  │   (Recommended)  │                  │                  │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Best For         │ Object detection │ Outdoor/variable │ Large workspace  │
│                  │ Pose estimation  │ lighting         │ mapping          │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Depth Method     │ Structured light │ Triangulation    │ Time-of-flight   │
│                  │ or ToF           │ (2 cameras)      │ laser scanning   │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Range            │ 0.2m - 10m       │ 0.5m - 20m       │ 0.1m - 100m+     │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Accuracy         │ ±1-2mm @ 1m      │ ±2-5mm @ 1m      │ ±2-5mm           │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Frame Rate       │ 30-90 fps        │ 30-60 fps        │ 10-20 Hz         │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Works in Sun      │   Poor          │   Good          │   Moderate      │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Texture-less     │  Good          │   Poor          │   Excellent     │
│ Objects          │                  │                  │                  │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Cost             │ $200 - $500      │ $300 - $1000     │ $500 - $5000+    │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ ROS2 Support     │   Excellent     │   Good          │   Good          │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘

 RECOMMENDATION FOR BOTTLE FLIP: RGB-D Camera (Intel RealSense D435i)
   - Best price/performance for indoor robotics
   - Excellent for bottle detection and pose estimation
   - Built-in IMU for motion tracking
   - Native ROS2 support
```

---

##  RECOMMENDED CAMERA: Intel RealSense D435i

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    INTEL REALSENSE D435i SPECIFICATIONS                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │    ┌───┐  ┌───────────────────────────────┐  ┌───┐        │
    │    │ L │  │      RGB Camera               │  │ R │        │
    │    │ I │  │      1920x1080 @ 30fps        │  │ I │        │
    │    │ R │  │                               │  │ R │        │
    │    └───┘  └───────────────────────────────┘  └───┘        │
    │   Left IR        Depth: 1280x720 @ 90fps     Right IR     │
    │   Projector                                   Sensor       │
    │                                                             │
    │              [ IMU - Accelerometer + Gyroscope ]           │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  SPECIFICATIONS                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  DEPTH SENSOR:                           RGB SENSOR:                                    │
│  • Resolution: 1280x720 (up to)          • Resolution: 1920x1080                       │
│  • Frame rate: Up to 90 fps              • Frame rate: Up to 30 fps                    │
│  • Min depth: 0.2m (20cm)                • Field of View: 69° x 42°                    │
│  • Max depth: 10m                        • Auto-exposure                               │
│  • Field of View: 87° x 58°                                                            │
│                                                                                         │
│  IMU (BUILT-IN):                         PHYSICAL:                                      │
│  • Accelerometer: ±4g                    • Size: 90mm x 25mm x 25mm                    │
│  • Gyroscope: ±1000°/s                   • Weight: 72g                                 │
│  • Update rate: 200Hz                    • Mount: 1/4"-20 tripod                       │
│                                          • Interface: USB 3.0                          │
│                                                                                         │
│  PRICE: ~$350 USD                        ROS2 PACKAGE: realsense2_camera              │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

WHY D435i FOR BOTTLE FLIP:
IMU helps track bottle motion during flip
High frame rate (90fps) captures fast motions
Good depth accuracy for grasp planning
RGB for color-based bottle detection (your blue bottle detection!)
Compact size - easy to mount on robot or workspace
Excellent ROS2 integration

```
##  COMPLETE SENSOR SUITE


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE SENSOR CONFIGURATION                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘


                         ┌─────────────────────────────────┐
                         │       WORKSPACE OVERVIEW        │
                         └─────────────────────────────────┘

                              [Overhead Camera]
                                    │
                                    │ (Optional: workspace monitoring)
                                    ▼
              ┌─────────────────────────────────────────────────┐
              │                                                 │
              │    ┌─────────────────────────────────────┐     │
              │    │                                     │     │
              │    │         ROBOT WORKSPACE             │     │
    [Side     │    │                                     │     │    [Side
    Camera]──►│    │     ┌───────────────────┐          │     │◄── Camera]
              │    │     │                   │          │     │   (Optional)
              │    │     │    UR10e Robot    │          │     │
              │    │     │    + Gripper      │          │     │
              │    │     │                   │          │     │
              │    │     └─────────┬─────────┘          │     │
              │    │               │                     │     │
              │    │        [Wrist Camera]              │     │
              │    │         (Eye-in-hand)              │     │
              │    │               │                     │     │
              │    │               ▼                     │     │
              │    │         ┌─────────┐                │     │
              │    │         │ Bottle  │                │     │
              │    │         └─────────┘                │     │
              │    │                                     │     │
              │    │    [Force/Torque Sensor]           │     │
              │    │                                     │     │
              │    └─────────────────────────────────────┘     │
              │                                                 │
              └─────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  SENSOR PLACEMENT OPTIONS                                                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  OPTION A: EYE-IN-HAND (Camera on Robot Wrist)  RECOMMENDED                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Pros:                              │  Cons:                                    │   │
│  │  • Camera moves with robot          │  • Limited field of view                  │   │
│  │  • Close-up detailed views          │  • Can't see bottle during flip          │   │
│  │  • Good for grasp planning          │  • Motion blur during fast moves         │   │
│  │  • Simpler calibration              │  • Added weight on wrist                 │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
│  OPTION B: EYE-TO-HAND (Fixed External Camera)                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Pros:                              │  Cons:                                    │   │
│  │  • Wide workspace view              │  • Occlusion from robot                   │   │
│  │  • Can track bottle in flight       │  • Needs hand-eye calibration            │   │
│  │  • No added robot weight            │  • Fixed perspective                      │   │
│  │  • Higher frame rate possible       │  • May need multiple cameras             │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
│  OPTION C: HYBRID (Both)  BEST FOR BOTTLE FLIP                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  • Wrist camera for grasp planning                                              │   │
│  │  • External camera for tracking bottle in flight                                │   │
│  │  • Best of both worlds                                                          │   │
│  │  • More complex but optimal results                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```
```
 
##  SENSOR SPECIFICATIONS DETAIL

### Force/Torque Sensors

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           FORCE/TORQUE SENSOR OPTIONS                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

PURPOSE: Detect grip force on bottle, sense contact, improve grasp learning

┌──────────────────────┬─────────────────────┬─────────────────────┬────────────────────┐
│ Sensor               │ Robotiq FT300       │ ATI Mini45          │ OnRobot HEX        │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Axes                 │ 6-axis (Fx,Fy,Fz,   │ 6-axis              │ 6-axis             │
│                      │ Tx,Ty,Tz)           │                     │                    │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Force Range          │ ±300N               │ ±290N               │ ±200N              │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Torque Range         │ ±30Nm               │ ±10Nm               │ ±10Nm              │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Resolution           │ 0.1N / 0.001Nm      │ 1/16N / 1/8Nmm      │ 0.2N / 0.02Nm      │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Sample Rate          │ 100Hz               │ 7000Hz              │ 1000Hz             │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ UR Integration       │  Native      │  Adapter        │  Good         │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Price                │ ~$2,500             │ ~$3,500             │ ~$2,000            │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ ROS2 Support         │  Good          │ Excellent    │  Moderate      │
└──────────────────────┴─────────────────────┴─────────────────────┴────────────────────┘

🎯 RECOMMENDATION: Robotiq FT300 (if available in lab)
   - Native UR integration
   - Easy to use with ROS2
   - Good for detecting bottle grip and release
```
```
### Gripper Options

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              GRIPPER OPTIONS                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘

PURPOSE: Grasp and release bottle at precise timing

┌──────────────────────┬─────────────────────┬─────────────────────┬────────────────────┐
│ Gripper              │ Robotiq 2F-85       │ Robotiq 2F-140      │ OnRobot RG2        │
│                      │  RECOMMENDED       │                     │                    │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Stroke               │ 85mm                │ 140mm               │ 110mm              │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Grip Force           │ 20-235N             │ 10-125N             │ 3-40N              │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Close Speed          │ 20-150mm/s          │ 30-250mm/s          │ Variable           │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Repeatability        │ ±0.05mm             │ ±0.05mm             │ ±0.1mm             │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Bottle Suitable      │  Yes (typical     │  Yes (larger      │  Gentle grip     │
│                      │ water bottle)       │ bottles)            │ only               │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ UR Integration       │                │              │                │
├──────────────────────┼─────────────────────┼─────────────────────┼────────────────────┤
│ Price                │ ~$5,000             │ ~$6,000             │ ~$3,000            │
└──────────────────────┴─────────────────────┴─────────────────────┴────────────────────┘


BOTTLE GRIP REQUIREMENTS:
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Typical water bottle: ~65mm diameter                                                   │
│  Solán de Cabras bottle: ~70mm diameter (wider at top)                                 │
│                                                                                         │
│  Required grip force: ~10-30N (secure but not crushing)                                │
│  Required release speed: FAST (<50ms for clean release)                                │
│                                                                                         │
│  CRITICAL: Gripper must support FAST OPEN for bottle flip release!                     │
│            Some grippers have slow open speed - verify specs!                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

##  SENSOR DATA FLOW

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           SENSOR DATA FLOW IN SYSTEM                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘


    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │   RGB-D Camera  │          │  Force/Torque   │          │    Gripper      │
    │   (RealSense)   │          │    Sensor       │          │   (Robotiq)     │
    └────────┬────────┘          └────────┬────────┘          └────────┬────────┘
             │                            │                            │
             │ RGB Image                  │ 6-axis F/T                │ Position
             │ Depth Image                │ @ 100Hz                    │ Current
             │ IMU Data                   │                            │ @ 125Hz
             │ @ 30-90Hz                  │                            │
             │                            │                            │
             ▼                            ▼                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                              ROS2 TOPICS                                     │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │  /camera/color/image_raw ─────────► Image (RGB)                            │
    │  /camera/depth/image_rect_raw ────► Image (Depth)                          │
    │  /camera/imu ─────────────────────► Imu                                    │
    │  /camera/aligned_depth_to_color ──► Image (Aligned Depth)                  │
    │                                                                             │
    │  /ft_sensor/wrench ───────────────► WrenchStamped (F/T data)              │
    │                                                                             │
    │  /gripper/state ──────────────────► GripperState                          │
    │  /gripper/command ────────────────► GripperCommand                         │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
             │                            │                            │
             ▼                            ▼                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           PROCESSING MODULES                                 │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
    │  │   PERCEPTION    │   │    GRASPING     │   │    PHYSICS      │          │
    │  │   MODULE        │   │    MODULE       │   │    MODULE       │          │
    │  │                 │   │                 │   │                 │          │
    │  │ • Bottle detect │   │ • Force control │   │ • Trajectory    │          │
    │  │ • Pose estimate │   │ • Grasp plan    │   │   optimization  │          │
    │  │ • Tracking      │   │ • Release time  │   │ • Release calc  │          │
    │  │                 │   │                 │   │                 │          │
    │  │ Owner: Heleri   │   │ Owner: Arthur   │   │ Owner: Elias    │          │
    │  └─────────────────┘   └─────────────────┘   └─────────────────┘          │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                        MOTION GENERATION                                     │
    │                     (Amaro's Module)                                         │
    │                                                                             │
    │  Inputs:   bottle_pose, grasp_config, release_params                        │
    │  Outputs:  joint_trajectory → UR10e Robot                                   │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

##  CAMERA MOUNTING SPECIFICATIONS

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           CAMERA MOUNTING OPTIONS                                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘


OPTION A: WRIST MOUNT (Eye-in-hand)
═══════════════════════════════════

                    UR10e Wrist
                         │
              ┌──────────┴──────────┐
              │    Tool Flange      │
              │    (ISO 9409-1-50)  │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │   Camera Bracket    │  ◄── 3D printed or machined
              │   (Custom mount)    │      aluminum bracket
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │   RealSense D405    │  ◄── Compact camera
              │   or D435i          │      (42x42mm or 90x25mm)
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │   Robotiq Gripper   │
              └─────────────────────┘

    Mounting considerations:
    • Camera should NOT obstruct gripper view
    • Angle camera ~30-45° downward for grasp view
    • Route USB cable along robot arm
    • Use strain relief at wrist
    • Weight: Add to tool center point (TCP) calculation


OPTION B: EXTERNAL MOUNT (Eye-to-hand)
══════════════════════════════════════

    Side view:                          Top view:

         [Camera]                       ┌─────────────────────┐
             \                          │                     │
              \  ~45°                   │    Robot            │
               \                        │    Workspace        │
                \                       │                     │
    ┌────────────\────────┐            │         ●           │
    │             ▼       │            │      (Robot)        │
    │    Robot Workspace  │            │                     │
    │         ●           │            └─────────────────────┘
    │      (Robot)        │                       │
    │                     │                       │
    │      ○ (Bottle)     │             [Camera] ◄┘
    │                     │            (Looking at workspace)
    └─────────────────────┘

    Mounting options:
    • Tripod with articulating arm
    • Wall/ceiling mount
    • 80/20 aluminum extrusion frame
    • Recommended height: 1-1.5m above table
    • Recommended distance: 1-2m from workspace center


CAMERA CALIBRATION REQUIREMENTS:
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│  INTRINSIC CALIBRATION (Camera internal parameters):                                   │
│  • Use checkerboard pattern (8x6 or 9x7)                                               │
│  • ROS2 package: camera_calibration                                                    │
│  • Output: camera_info.yaml (focal length, distortion, etc.)                          │
│                                                                                         │
│  EXTRINSIC CALIBRATION (Camera-to-Robot transform):                                    │
│  • Eye-in-hand: Move robot to known poses, record transforms                           │
│  • Eye-to-hand: Use ArUco markers or known object positions                           │
│  • ROS2 package: easy_handeye2                                                         │
│  • Output: TF transform from camera_frame to robot_base_link                          │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

##  FINAL RECOMMENDATION SUMMARY

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           RECOMMENDED SENSOR CONFIGURATION                               │
│                              (For Academic Project)                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│  MUST HAVE:                                                                            │
│  ═══════════                                                                           │
│  ☑ Intel RealSense D435i .......................... $350                               │
│    - Primary camera for detection + depth                                              │
│    - Eye-to-hand mount (external, looking at workspace)                                │
│    - Tracks bottle position for grasp planning                                         │
│    - IMU helps with motion estimation                                                  │
│                                                                                         │
│  ☑ UR10e Robot + Controller ....................... (Lab equipment)                    │
│    - 6-DOF industrial robot arm                                                        │
│    - ROS2 driver available                                                             │
│                                                                                         │
│  ☑ Gripper (Robotiq 2F-85 or similar) ............. (Lab equipment)                    │
│    - Fast open/close for bottle release                                                │
│    - Position feedback                                                                 │
│                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  NICE TO HAVE:                                                                         │
│  ══════════════                                                                        │
│  ☐ Second camera for tracking flip ................ $70-350                            │
│    - Logitech C920 (budget) or another RealSense                                       │
│    - Positioned to see bottle trajectory in air                                        │
│                                                                                         │
│  ☐ Force/Torque sensor ............................ (Lab equipment)                    │
│    - Helps optimize grip force                                                         │
│    - Detects successful grasp                                                          │
│                                                                                         │
│  ☐ LED lighting ................................... $30                                │
│    - Consistent lighting for detection                                                 │
│    - Reduces shadows                                                                   │
│                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  TOTAL ESTIMATED COST (if buying):                                                     │
│  ═════════════════════════════════                                                     │
│  Minimum viable:  ~$500   (webcam + mounting)                                          │
│  Recommended:     ~$800   (RealSense D435i + mounting)                                 │
│  Optimal:         ~$1500  (Multiple cameras + sensors)                                 │
│                                                                                         │
│  Note: Check what's available in your robotics lab first!                              │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---



*Document created for: Robotic Bottle Flipping Project - Group 1*
