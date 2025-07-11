# robotis_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**robotis_lab** is a research-oriented repository based on [Isaac Lab](https://isaac-sim.github.io/IsaacLab), designed to enable reinforcement learning and imitation learning experiments using Robotis robots in simulation.
This project provides simulation environments, configuration tools, and task definitions tailored for Robotis hardware, leveraging NVIDIA Isaac Sim’s powerful GPU-accelerated physics engine and Isaac Lab’s modular RL pipeline.

> [!IMPORTANT]
> This repository currently depends on **IsaacLab v2.0.0** or higher.
>

## Installation

1. Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to set up the environment.  
  Instead of the recommended local installation, you can run Isaac Lab in a Docker container to simplify dependency management and ensure consistency across systems.

2. Clone the Isaac Lab Repository:
  ```bash
  git clone https://github.com/isaac-sim/IsaacLab.git
  ```

3. Start and enter the Docker container:
  ```bash
  # start
  ./IsaacLab/docker/container.py start base

  # enter
  ./IsaacLab/docker/container.py enter base
  ```

4. Clone the robotis_lab repository (outside the IsaacLab directory):

  ```bash
  cd /workspace && git clone https://github.com/ROBOTIS-GIT/robotis_lab.git
  ```

5. Install the robotis_lab Package.

  ```bash
  cd robotis_lab && python -m pip install -e source/robotis_lab
  ```

6. Verify that the extension is correctly installed by listing all available environments:

  ```bash
  python scripts/tools/list_envs.py
  ```

## Try examples
![ISAAC_FFW_RL_reach_train (1)](https://github.com/user-attachments/assets/679ce5ce-eef9-4f55-bc03-55f41ca77af9)

> [!NOTE]
> If you want to control a **SINGLE ROBOT** with the keyboard during playback, add `--keyboard` at the end of the play script.
>
> ```
> Key bindings:
> =========================== =========================
> Command                     Key
> =========================== =========================
> Toggle gripper (open/close) K      
> Move arm along x-axis       W / S   
> Move arm along y-axis       A / D
> Move arm along z-axis       Q / E
> Rotate arm along x-axis     Z / X
> Rotate arm along y-axis     T / G
> Rotate arm along z-axis     C / V
> =========================== =========================
> ```

### Reinforcement learning

OMY Reach task

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotisLab-Reach-OMY-v0 --num_envs=1024 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotisLab-Reach-OMY-v0 --num_envs=16
```

OMY Lift task

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotisLab-Lift-Cube-OMY-v0 --num_envs=1024 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotisLab-Lift-Cube-OMY-v0 --num_envs=16
```

OMY Open drawer task

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotisLab-Open-Drawer-OMY-v0 --num_envs=1024 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotisLab-Open-Drawer-OMY-v0 --num_envs=16
```

FFW-BG2 reach task

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotisLab-Reach-FFW-BG2-v0 --num_envs=1024 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotisLab-Reach-FFW-BG2-v0 --num_envs=16
```

### Imitation learning

OMY Stack task (Stack the blocks in the following order: blue → red → green.)

```bash
# Teleop
python scripts/tools/record_demos.py --task RobotisLab-Stack-Cube-OMY-IK-Rel-v0 --teleop_device keyboard --dataset_file ./datasets/dataset.hdf5 --num_demos 10

# Annotate
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --device cuda --task RobotisLab-Stack-Cube-OMY-IK-Rel-Mimic-v0 --auto --input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5 --headless

# Mimic data
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--device cuda --num_envs 100 --generation_num_trials 1000 \
--input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5 --headless

# Train
python scripts/imitation_learning/robomimic/train.py \
--task RobotisLab-Stack-Cube-OMY-IK-Rel-v0 --algo bc \
--dataset ./datasets/generated_dataset.hdf5

# Play
python scripts/imitation_learning/robomimic/play.py \
--device cuda --task RobotisLab-Stack-Cube-OMY-IK-Rel-v0 --num_rollouts 50 \
--checkpoint /PATH/TO/desired_model_checkpoint.pth
```

### Sim2Real Deployment
We provide a Sim2Real pipeline to deploy policies trained in Isaac Lab simulation directly onto the real OMY robot.

<details>
<summary>🎥 Show demo video</summary>

![OMY Sim2Real Demo](https://github.com/user-attachments/assets/c7bd5dac-1578-4ba1-845f-afaeefc1677c)

</details>

> [!IMPORTANT]
> More on OMY Hardware Setup:
> For details on how to set up and operate the OMY robot, please refer to the [open_manipulator](https://isaac-sim.github.io/IsaacLab)

In this pipeline:
- The trained policy (exported as a TorchScript .pt file) is executed on the real robot using ROS 2.
- The robot receives joint state feedback and sends joint trajectory commands via a ROS 2 control interface.
- A TF frame for the sampled target pose is broadcast for visualization and debugging.

Prerequisites
- A trained policy (under logs/rsl_rl/reach_omy/<TIMESTAMP>).
- ROS 2 Humble (or later) installed and sourced.
- Robot hardware must be ready and controllable via the joint trajectory interface.

Run Sim2Real Reach Policy on OMY
```bash
python3 scripts/sim2real/OMY/reach/run_omy_reach.py --model_dir=2025-07-10_08-47-09
```

Replace <2025-07-10_08-47-09> with the actual timestamp folder name under:
```bash
logs/rsl_rl/reach_omy/
```

OMY Reach task

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotisLab-Reach-OMY-v0 --num_envs=1024 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotisLab-Reach-OMY-v0 --num_envs=16

# Sim2Real
python3 scripts/sim2real/OMY/reach/run_omy_reach.py --model_dir=<2025-07-10_08-47-09>
```
