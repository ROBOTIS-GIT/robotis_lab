# robotis_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**robotis_lab** is a research-oriented repository based on [Isaac Lab](https://isaac-sim.github.io/IsaacLab), designed to enable reinforcement learning (RL) and imitation learning (IL) experiments using Robotis robots in simulation.
This project provides simulation environments, configuration tools, and task definitions tailored for Robotis hardware, leveraging NVIDIA Isaac Sim’s powerful GPU-accelerated physics engine and Isaac Lab’s modular RL pipeline.

> [!IMPORTANT]
> This repository currently depends on **IsaacLab v2.0.0** or higher.
>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone the robotis_lab Repository (i.e. outside the `IsaacLab` directory):

  ```bash
  git clone https://github.com/ROBOTIS-GIT/robotis_lab.git
  ```

- Install the robotis_lab Package

  ```bash
  cd robotis_lab
  python -m pip install -e source/robotis_lab
  ```

- Verify that the extension is correctly installed by running the following command to print all the available environments in the extension:

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
python scripts/reinforcement_learning/skrl/train.py --task RobotisLab-Reach-OMY-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/skrl/play.py --task RobotisLab-Reach-OMY-v0 --num_envs=16
```

OMY Lift task

```bash
# Train
python scripts/reinforcement_learning/skrl/train.py --task RobotisLab-Lift-Cube-OMY-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/skrl/play.py --task RobotisLab-Lift-Cube-OMY-v0 --num_envs=16
```

OMY Open drawer task

```bash
# Train
python scripts/reinforcement_learning/skrl/train.py --task RobotisLab-Open-Drawer-OMY-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/skrl/play.py --task RobotisLab-Open-Drawer-OMY-v0 --num_envs=16
```

FFW-BG2 reach task

```bash
# Train
python scripts/reinforcement_learning/skrl/train.py --task RobotisLab-Reach-FFW-BG2-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/skrl/play.py --task RobotisLab-Reach-FFW-BG2-v0 --num_envs=16
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
