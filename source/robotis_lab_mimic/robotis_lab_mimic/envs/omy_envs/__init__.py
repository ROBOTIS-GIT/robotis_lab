# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .omy_stack_ik_rel_mimic_env import OMYCubeStackIKRelMimicEnv
from .omy_stack_ik_rel_mimic_env_cfg import OMYCubeStackIKRelMimicEnvCfg

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-OMY-IK-Rel-Mimic-v0",
    entry_point="robotis_lab_mimic.envs.omy_envs:OMYCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": omy_stack_ik_rel_mimic_env_cfg.OMYCubeStackIKRelMimicEnvCfg,
    },
    disable_env_checker=True,
)