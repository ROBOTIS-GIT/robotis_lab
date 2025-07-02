# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass

import robotis_lab.tasks.manager_based.OMY.reach.mdp as mdp
from robotis_lab.tasks.manager_based.OMY.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from robotis_lab.assets.OMY import OMY_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class OMYReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to OMY robot
        self.scene.robot = OMY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link6"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link6"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link6"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "link6"


@configclass
class OMYReachEnvCfg_PLAY(OMYReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
