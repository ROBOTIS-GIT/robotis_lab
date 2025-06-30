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

import robotis_lab.tasks.manager_based.FFW_SG2.reach.mdp as mdp
from robotis_lab.tasks.manager_based.FFW_SG2.reach.reach_env_cfg import ReachEnvCfg
from robotis_lab.assets.FFW_SG2 import FFW_SG2_CFG  # isort: skip

@configclass
class FFWSG2ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Assign robot asset
        self.scene.robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Joint reset configuration
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)

        # Reward configuration: set correct end-effector body
        ee_link = "head_link2"
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [ee_link]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [ee_link]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [ee_link]

        # Action configuration (velocity: base, position: lift, head)
        self.actions.base_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=[
                "left_wheel_drive", "left_wheel_steer",
                "right_wheel_drive", "right_wheel_steer",
                "rear_wheel_drive", "rear_wheel_steer",
            ],
            scale=1.0,
        )
        self.actions.lift_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["lift_joint"],
            scale=0.5,
        )
        self.actions.head_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_joint[1-2]"],
            scale=0.5,
        )

        self.commands.ee_pose.body_name = ee_link


@configclass
class FFWSG2ReachEnvCfg_PLAY(FFWSG2ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
