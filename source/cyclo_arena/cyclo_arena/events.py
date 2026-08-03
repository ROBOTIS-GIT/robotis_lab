# Copyright 2026 ROBOTIS CO., LTD.
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
# Author: Seongwoo Kim

"""Reset events for Cyclo-owned Arena environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_articulation_joint_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_positions: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Write named joint positions to the articulation at reset."""
    asset: Articulation = env.scene[asset_cfg.name]
    positions = asset.data.default_joint_pos[env_ids].clone()
    velocities = torch.zeros_like(positions)

    for joint_name, position in joint_positions.items():
        assert joint_name in asset.joint_names, f"Unknown joint: {joint_name}"
        positions[:, asset.joint_names.index(joint_name)] = position

    asset.data.default_joint_pos[env_ids] = positions
    asset.set_joint_position_target(positions, env_ids=env_ids)
    asset.set_joint_velocity_target(velocities, env_ids=env_ids)
    asset.write_joint_state_to_sim(positions, velocities, env_ids=env_ids)
