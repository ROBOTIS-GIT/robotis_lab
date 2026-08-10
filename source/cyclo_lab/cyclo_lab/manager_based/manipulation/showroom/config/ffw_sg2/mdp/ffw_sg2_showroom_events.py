"""Reset events for the continuous SG2 showroom environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def create_joint_position_mapping(joint_names: list[str], desired_values: dict[str, float]) -> torch.Tensor:
    """Create a joint-position tensor ordered by the articulation joint names."""
    return torch.tensor([desired_values.get(joint_name, 0.0) for joint_name in joint_names], dtype=torch.float32)


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_positions: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set SG2 joint state and position target on reset."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = create_joint_position_mapping(asset.joint_names, joint_positions).to(device=env.device)
    if joint_pos.dim() == 1:
        joint_pos = joint_pos.unsqueeze(0).repeat(len(env_ids), 1)
    joint_vel = torch.zeros_like(joint_pos)

    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
