"""Observation helpers for SG2 showroom recording."""

from __future__ import annotations

import time

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv


def joint_pos_name(env: ManagerBasedEnv, joint_names: tuple[str, ...], asset_name: str = "robot") -> torch.Tensor:
    asset: Articulation = env.scene[asset_name]
    joint_ids = [asset.joint_names.index(name) for name in joint_names]
    return asset.data.joint_pos[:, joint_ids]


def joint_pos_target_name(
    env: ManagerBasedEnv,
    joint_names: tuple[str, ...],
    asset_name: str = "robot",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_name]
    joint_ids = [asset.joint_names.index(name) for name in joint_names]
    return asset.data.joint_pos_target[:, joint_ids]


def base_twist(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    asset: Articulation = env.scene[asset_name]
    return torch.cat([asset.data.root_lin_vel_b[:, 0:2], asset.data.root_ang_vel_b[:, 2:3]], dim=-1)


def wall_time(env: ManagerBasedEnv) -> torch.Tensor:
    """Return monotonic wall-clock time in seconds, relative to the first call."""
    now = time.perf_counter()
    if not hasattr(env, "_showroom_wall_time_start"):
        env._showroom_wall_time_start = now
    elapsed = now - env._showroom_wall_time_start
    return torch.full((env.num_envs, 1), elapsed, device=env.device, dtype=torch.float64)
