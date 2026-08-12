# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run a short K1 simulation and verify that Newton is the active backend."""

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test the K1 task with Newton.")
parser.add_argument("--task", default="Cyclo-Velocity-Flat-K1-Rev1-Play-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--action_value", type=float, default=None, help="Use one constant action value for all joints.")
parser.add_argument("--zero_gravity", action="store_true", help="Disable gravity to isolate actuator response.")
parser.add_argument("--fix_base", action="store_true", help="Fix the robot base to isolate joint-drive response.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# K1 is sourced from URDF, whose converter is provided by Isaac Sim Kit. Start
# Kit before importing the task configuration so its USD modules are initialized
# before the pip-installed USD bindings. Dynamics are still selected by NewtonCfg.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_newton.physics import NewtonCfg, NewtonManager
from isaaclab_tasks.utils import resolve_task_config

import cyclo_lab  # noqa: F401


def main():
    """Create K1, take a few random actions, and fail if Newton is not selected."""
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    if not isinstance(env_cfg.sim.physics, NewtonCfg):
        raise RuntimeError(f"Expected NewtonCfg, got {type(env_cfg.sim.physics).__name__}")

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.zero_gravity:
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 10.0)
        env_cfg.terminations.base_contact = None
    if args_cli.fix_base:
        env_cfg.scene.robot.spawn.fix_base = True
        env_cfg.terminations.base_contact = None
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    print(f"[INFO] Physics backend: {type(env_cfg.sim.physics).__name__}", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[INFO] K1 environment created.", flush=True)
    try:
        env.reset()
        print("[INFO] K1 environment reset completed.", flush=True)
        robot = env.unwrapped.scene["robot"]
        joint_target_mode = robot.root_view.get_attribute("joint_target_mode", NewtonManager.get_model())[:, 0]
        joint_target_mode_torch = wp.to_torch(joint_target_mode)
        mode_values, mode_counts = torch.unique(joint_target_mode_torch, return_counts=True)
        print(
            "[ACTUATOR] joint_target_mode="
            + ", ".join(
                f"{int(value.item())}:{int(count.item())}"
                for value, count in zip(mode_values, mode_counts, strict=True)
            ),
            flush=True,
        )
        initial_joint_pos = wp.to_torch(robot.data.joint_pos).clone()
        for _ in range(args_cli.steps):
            with torch.inference_mode():
                if args_cli.action_value is None:
                    actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
                else:
                    actions = torch.full(
                        env.action_space.shape, args_cli.action_value, device=env.unwrapped.device
                    )
                env.step(actions)
        joint_pos = wp.to_torch(robot.data.joint_pos)
        joint_target = wp.to_torch(robot.data.joint_pos_target)
        applied_torque = wp.to_torch(robot.data.applied_torque)
        print(
            "[ACTUATOR] "
            f"joint_delta(abs_mean/max)={(joint_pos - initial_joint_pos).abs().mean().item():.6f}/"
            f"{(joint_pos - initial_joint_pos).abs().max().item():.6f} "
            f"target_error(abs_mean/max)={(joint_target - joint_pos).abs().mean().item():.6f}/"
            f"{(joint_target - joint_pos).abs().max().item():.6f} "
            f"torque(abs_mean/max)={applied_torque.abs().mean().item():.6f}/"
            f"{applied_torque.abs().max().item():.6f}",
            flush=True,
        )
    finally:
        env.close()

    print(f"[PASS] K1 completed {args_cli.steps} simulation steps with Newton.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
