# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Author: Taehyeong Kim

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="FFW SG2 Zenoh ROS2 runtime for Isaac Sim.")
    parser.add_argument("--disable_left_arm", action="store_true", help="Do not subscribe to the left arm topic.")
    parser.add_argument("--disable_right_arm", action="store_true", help="Do not subscribe to the right arm topic.")
    parser.add_argument("--disable_head", action="store_true", help="Do not subscribe to the head topic.")
    parser.add_argument("--disable_lift", action="store_true", help="Do not subscribe to the lift topic.")
    parser.add_argument("--disable_tf", action="store_true", help="Do not publish TF.")
    parser.add_argument("--disable_cmd_vel", action="store_true", help="Do not subscribe to cmd_vel for the swerve base.")
    parser.add_argument(
        "--publish_measured_lift_state",
        action="store_true",
        help="Publish measured lift position in joint_states instead of the held lift target.",
    )
    parser.add_argument("--enable_camera", action="store_true", help="Enable SG2 observation camera publishing.")
    parser.add_argument("--camera_publish_hz", type=float, default=15.0, help="Compressed image publish rate in Hz.")
    parser.add_argument("--enable_environment", action="store_true", help="Spawn the selected environment USD.")
    parser.add_argument(
        "--environment",
        choices=("robotis_showroom", "galileo_locomanip"),
        default="robotis_showroom",
        help="Environment preset to spawn. Defaults to robotis_showroom.",
    )
    parser.add_argument(
        "--environment_usd",
        default=None,
        help="Override the selected environment preset's USD path or URL.",
    )
    parser.add_argument("--base_frame", default="base_link", help="ROS base frame name for joint_states, odometry, and TF.")
    parser.add_argument("--base_body", default="world", help="SG2 USD body name to use for the base pose in Isaac Sim.")
    parser.add_argument("--max_runtime", type=float, default=0.0, help="Stop after this many seconds. 0 means run until closed.")
    parser.add_argument("--profile", action="store_true", help="Print timing statistics for the SG2 runtime loop.")
    parser.add_argument("--profile_interval", type=int, default=120, help="Runtime loop iterations between profile reports.")
    parser.add_argument(
        "--profile_cuda_sync",
        action="store_true",
        help="Synchronize CUDA around profiled sections for more accurate GPU timing. This adds overhead.",
    )
    parser.add_argument("--print_robot_info", action="store_true", help="Print SG2 joint and body names after the scene is ready.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args_cli = parse_args()
    if args_cli.enable_camera:
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from cyclo_lab.runtime.bridges.sg2.app import close_simulation_app, main as run_sg2_runtime

    try:
        run_sg2_runtime(args_cli, simulation_app)
    finally:
        close_simulation_app(simulation_app, skip_cleanup=args_cli.max_runtime > 0.0)


if __name__ == "__main__":
    main()
