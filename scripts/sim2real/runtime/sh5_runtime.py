# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Author: Howon Kim

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="FFW SH5 Zenoh ROS2 runtime for Isaac Sim.")
    parser.add_argument("--disable_head", action="store_true", help="Do not subscribe to the head topic.")
    parser.add_argument("--disable_lift", action="store_true", help="Do not subscribe to the lift topic.")
    parser.add_argument("--disable_cmd_vel", action="store_true", help="Do not subscribe to cmd_vel for the swerve base.")
    parser.add_argument("--enable_environment", action="store_true", help="Spawn the environment USD.")
    parser.add_argument(
        "--enable_camera_views",
        action="store_true",
        help="Open Isaac Sim viewport windows for overview, Head_Camera, Left_Camera, and Right_Camera.",
    )
    parser.add_argument("--max_runtime", type=float, default=0.0, help="Stop after this many seconds. 0 means run until closed.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args_cli = parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from cyclo_lab.sim2real.runtime.sh5_app import main as run_sh5_runtime

    try:
        run_sh5_runtime(args_cli, simulation_app)
    finally:
        close_simulation_app(simulation_app, skip_cleanup=args_cli.max_runtime > 0.0)


def close_simulation_app(simulation_app, *, skip_cleanup: bool = False):
    try:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=skip_cleanup)
    except TypeError:
        simulation_app.close()


if __name__ == "__main__":
    main()
