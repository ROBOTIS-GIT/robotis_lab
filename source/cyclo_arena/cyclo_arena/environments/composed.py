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

"""Generic external environment for Cyclo robot, scene, and task compositions."""

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

from cyclo_arena.catalog import REGISTRY
from cyclo_arena.core.composer import ArenaEnvironmentComposer


class CycloArenaEnvironment(ExampleEnvironmentBase):
    """Compose the selected Cyclo robot, scene, and task at runtime."""

    name = "cyclo_composed"

    def get_env(self, args_cli: argparse.Namespace):
        plan = REGISTRY.compose(args_cli.robot, args_cli.scene, args_cli.cyclo_task)
        if args_cli.embodiment is None:
            args_cli.embodiment = plan.robot.default_embodiment
        assert args_cli.embodiment in plan.robot.embodiments, (
            f"Embodiment {args_cli.embodiment!r} is not valid for "
            f"robot {plan.robot.name!r}"
        )
        return ArenaEnvironmentComposer(
            plan=plan,
            asset_registry=self.asset_registry,
        ).compose(args_cli)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--robot", choices=tuple(REGISTRY.robots), required=True)
        parser.add_argument("--scene", choices=tuple(REGISTRY.scenes), required=True)
        parser.add_argument(
            "--cyclo_task",
            choices=tuple(REGISTRY.tasks),
            required=True,
        )
        parser.add_argument("--embodiment")
        parser.add_argument(
            "--task_description",
            default="Inspect the scene while holding the current robot pose.",
        )
        parser.add_argument(
            "--robot_position_xyz",
            type=float,
            nargs=3,
            metavar=("X", "Y", "Z"),
            help="Override the robot's registered scene position.",
        )
        parser.add_argument(
            "--robot_pose",
            help="Named initial joint pose from Cyclo Arena config.",
        )
        parser.add_argument(
            "--robot_yaw",
            type=float,
            help="Override the robot's registered scene yaw in radians.",
        )
        parser.add_argument(
            "--head_position",
            type=float,
            nargs=2,
            metavar=("HEAD_1", "HEAD_2"),
            help="Optional two-joint head position override.",
        )
        parser.add_argument(
            "--lift_position",
            type=float,
            help="Optional lift position override.",
        )
        parser.add_argument(
            "--kitchen_layout",
            type=int,
            default=1,
            help="Lightwheel RoboCasa kitchen layout ID.",
        )
        parser.add_argument(
            "--kitchen_style",
            type=int,
            default=1,
            help="Lightwheel RoboCasa kitchen style ID.",
        )
