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

"""Compose a registered Cyclo robot, scene, and task into an Arena environment."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cyclo_arena.core.contracts import CompositionPlan, import_target


@dataclass(frozen=True)
class RobotRuntimeBuild:
    """Return an embodiment and its environment configuration callback."""

    embodiment: Any
    configure_env: Callable[[Any], Any]


class ArenaEnvironmentComposer:
    """Build one validated composition after Isaac Sim has started."""

    def __init__(self, plan: CompositionPlan, asset_registry: Any):
        self.plan = plan
        self.asset_registry = asset_registry

    def _make_background(self, args_cli: argparse.Namespace) -> Any:
        scene = self.plan.scene
        for module_name in scene.registration_modules:
            importlib.import_module(module_name)
        constructor_args = {
            constructor_name: getattr(args_cli, cli_name)
            for constructor_name, cli_name in scene.constructor_arg_names.items()
        }
        background = self.asset_registry.get_asset_by_name(scene.background_name)(
            **constructor_args
        )
        if scene.background_position_xyz is not None:
            from isaaclab_arena.utils.pose import Pose

            background.set_initial_pose(
                Pose(
                    position_xyz=scene.background_position_xyz,
                    rotation_wxyz=scene.background_rotation_wxyz,
                )
            )
        return background

    @staticmethod
    def _configure_common_env(env_cfg: Any) -> Any:
        env_cfg.sim.dt = 1.0 / 120.0
        env_cfg.sim.render_interval = 4
        env_cfg.decimation = 4
        return env_cfg

    def compose(self, args_cli: argparse.Namespace) -> Any:
        """Create the runtime Arena environment represented by this plan."""
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene

        background = self._make_background(args_cli)
        robot_adapter_class = import_target(self.plan.robot.runtime_adapter)
        robot_build = robot_adapter_class().build(
            args_cli=args_cli,
            asset_registry=self.asset_registry,
            placement=self.plan.placement,
        )
        task_class = import_target(self.plan.task.factory)
        task_constructor_args = {
            constructor_name: getattr(args_cli, cli_name)
            for constructor_name, cli_name in self.plan.task.constructor_arg_names.items()
        }
        task = task_class(**task_constructor_args)

        scene_assets = [background]
        if self.plan.scene.add_ground_plane:
            scene_assets.append(
                self.asset_registry.get_asset_by_name("ground_plane")()
            )
        scene_assets.append(self.asset_registry.get_asset_by_name("light")())

        def configure_env(env_cfg: Any) -> Any:
            env_cfg = self._configure_common_env(env_cfg)
            return robot_build.configure_env(env_cfg)

        return IsaacLabArenaEnvironment(
            name=self.plan.name,
            scene=Scene(assets=scene_assets),
            embodiment=robot_build.embodiment,
            task=task,
            env_cfg_callback=configure_env,
        )
