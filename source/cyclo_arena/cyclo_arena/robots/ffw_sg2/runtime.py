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

"""Runtime construction of the FFW-SG2 Arena embodiment."""

from __future__ import annotations

import argparse
import math
from typing import Any

from cyclo_arena.core.composer import RobotRuntimeBuild
from cyclo_arena.core.contracts import RobotPlacementSpec


class FFWSG2RuntimeAdapter:
    """Apply FFW-SG2-specific embodiment, pose, and reset configuration."""

    def build(
        self,
        args_cli: argparse.Namespace,
        asset_registry: Any,
        placement: RobotPlacementSpec,
    ) -> RobotRuntimeBuild:
        """Build FFW-SG2 while reusing Cyclo Lab's articulation and sensors."""
        import cyclo_arena.embodiments.ffw_sg2  # noqa: F401
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        from isaaclab_arena.utils.pose import Pose

        from cyclo_arena.events import reset_articulation_joint_positions

        embodiment = asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras
        )
        if args_cli.robot_pose is not None:
            from cyclo_arena.core.robot_pose import load_robot_pose

            robot_pose = load_robot_pose("ffw_sg2", args_cli.robot_pose)
            known_joints = set(embodiment.scene_config.robot.init_state.joint_pos)
            unknown_joints = set(robot_pose.joint_positions) - known_joints
            assert not unknown_joints, (
                f"Pose {robot_pose.name!r} contains unknown FFW-SG2 joints: "
                f"{sorted(unknown_joints)}"
            )
            embodiment.scene_config.robot.init_state.joint_pos.update(
                robot_pose.joint_positions
            )
        if args_cli.head_position is not None:
            embodiment.scene_config.robot.init_state.joint_pos.update(
                {
                    "head_joint1": args_cli.head_position[0],
                    "head_joint2": args_cli.head_position[1],
                }
            )
        if args_cli.lift_position is not None:
            assert -0.5 <= args_cli.lift_position <= 0.0, (
                "lift_position must be within the FFW-SG2 range [-0.5, 0.0]"
            )
            embodiment.scene_config.robot.init_state.joint_pos["lift_joint"] = (
                args_cli.lift_position
            )

        position_xyz = args_cli.robot_position_xyz or placement.position_xyz
        yaw = args_cli.robot_yaw if args_cli.robot_yaw is not None else placement.yaw
        half_yaw = yaw / 2.0
        embodiment.set_initial_pose(
            Pose(
                position_xyz=tuple(position_xyz),
                rotation_wxyz=(
                    math.cos(half_yaw),
                    0.0,
                    0.0,
                    math.sin(half_yaw),
                ),
            )
        )
        initial_joint_positions = dict(
            embodiment.scene_config.robot.init_state.joint_pos
        )

        def configure_env(env_cfg: Any) -> Any:
            env_cfg.events.reset_ffw_sg2_joint_positions = EventTerm(
                func=reset_articulation_joint_positions,
                mode="reset",
                params={
                    "joint_positions": initial_joint_positions,
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            )
            return env_cfg

        return RobotRuntimeBuild(
            embodiment=embodiment,
            configure_env=configure_env,
        )
