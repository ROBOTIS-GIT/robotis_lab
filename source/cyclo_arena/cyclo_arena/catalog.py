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

"""Static Cyclo Arena catalog that is safe to inspect before Isaac Sim starts."""

from __future__ import annotations

import math

from cyclo_arena.core.capabilities import Capability
from cyclo_arena.core.contracts import (
    ModelAdapterSpec,
    PolicySpec,
    RobotPlacementSpec,
    RobotSpec,
    SceneSpec,
    TaskSpec,
)
from cyclo_arena.core.registry import CycloArenaRegistry

FFW_SG2_EMBODIMENTS = (
    "ffw_sg2_abs_joint_pos",
    "ffw_sg2_mobile_abs_joint_pos",
)
_SCENE_REGISTRATION_MODULES = ("cyclo_arena.assets.backgrounds",)


def _placement(position_xyz: tuple[float, float, float], yaw: float) -> dict[str, RobotPlacementSpec]:
    """Create the current FFW-SG2 placement map for one scene."""
    return {"ffw_sg2": RobotPlacementSpec(position_xyz=position_xyz, yaw=yaw)}


def _build_registry() -> CycloArenaRegistry:
    """Build the process-wide static catalog in dependency order."""
    registry = CycloArenaRegistry()
    registry.register_robot(
        RobotSpec(
            name="ffw_sg2",
            description="ROBOTIS FFW-SG2 dual-arm mobile manipulator.",
            embodiments=FFW_SG2_EMBODIMENTS,
            default_embodiment="ffw_sg2_abs_joint_pos",
            runtime_adapter="cyclo_arena.robots.ffw_sg2.runtime:FFWSG2RuntimeAdapter",
            capabilities=frozenset({
                Capability.ABSOLUTE_JOINT_POSITION,
                Capability.DUAL_ARM,
                Capability.HEAD_CAMERA,
                Capability.WRIST_CAMERAS,
                Capability.GR00T_REMOTE,
                Capability.SCENE_ONLY,
            }),
        )
    )

    scenes = (
        SceneSpec(
            name="galileo",
            description="Galileo locomotion-manipulation warehouse.",
            background_name="cyclo_galileo_locomanip",
            placements=_placement((-0.0955, -1.107, 0.0), -1.78),
            registration_modules=_SCENE_REGISTRATION_MODULES,
        ),
        SceneSpec(
            name="robotis_showroom",
            description="Cyclo Lab's local ROBOTIS showroom.",
            background_name="cyclo_robotis_showroom",
            placements=_placement((-1.316, 1.681, 0.0), math.pi),
            registration_modules=_SCENE_REGISTRATION_MODULES,
            add_ground_plane=True,
        ),
        SceneSpec(
            name="robotis_showroom_training",
            description="ROBOTIS showroom layout used for FFW-SG2 training.",
            background_name="cyclo_robotis_showroom_training",
            placements=_placement((-1.316, 1.681, 0.0), math.pi),
            registration_modules=_SCENE_REGISTRATION_MODULES,
            additional_assets_factory="cyclo_arena.assets.backgrounds:make_robotis_showroom_training_objects",
            add_ground_plane=True,
        ),
        SceneSpec(
            name="simple_warehouse",
            description="Isaac Sim simple multi-shelf warehouse.",
            background_name="cyclo_simple_warehouse",
            placements=_placement((0.0, 0.0, 0.0), 0.0),
            registration_modules=_SCENE_REGISTRATION_MODULES,
            add_ground_plane=True,
        ),
        SceneSpec(
            name="kitchen",
            description="Arena kitchen.",
            background_name="kitchen",
            placements=_placement((-0.8, 0.0, 0.0), 0.0),
            registration_modules=_SCENE_REGISTRATION_MODULES,
        ),
        SceneSpec(
            name="kitchen_with_open_drawer",
            description="Arena kitchen with an open drawer.",
            background_name="kitchen_with_open_drawer",
            placements=_placement((-0.8, 0.0, 0.0), 0.0),
            registration_modules=_SCENE_REGISTRATION_MODULES,
        ),
        SceneSpec(
            name="lightwheel_robocasa_kitchen",
            description="Lightwheel RoboCasa kitchen.",
            background_name="lightwheel_robocasa_kitchen",
            placements=_placement((3.943, -1.0, 0.0), math.pi / 2.0),
            registration_modules=_SCENE_REGISTRATION_MODULES,
            constructor_arg_names={
                "layout_id": "kitchen_layout",
                "style_id": "kitchen_style",
            },
        ),
        SceneSpec(
            name="packing_table",
            description="Arena packing-table workstation.",
            background_name="packing_table",
            placements=_placement((0.0, 0.0, 0.0), 0.0),
            registration_modules=_SCENE_REGISTRATION_MODULES,
        ),
        SceneSpec(
            name="table",
            description="Isaac Sim Seattle Lab table.",
            background_name="table",
            placements=_placement((-0.8, 0.0, 0.0), 0.0),
            registration_modules=_SCENE_REGISTRATION_MODULES,
            add_ground_plane=True,
            background_position_xyz=(0.8, 0.0, 0.0),
            background_rotation_wxyz=(
                math.sqrt(0.5),
                0.0,
                0.0,
                math.sqrt(0.5),
            ),
        ),
        SceneSpec(
            name="office_table_background",
            description="Arena office table.",
            background_name="cyclo_office_table_background",
            placements=_placement((-0.8, 0.0, 0.0), 0.0),
            registration_modules=_SCENE_REGISTRATION_MODULES,
            add_ground_plane=True,
            background_position_xyz=(0.8, 0.0, 0.0),
        ),
        SceneSpec(
            name="maple_table_robolab",
            description="Arena Robolab maple table.",
            background_name="maple_table_robolab",
            placements=_placement((-0.8, 0.0, 0.0), 0.0),
            registration_modules=_SCENE_REGISTRATION_MODULES,
            add_ground_plane=True,
        ),
        SceneSpec(
            name="table_oak_robolab",
            description="Arena Robolab oak table.",
            background_name="cyclo_table_oak_robolab",
            placements=_placement((-0.8, 0.0, 0.0), 0.0),
            registration_modules=_SCENE_REGISTRATION_MODULES,
            add_ground_plane=True,
            background_position_xyz=(0.8, 0.0, 0.0),
        ),
    )
    for scene in scenes:
        registry.register_scene(scene)

    registry.register_task(
        TaskSpec(
            name="scene_only",
            description="Inspect a scene without task success conditions.",
            factory="cyclo_arena.tasks.scene_only:SceneOnlyTask",
            required_capabilities=frozenset({Capability.SCENE_ONLY}),
            constructor_arg_names={"task_description": "task_description"},
        )
    )

    policies = (
        PolicySpec(
            name="zero_action",
            description="Arena zero-action smoke-test policy.",
        ),
        PolicySpec(
            name="replay",
            description="Arena HDF5 action replay policy.",
        ),
        PolicySpec(
            name="rsl_rl",
            description="Arena RSL-RL checkpoint policy.",
        ),
        PolicySpec(
            name="gr00t_closedloop",
            description="Arena-native GR00T N1.7 remote chunking policy.",
            runtime_target="isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy",
            required_capabilities=frozenset({Capability.HEAD_CAMERA, Capability.GR00T_REMOTE}),
        ),
    )
    for policy in policies:
        registry.register_policy(policy)

    registry.register_model_adapter(
        ModelAdapterSpec(
            name="ffw_sg2_gr00t_n17",
            description="GR00T N1.7 new_embodiment interface for FFW-SG2.",
            robot="ffw_sg2",
            policy="gr00t_closedloop",
            embodiment="ffw_sg2_abs_joint_pos",
            model_types=("Gr00tN1d7",),
            processor_embodiment="new_embodiment",
            modality_keys={
                "video": ("cam_left_head",),
                "state": ("arm_left", "arm_right"),
                "action": ("arm_left", "arm_right"),
                "language": ("annotation.human.task_description",),
            },
            action_horizon=40,
            server_robot_adapter="cyclo_arena.policies.adapters.ffw_sg2:FFWSG2Gr00tAdapter",
            server_image="cyclo-gr00t:n1.7",
            server_source_revision="23ace64f17aa5015259b8609d371eb61a357c776",
            server_workdir="/workspace",
            startup_timeout_seconds=600,
        )
    )
    registry.register_model_adapter(
        ModelAdapterSpec(
            name="ffw_sg2_gr00t_n17_showroom",
            description="GR00T N1.7 three-camera mobile interface for FFW-SG2 showroom.",
            robot="ffw_sg2",
            policy="gr00t_closedloop",
            embodiment="ffw_sg2_mobile_abs_joint_pos",
            model_types=("Gr00tN1d7",),
            processor_embodiment="new_embodiment",
            modality_keys={
                "video": (
                    "cam_left_head",
                    "cam_left_wrist",
                    "cam_right_wrist",
                ),
                "state": ("arm_left", "arm_right", "odometry"),
                "action": ("arm_left", "arm_right", "odometry"),
                "language": ("annotation.human.task_description",),
            },
            action_horizon=16,
            server_robot_adapter="cyclo_arena.policies.adapters.ffw_sg2:FFWSG2ShowroomGr00tAdapter",
            action_repeat=3,
            action_chunk_length=48,
            server_image="cyclo-gr00t:n1.7",
            server_source_revision="23ace64f17aa5015259b8609d371eb61a357c776",
            server_workdir="/workspace",
            startup_timeout_seconds=600,
        )
    )

    return registry


REGISTRY = _build_registry()
