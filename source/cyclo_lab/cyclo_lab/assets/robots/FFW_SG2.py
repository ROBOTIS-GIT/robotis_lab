# Copyright 2025 ROBOTIS CO., LTD.
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
# Author: Taehyeong Kim

import re
from copy import deepcopy

from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf, Usd, UsdPhysics

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import (
    ArticulationRootPropertiesCfg,
    RigidBodyMaterialCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import bind_physics_material, clone, make_uninstanceable

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR


_SG2_WHEEL_PHYSICS_MATERIAL = RigidBodyMaterialCfg(
    friction_combine_mode="max",
    restitution_combine_mode="min",
    static_friction=2.0,
    dynamic_friction=1.8,
    restitution=0.0,
)

_SG2_BASE_LINK_NAME = "world"
_SG2_WHEEL_LINKS = (
    "left_wheel_steer_link",
    "left_wheel_drive_link",
    "right_wheel_steer_link",
    "right_wheel_drive_link",
    "rear_wheel_steer_link",
    "rear_wheel_drive_link",
)
_SG2_WHEEL_DRIVE_LINKS = ("left_wheel_drive_link", "right_wheel_drive_link", "rear_wheel_drive_link")

SG2_SWERVE_STEERING_JOINTS = ("left_wheel_steer", "right_wheel_steer", "rear_wheel_steer")
SG2_SWERVE_WHEEL_JOINTS = ("left_wheel_drive", "right_wheel_drive", "rear_wheel_drive")
SG2_SWERVE_MODULE_X_OFFSETS = (0.1371, 0.1371, -0.2899)
SG2_SWERVE_MODULE_Y_OFFSETS = (0.2554, -0.2554, 0.0)
SG2_SWERVE_MODULE_ANGLE_OFFSETS = (0.0, 0.0, 0.0)
SG2_SWERVE_WHEEL_RADIUS = 0.0865
SG2_SWERVE_DRIVE_DAMPING = 40.0
SG2_BRINGUP_LIFT_EFFORT_LIMIT = 5_000_000.0
SG2_BRINGUP_LIFT_STIFFNESS = 250_000.0
SG2_BRINGUP_LIFT_DAMPING = 5_000.0


def _iter_robot_prims(stage, prim_path: str):
    robot_prim = stage.GetPrimAtPath(prim_path)
    if not robot_prim.IsValid():
        return ()
    return Usd.PrimRange(robot_prim)


def _add_filtered_collision_pairs(stage, source_paths: list[str], target_paths: list[str]) -> None:
    for source_path in source_paths:
        source_prim = stage.GetPrimAtPath(source_path)
        if not source_prim.IsValid():
            continue
        filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(source_prim)
        filtered_pairs_rel = filtered_pairs_api.CreateFilteredPairsRel()
        for target_path in target_paths:
            filtered_pairs_rel.AddTarget(Sdf.Path(target_path))


def _remove_sg2_world_fixed_joint(stage, prim_path: str) -> None:
    fixed_joint_path = Sdf.Path(f"{prim_path}/ffw_sg2_follower/FixedJoint")
    fixed_joint_prim = stage.GetPrimAtPath(fixed_joint_path)
    if not fixed_joint_prim.IsValid():
        return

    joint_enabled_attr = fixed_joint_prim.GetAttribute("physics:jointEnabled")
    if joint_enabled_attr.IsValid():
        joint_enabled_attr.Set(False)
    exclude_attr = fixed_joint_prim.GetAttribute("physics:excludeFromArticulation")
    if exclude_attr.IsValid():
        exclude_attr.Set(True)
    for rel_name in ("physics:body0", "physics:body1"):
        rel = fixed_joint_prim.GetRelationship(rel_name)
        if rel.IsValid():
            rel.ClearTargets(True)
    fixed_joint_prim.SetActive(False)
    print("[SG2 base physics] disabled world fixed joint.")


def _apply_sg2_world_articulation_root(stage, prim_path: str) -> None:
    follower_path = Sdf.Path(f"{prim_path}/ffw_sg2_follower")
    base_path = follower_path.AppendChild(_SG2_BASE_LINK_NAME)

    follower_prim = stage.GetPrimAtPath(follower_path)
    if follower_prim.IsValid() and follower_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        try:
            follower_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        except Exception:
            pass

    base_prim = stage.GetPrimAtPath(base_path)
    if base_prim.IsValid():
        UsdPhysics.ArticulationRootAPI.Apply(base_prim)


def _filter_sg2_base_wheel_collisions(stage, prim_path: str) -> None:
    base_collision_paths = []
    wheel_collision_paths = []
    wheel_pattern = "|".join(re.escape(link_name) for link_name in _SG2_WHEEL_LINKS)

    for child_prim in _iter_robot_prims(stage, prim_path):
        child_path = str(child_prim.GetPath())
        if "/collisions/" not in child_path:
            continue

        lower_path = child_path.lower()
        if re.search(rf"(^|/){re.escape(_SG2_BASE_LINK_NAME)}/collisions(/|_|$)", lower_path):
            base_collision_paths.append(child_path)
        elif re.search(rf"(^|/)({wheel_pattern})/collisions(/|_|$)", lower_path):
            wheel_collision_paths.append(child_path)

    if not base_collision_paths or not wheel_collision_paths:
        return

    _add_filtered_collision_pairs(stage, base_collision_paths, wheel_collision_paths)
    _add_filtered_collision_pairs(stage, wheel_collision_paths, base_collision_paths)
    print("[SG2 base physics] disabled base body collision with swerve wheel links.")


def _iter_sg2_wheel_drive_collision_prims(stage, prim_path: str):
    wheel_drive_pattern = "|".join(re.escape(link_name) for link_name in _SG2_WHEEL_DRIVE_LINKS)

    for child_prim in _iter_robot_prims(stage, prim_path):
        child_path = str(child_prim.GetPath())
        lower_path = child_path.lower()
        if "/collisions/" not in lower_path:
            continue
        if not re.search(rf"(^|/)({wheel_drive_pattern})/collisions(/|_|$)", lower_path):
            continue
        if not child_prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_enabled = UsdPhysics.CollisionAPI(child_prim).GetCollisionEnabledAttr().Get()
        if collision_enabled is False:
            continue
        yield child_prim


def _bind_sg2_wheel_physics_material(stage, prim_path: str, material_path: str) -> None:
    wheel_collision_paths = [
        str(collision_prim.GetPath())
        for collision_prim in _iter_sg2_wheel_drive_collision_prims(stage, prim_path)
    ]
    for collision_path in wheel_collision_paths:
        bind_physics_material(collision_path, material_path)
    if wheel_collision_paths:
        print(f"[SG2 base physics] bound wheel physics material to {len(wheel_collision_paths)} drive collisions.")


@clone
def spawn_sg2_with_base_physics(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn SG2 with a free mobile base and wheel-contact physics helpers."""
    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    stage = get_current_stage()
    make_uninstanceable(prim_path, stage)
    _remove_sg2_world_fixed_joint(stage, prim_path)
    _apply_sg2_world_articulation_root(stage, prim_path)

    material_path = f"{prim_path}/wheelPhysicsMaterial"
    _SG2_WHEEL_PHYSICS_MATERIAL.func(material_path, _SG2_WHEEL_PHYSICS_MATERIAL)
    _bind_sg2_wheel_physics_material(stage, prim_path, material_path)

    _filter_sg2_base_wheel_collisions(stage, prim_path)
    return prim


FFW_SG2_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_SG2.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Left arm joints
            **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
            # Right arm joints
            **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},

            # Left and right gripper joints
            **{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
            **{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},

            # Head joints
            "head_joint1": 0.0,
            "head_joint2": 0.0,

            # Lift joint
            "lift_joint": 0.0,
        },
    ),
    actuators={
        # Actuator for vertical lift joint
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,
            effort_limit_sim=1_000_000.0,
            stiffness=10_000.0,
            damping=100.0,
        ),

        # Actuators for both arms
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[1-2]",
                "arm_r_joint[1-2]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=61.4,
            stiffness=600.0,
            damping=30.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[3-6]",
                "arm_r_joint[3-6]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=31.7,
            stiffness=600.0,
            damping=20.0,
        ),
        "DP-42": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint7",
                "arm_r_joint7",
            ],
            velocity_limit_sim=6.0,
            effort_limit_sim=5.1,
            stiffness=200.0,
            damping=3.0,
        ),

        # Actuators for grippers
        "gripper_master": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint1", "gripper_r_joint1"],
            velocity_limit_sim=2.2,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
        ),
        "gripper_slave": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
            effort_limit_sim=20.0,
            stiffness=2.0,
            damping=0.5,
        ),

        # Actuators for head joints
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
    }
)


def _tune_sg2_bringup_lift(robot_cfg: ArticulationCfg) -> None:
    lift_actuator = robot_cfg.actuators["lift"]
    lift_actuator.effort_limit_sim = SG2_BRINGUP_LIFT_EFFORT_LIMIT
    lift_actuator.stiffness = SG2_BRINGUP_LIFT_STIFFNESS
    lift_actuator.damping = SG2_BRINGUP_LIFT_DAMPING


def _enable_sg2_swerve_actuators(robot_cfg: ArticulationCfg) -> None:
    robot_cfg.init_state.joint_pos.update(
        {steering_joint: 0.0 for steering_joint in SG2_SWERVE_STEERING_JOINTS}
    )
    robot_cfg.init_state.joint_pos.update(
        {wheel_joint: 0.0 for wheel_joint in SG2_SWERVE_WHEEL_JOINTS}
    )
    robot_cfg.actuators = {
        "base_steer": ImplicitActuatorCfg(
            joint_names_expr=list(SG2_SWERVE_STEERING_JOINTS),
            velocity_limit_sim=10.0,
            effort_limit_sim=100000.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "base_drive": ImplicitActuatorCfg(
            joint_names_expr=list(SG2_SWERVE_WHEEL_JOINTS),
            velocity_limit_sim=50.0,
            effort_limit_sim=100000.0,
            stiffness=0.0,
            damping=SG2_SWERVE_DRIVE_DAMPING,
        ),
        **robot_cfg.actuators,
    }


FFW_SG2_PHYSICS_CFG = deepcopy(FFW_SG2_CFG)
FFW_SG2_PHYSICS_CFG.spawn.func = spawn_sg2_with_base_physics
FFW_SG2_PHYSICS_CFG.spawn.rigid_props.linear_damping = 2.0
FFW_SG2_PHYSICS_CFG.spawn.rigid_props.angular_damping = 4.0
FFW_SG2_PHYSICS_CFG.articulation_root_prim_path = "/ffw_sg2_follower/world"
_tune_sg2_bringup_lift(FFW_SG2_PHYSICS_CFG)
_enable_sg2_swerve_actuators(FFW_SG2_PHYSICS_CFG)

# Fixed-root config kept for old kinematic integrations.
FFW_SG2_KINEMATIC_CFG = deepcopy(FFW_SG2_CFG)
_tune_sg2_bringup_lift(FFW_SG2_KINEMATIC_CFG)
_enable_sg2_swerve_actuators(FFW_SG2_KINEMATIC_CFG)
