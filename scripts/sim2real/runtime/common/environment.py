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
# Author: Howon Kim

from __future__ import annotations

from collections import deque
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import bind_physics_material, clone, make_uninstanceable
from isaacsim.core.utils.stage import get_current_stage
from pxr import Usd, UsdGeom, UsdPhysics


SIMPLE_WAREHOUSE_ENVIRONMENT_USD_PATH = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
)
ROBOTIS_SHOWROOM_USD_PATH = str(
    Path(__file__).resolve().parents[4]
    / "source/cyclo_lab/data/environments/robotis_showroom/robotis_showroom_scene.usda"
)
# Backward-compatible name for older scripts.
FIXED_WORLD_USD_PATH = ROBOTIS_SHOWROOM_USD_PATH

ENVIRONMENT_SCALE = 0.7
ENVIRONMENT_POS = (1.25, 0.5, 0.0)
ENVIRONMENT_ROT = (1.0, 0.0, 0.0, 0.0)
FIXED_WORLD_ENVIRONMENT_POS = (0.0, 0.0, 0.0)
FIXED_WORLD_ENVIRONMENT_ROT = (1.0, 0.0, 0.0, 0.0)
ROBOTIS_SHOWROOM_ENVIRONMENT_POS = FIXED_WORLD_ENVIRONMENT_POS
ROBOTIS_SHOWROOM_ENVIRONMENT_ROT = FIXED_WORLD_ENVIRONMENT_ROT

GRASPABLE_CARD_BOX_MASS = 0.2
GRASPABLE_CARD_BOX_PRIM_PATHS = tuple(
    f"/World/envs/env_0/Environment/Shelf_1/SM_CardBoxD_{box_idx:02d}" for box_idx in range(1, 6)
)

ENVIRONMENT_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="max",
    restitution_combine_mode="min",
    static_friction=2.0,
    dynamic_friction=1.8,
    restitution=0.0,
)


def _make_showroom_floor_visual_only(prim_path: str) -> None:
    stage = get_current_stage()
    showroom_prim = stage.GetPrimAtPath(prim_path)
    if not showroom_prim.IsValid():
        return

    visual_only_paths = []
    for prim in Usd.PrimRange(showroom_prim):
        prim_path_text = str(prim.GetPath())
        if not prim_path_text.endswith("/ShowroomShell/Floor"):
            continue

        UsdGeom.Imageable(prim).MakeVisible()
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_api.CreateCollisionEnabledAttr(False).Set(False)
        visual_only_paths.append(prim_path_text)

    if visual_only_paths:
        print("[Robotis showroom] using visual-only showroom floor over Isaac ground plane contact.")


@clone
def spawn_environment_with_friction(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the environment USD and bind a high-friction material to its collision geometry."""
    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    material_path = f"{prim_path}/environmentPhysicsMaterial"
    ENVIRONMENT_PHYSICS_MATERIAL.func(material_path, ENVIRONMENT_PHYSICS_MATERIAL)
    bind_physics_material(prim_path, material_path)
    _make_showroom_floor_visual_only(prim_path)

    return prim


def environment_scale() -> tuple[float, float, float]:
    return (ENVIRONMENT_SCALE, ENVIRONMENT_SCALE, ENVIRONMENT_SCALE)


def make_simple_warehouse_environment_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_environment_with_friction,
            usd_path=SIMPLE_WAREHOUSE_ENVIRONMENT_USD_PATH,
            scale=environment_scale(),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=ENVIRONMENT_POS,
            rot=ENVIRONMENT_ROT,
        ),
    )


def make_robotis_showroom_environment_cfg(usd_path: str | None = None) -> AssetBaseCfg:
    environment_usd_path = usd_path or ROBOTIS_SHOWROOM_USD_PATH
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotisShowroom",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_environment_with_friction,
            usd_path=environment_usd_path,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=ROBOTIS_SHOWROOM_ENVIRONMENT_POS,
            rot=ROBOTIS_SHOWROOM_ENVIRONMENT_ROT,
        ),
    )


def make_fixed_world_environment_cfg(usd_path: str | None = None) -> AssetBaseCfg:
    return make_robotis_showroom_environment_cfg(usd_path)


def make_card_boxes_graspable():
    """Apply rigid body, mass, and collision properties to selected warehouse card boxes."""
    stage = get_current_stage()
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        kinematic_enabled=False,
        disable_gravity=False,
        max_depenetration_velocity=2.0,
    )
    mass_props = sim_utils.MassPropertiesCfg(mass=GRASPABLE_CARD_BOX_MASS)
    convex_hull_props = sim_utils.ConvexHullPropertiesCfg()

    for prim_path in GRASPABLE_CARD_BOX_PRIM_PATHS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[WARN] Graspable card box prim not found: {prim_path}")
            continue

        make_uninstanceable(prim_path, stage)
        sim_utils.define_rigid_body_properties(prim_path, rigid_props, stage)
        sim_utils.define_mass_properties(prim_path, mass_props, stage)

        queue = deque([prim])
        while queue:
            child_prim = queue.popleft()
            if child_prim.GetTypeName() == "Mesh":
                sim_utils.define_mesh_collision_properties(str(child_prim.GetPath()), convex_hull_props, stage)
            queue.extend(child_prim.GetChildren())

        print(f"[INFO] Graspable card box enabled: {prim_path} mass={GRASPABLE_CARD_BOX_MASS} kg")
