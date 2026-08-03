# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from pathlib import Path


ROBOTIS_SHOWROOM_BASE_USD_PATH = str(
    Path(__file__).resolve().parents[3]
    / "data/environments/robotis_showroom/robotis_showroom.usd"
)
ROBOTIS_SHOWROOM_OBJECTS_USD_PATH = str(
    Path(__file__).resolve().parents[3]
    / "data/environments/robotis_showroom/robotis_showroom_objects.usd"
)
ROBOTIS_SHOWROOM_USD_PATH = str(
    Path(__file__).resolve().parents[3]
    / "data/environments/robotis_showroom/robotis_showroom_scene.usda"
)

ROBOTIS_SHOWROOM_ENVIRONMENT_POS = (0.0, 0.0, 0.0)
ROBOTIS_SHOWROOM_ENVIRONMENT_ROT = (1.0, 0.0, 0.0, 0.0)

_SPAWN_ENVIRONMENT_WITH_FRICTION = None


def _environment_physics_material():
    import isaaclab.sim as sim_utils

    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="min",
        static_friction=2.0,
        dynamic_friction=1.8,
        restitution=0.0,
    )


def _make_showroom_floor_visual_only(prim_path: str) -> None:
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom, UsdPhysics

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


def _spawn_environment_with_friction_impl(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn an environment USD and bind a high-friction material to collision geometry."""
    from isaaclab.sim.spawners.from_files import from_files
    from isaaclab.sim.utils import bind_physics_material

    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    material_path = f"{prim_path}/environmentPhysicsMaterial"
    physics_material = _environment_physics_material()
    physics_material.func(material_path, physics_material)
    bind_physics_material(prim_path, material_path)
    _make_showroom_floor_visual_only(prim_path)

    return prim


def spawn_environment_with_friction(prim_path, cfg, translation=None, orientation=None, **kwargs):
    global _SPAWN_ENVIRONMENT_WITH_FRICTION
    if _SPAWN_ENVIRONMENT_WITH_FRICTION is None:
        from isaaclab.sim.utils import clone

        _SPAWN_ENVIRONMENT_WITH_FRICTION = clone(_spawn_environment_with_friction_impl)
    return _SPAWN_ENVIRONMENT_WITH_FRICTION(prim_path, cfg, translation, orientation, **kwargs)


def make_robotis_showroom_environment_cfg(usd_path: str | None = None):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

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
