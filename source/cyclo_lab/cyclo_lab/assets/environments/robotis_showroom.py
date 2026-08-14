# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from pathlib import Path

from .physics import spawn_environment_with_friction_once


ROBOTIS_SHOWROOM_BASE_USD_PATH = str(
    Path(__file__).resolve().parents[3]
    / "data/environments/robotis_showroom/robotis_showroom.usd"
)
ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH = str(
    Path(__file__).resolve().parents[3]
    / "data/environments/robotis_showroom/robotis_showroom_background.usda"
)
ROBOTIS_SHOWROOM_BACKGROUND_TEXTURE_PATHS = tuple(
    str(
        Path(__file__).resolve().parents[3]
        / f"data/environments/robotis_showroom/textures/lab_background_{index:02d}.jpg"
    )
    for index in range(1, 4)
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
ROBOTIS_SHOWROOM_OBJECT_ROT_X_90 = (0.70710677, 0.70710677, 0.0, 0.0)

_SPAWN_ROBOTIS_SHOWROOM_ENVIRONMENT = None


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


def _spawn_robotis_showroom_environment_once(
    prim_path,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn one showroom instance and apply its floor-specific behavior."""
    prim = spawn_environment_with_friction_once(
        prim_path,
        cfg,
        translation,
        orientation,
        **kwargs,
    )
    _make_showroom_floor_visual_only(prim_path)
    return prim


def spawn_robotis_showroom_environment(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Clone and spawn the showroom with its visual-only authored floor."""
    global _SPAWN_ROBOTIS_SHOWROOM_ENVIRONMENT
    if _SPAWN_ROBOTIS_SHOWROOM_ENVIRONMENT is None:
        from isaaclab.sim.utils import clone

        _SPAWN_ROBOTIS_SHOWROOM_ENVIRONMENT = clone(_spawn_robotis_showroom_environment_once)
    return _SPAWN_ROBOTIS_SHOWROOM_ENVIRONMENT(prim_path, cfg, translation, orientation, **kwargs)


def make_robotis_showroom_environment_cfg(usd_path: str | None = None):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    environment_usd_path = usd_path or ROBOTIS_SHOWROOM_USD_PATH
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotisShowroom",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_robotis_showroom_environment,
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


def robotis_showroom_object_cfgs():
    """Return the supported dynamic showroom object assets keyed by placement type."""
    from cyclo_lab.assets.object import (
        JELLY_BAG_CFG,
        PEANUT_MIX_BAG_CFG,
        PLASTIC_BASKET_CFG,
        ROASTED_CHESTNUT_BAG_CFG,
    )

    return {
        "jelly_bag": JELLY_BAG_CFG,
        "peanut_mix_bag": PEANUT_MIX_BAG_CFG,
        "roasted_chestnut_bag": ROASTED_CHESTNUT_BAG_CFG,
        "plastic_basket": PLASTIC_BASKET_CFG,
    }


def _robotis_showroom_object_type(object_name: str, object_types) -> str | None:
    for object_type in object_types:
        if object_name == object_type or object_name.startswith(f"{object_type}_"):
            return object_type
    return None


def read_robotis_showroom_object_placements():
    """Read dynamic object transforms from the authored showroom object USD."""
    object_types = robotis_showroom_object_cfgs().keys()
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(ROBOTIS_SHOWROOM_OBJECTS_USD_PATH)
    if stage is None:
        raise RuntimeError(f"Failed to open {ROBOTIS_SHOWROOM_OBJECTS_USD_PATH}")

    object_parent = next(
        (prim for prim in stage.Traverse() if prim.GetName() == "robotis_showroom_objects"),
        None,
    )
    if object_parent is None:
        raise RuntimeError("Could not find robotis_showroom_objects prim")

    placements = []
    for prim in object_parent.GetChildren():
        object_name = prim.GetName()
        object_type = _robotis_showroom_object_type(object_name, object_types)
        if object_type is None or not prim.IsA(UsdGeom.Xformable):
            continue

        pos = None
        rot = None
        rotate_x_units = None
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            value = op.Get()
            if op.GetName() == "xformOp:translate":
                pos = tuple(float(value[index]) for index in range(3))
            elif op.GetName() == "xformOp:orient":
                imaginary = value.GetImaginary()
                rot = (float(value.GetReal()), *(float(imaginary[index]) for index in range(3)))
            elif op.GetName() == "xformOp:rotateX:unitsResolve":
                rotate_x_units = float(value)

        if pos is None:
            continue
        if rot is None and rotate_x_units is not None and abs(rotate_x_units - 90.0) < 1e-4:
            rot = ROBOTIS_SHOWROOM_OBJECT_ROT_X_90
        placements.append((object_name, object_type, pos, rot or ROBOTIS_SHOWROOM_OBJECT_ROT_X_90))

    if not placements:
        raise RuntimeError(f"No supported showroom object placements found in {ROBOTIS_SHOWROOM_OBJECTS_USD_PATH}")
    return tuple(placements)


def make_robotis_showroom_object_cfg(object_name, base_cfg, pos, rot):
    """Instantiate one registered rigid object at an authored showroom pose."""
    cfg = base_cfg.replace(prim_path=f"{{ENV_REGEX_NS}}/{object_name}")
    cfg.init_state.pos = list(pos)
    cfg.init_state.rot = list(rot)
    return cfg


def iter_robotis_showroom_object_cfgs():
    """Yield ``(name, cfg)`` pairs for all authored dynamic showroom objects."""
    object_cfgs = robotis_showroom_object_cfgs()
    for object_name, object_type, pos, rot in read_robotis_showroom_object_placements():
        yield object_name, make_robotis_showroom_object_cfg(
            object_name,
            object_cfgs[object_type],
            pos,
            rot,
        )
