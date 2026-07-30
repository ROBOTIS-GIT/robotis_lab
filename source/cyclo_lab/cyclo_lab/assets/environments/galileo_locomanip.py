# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Galileo locomotion-manipulation warehouse scene from IsaacLab-Arena."""

from __future__ import annotations

from .robotis_showroom import spawn_environment_with_friction


GALILEO_LOCOMANIP_ENVIRONMENT_USD_PATH = (
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/IsaacLab/Arena/assets/background_library/"
    "galileo_locomanip/galileo_locomanip.usd"
)
GALILEO_LOCOMANIP_ENVIRONMENT_POS = (4.420, 1.408, -0.795)
GALILEO_LOCOMANIP_ENVIRONMENT_ROT = (1.0, 0.0, 0.0, 0.0)


def make_galileo_locomanip_environment_cfg(usd_path: str | None = None):
    """Create the Galileo warehouse configuration used by IsaacLab-Arena."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    environment_usd_path = usd_path or GALILEO_LOCOMANIP_ENVIRONMENT_USD_PATH
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GalileoLocomanip",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_environment_with_friction,
            usd_path=environment_usd_path,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=GALILEO_LOCOMANIP_ENVIRONMENT_POS,
            rot=GALILEO_LOCOMANIP_ENVIRONMENT_ROT,
        ),
    )
