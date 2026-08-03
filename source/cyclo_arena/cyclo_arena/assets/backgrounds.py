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

"""Cyclo Lab and compatibility backgrounds for Cyclo Arena environments."""

import isaaclab.sim as sim_utils
from cyclo_lab.assets.environments.galileo_locomanip import (
    GALILEO_LOCOMANIP_ENVIRONMENT_POS,
    GALILEO_LOCOMANIP_ENVIRONMENT_ROT,
    GALILEO_LOCOMANIP_ENVIRONMENT_USD_PATH,
)
from cyclo_lab.assets.environments.robotis_showroom import (
    ROBOTIS_SHOWROOM_ENVIRONMENT_POS,
    ROBOTIS_SHOWROOM_ENVIRONMENT_ROT,
    ROBOTIS_SHOWROOM_USD_PATH,
    spawn_environment_with_friction,
)
from cyclo_lab.assets.environments.simple_warehouse import (
    ENVIRONMENT_POS as SIMPLE_WAREHOUSE_ENVIRONMENT_POS,
    ENVIRONMENT_ROT as SIMPLE_WAREHOUSE_ENVIRONMENT_ROT,
    ENVIRONMENT_SCALE as SIMPLE_WAREHOUSE_ENVIRONMENT_SCALE,
    SIMPLE_WAREHOUSE_ENVIRONMENT_USD_PATH,
)
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object_library import ISAACLAB_STAGING_NUCLEUS_DIR
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


_FRICTION_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg(
    contact_offset=0.003,
    rest_offset=0.0,
)


@register_asset
class CycloGalileoLocomanipBackground(Background):
    """Galileo warehouse using Cyclo Lab's friction-aware USD spawner."""

    name = "cyclo_galileo_locomanip"
    tags = ["background", "warehouse", "cyclo_lab"]

    def __init__(self):
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=GALILEO_LOCOMANIP_ENVIRONMENT_USD_PATH,
            initial_pose=Pose(
                position_xyz=GALILEO_LOCOMANIP_ENVIRONMENT_POS,
                rotation_wxyz=GALILEO_LOCOMANIP_ENVIRONMENT_ROT,
            ),
            object_min_z=-0.2,
            spawn_cfg_addon={
                "func": spawn_environment_with_friction,
                "collision_props": _FRICTION_COLLISION_PROPS,
            },
        )


@register_asset
class CycloRobotisShowroomBackground(Background):
    """ROBOTIS showroom using Cyclo Lab's local USD and friction setup."""

    name = "cyclo_robotis_showroom"
    tags = ["background", "showroom", "cyclo_lab"]

    def __init__(self):
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=ROBOTIS_SHOWROOM_USD_PATH,
            initial_pose=Pose(
                position_xyz=ROBOTIS_SHOWROOM_ENVIRONMENT_POS,
                rotation_wxyz=ROBOTIS_SHOWROOM_ENVIRONMENT_ROT,
            ),
            object_min_z=-0.2,
            spawn_cfg_addon={
                "func": spawn_environment_with_friction,
                "collision_props": _FRICTION_COLLISION_PROPS,
            },
        )


@register_asset
class CycloSimpleWarehouseBackground(Background):
    """Isaac Sim simple warehouse using Cyclo Lab's existing scene contract."""

    name = "cyclo_simple_warehouse"
    tags = ["background", "warehouse", "cyclo_lab"]

    def __init__(self):
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=SIMPLE_WAREHOUSE_ENVIRONMENT_USD_PATH,
            scale=(
                SIMPLE_WAREHOUSE_ENVIRONMENT_SCALE,
                SIMPLE_WAREHOUSE_ENVIRONMENT_SCALE,
                SIMPLE_WAREHOUSE_ENVIRONMENT_SCALE,
            ),
            initial_pose=Pose(
                position_xyz=SIMPLE_WAREHOUSE_ENVIRONMENT_POS,
                rotation_wxyz=SIMPLE_WAREHOUSE_ENVIRONMENT_ROT,
            ),
            object_min_z=-0.2,
            spawn_cfg_addon={
                "func": spawn_environment_with_friction,
                "collision_props": _FRICTION_COLLISION_PROPS,
            },
        )


@register_asset
class CycloOfficeTableBackground(Background):
    """Arena office table exposed as a background on the pinned compatibility branch."""

    name = "cyclo_office_table_background"
    tags = ["background", "table", "cyclo_arena_compat"]

    def __init__(self):
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/"
                "nut_pour_assets/table.usd"
            ),
            scale=(1.0, 1.0, 0.7),
            object_min_z=-0.05,
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True
                )
            },
        )


@register_asset
class CycloTableOakRobolabBackground(Background):
    """Robolab oak table exposed on the pinned Arena compatibility branch."""

    name = "cyclo_table_oak_robolab"
    tags = ["background", "table", "robolab", "cyclo_arena_compat"]

    def __init__(self):
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=(
                f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/"
                "srl_robolab_assets/fixtures/table_oak.usd"
            ),
            object_min_z=-0.05,
        )
