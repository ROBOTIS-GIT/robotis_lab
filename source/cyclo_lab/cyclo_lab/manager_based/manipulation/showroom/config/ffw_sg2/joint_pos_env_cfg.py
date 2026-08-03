"""Joint-position SG2 showroom recording task configuration."""

from __future__ import annotations

from copy import deepcopy

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.robotis_showroom import (
    ROBOTIS_SHOWROOM_BASE_USD_PATH,
    make_robotis_showroom_environment_cfg,
)
from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    make_ffw_sg2_head_camera_cfg,
    make_ffw_sg2_wrist_camera_cfg,
)
from cyclo_lab.assets.robots import FFW_SG2_PHYSICS_CFG
from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_SWERVE_DRIVE_SPEED_SCALE

from .mdp import ffw_sg2_showroom_events
from .showroom_env_cfg import (
    SHOWROOM_OBJECT_CFGS,
    ShowroomEnvCfg,
    make_showroom_object_cfg,
    read_showroom_object_placements,
)


SG2_SHOWROOM_ROBOT_POS = (-1.316, 1.681, 0.0)
SG2_SHOWROOM_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
SG2_SHOWROOM_INITIAL_JOINT_POSITIONS = {
    "arm_l_joint1": 0.0659,
    "arm_l_joint2": 0.3421,
    "arm_l_joint3": 0.5123,
    "arm_l_joint4": -2.4973,
    "arm_l_joint5": 0.612,
    "arm_l_joint6": 0.8882,
    "arm_l_joint7": -0.6281,
    "gripper_l_joint1": 0.0,
    "arm_r_joint1": 0.0659,
    "arm_r_joint2": -0.3421,
    "arm_r_joint3": -0.5123,
    "arm_r_joint4": -2.4973,
    "arm_r_joint5": -0.612,
    "arm_r_joint6": 0.8882,
    "arm_r_joint7": 0.6281,
    "gripper_r_joint1": 0.0,
}


def make_sg2_showroom_robot_cfg() -> ArticulationCfg:
    robot_cfg = deepcopy(FFW_SG2_PHYSICS_CFG)
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.init_state.pos = SG2_SHOWROOM_ROBOT_POS
    robot_cfg.init_state.rot = SG2_SHOWROOM_ROBOT_ROT
    robot_cfg.init_state.joint_pos.update(SG2_SHOWROOM_INITIAL_JOINT_POSITIONS)
    base_drive_actuator = robot_cfg.actuators.get("base_drive")
    if base_drive_actuator is not None:
        base_drive_actuator.velocity_limit_sim *= FFW_SG2_SWERVE_DRIVE_SPEED_SCALE
    return robot_cfg


@configclass
class EventCfg:
    """Reset events for the SG2 showroom joint-position task."""

    reset_scene_to_default = EventTerm(
        func=ffw_sg2_showroom_events.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    set_robot_joint_pose = EventTerm(
        func=ffw_sg2_showroom_events.set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": SG2_SHOWROOM_INITIAL_JOINT_POSITIONS,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class FFWSG2ShowroomEnvCfg(ShowroomEnvCfg):
    """SG2 showroom env used by ``record_demos.py`` for HDF5 collection."""

    def __post_init__(self):
        super().__post_init__()
        self.events = EventCfg()

        self.scene.robot = make_sg2_showroom_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.environment = make_robotis_showroom_environment_cfg(ROBOTIS_SHOWROOM_BASE_USD_PATH)
        self.scene.cam_head = make_ffw_sg2_head_camera_cfg(update_period=0.0)
        self.scene.cam_wrist_left = make_ffw_sg2_wrist_camera_cfg("left", update_period=0.0)
        self.scene.cam_wrist_right = make_ffw_sg2_wrist_camera_cfg("right", update_period=0.0)

        for object_name, object_type, pos, rot in read_showroom_object_placements():
            object_cfg = SHOWROOM_OBJECT_CFGS[object_type]
            setattr(self.scene, object_name, make_showroom_object_cfg(object_name, object_cfg, pos, rot))
