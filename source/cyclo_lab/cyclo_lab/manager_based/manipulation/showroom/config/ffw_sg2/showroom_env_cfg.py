"""SG2 showroom environment configuration for HDF5 demonstration recording."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.robotis_showroom import (
    ROBOTIS_SHOWROOM_OBJECTS_USD_PATH,
)
from cyclo_lab.assets.object import (
    JELLY_BAG_CFG,
    PEANUT_MIX_BAG_CFG,
    PLASTIC_BASKET_CFG,
    ROASTED_CHESTNUT_BAG_CFG,
)
from cyclo_lab.manager_based.actions import SwerveBaseVelocityActionCfg
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_LIFT_JOINT_NAMES,
    FFW_SG2_PUBLISHED_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    FFW_SG2_SWERVE_ANGULAR_ACCELERATION_LIMIT,
    FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
    FFW_SG2_SWERVE_LINEAR_ACCELERATION_LIMIT,
    FFW_SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
    FFW_SG2_SWERVE_STEERING_LIMIT_LOWER,
    FFW_SG2_SWERVE_STEERING_LIMIT_UPPER,
    FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
    FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
    SG2_SWERVE_MODULE_ANGLE_OFFSETS,
    SG2_SWERVE_MODULE_X_OFFSETS,
    SG2_SWERVE_MODULE_Y_OFFSETS,
    SG2_SWERVE_STEERING_JOINTS,
    SG2_SWERVE_WHEEL_JOINTS,
    SG2_SWERVE_WHEEL_RADIUS,
)

from . import mdp


_OBJECT_ROT_X_90 = (0.70710677, 0.70710677, 0.0, 0.0)
SHOWROOM_CAMERA_NAMES = ("cam_head", "cam_wrist_left", "cam_wrist_right")

SHOWROOM_OBJECT_PLACEMENTS_FALLBACK = (
    ("jelly_bag_01", "jelly_bag", (-1.994056706, 1.386126380, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_02", "jelly_bag", (-1.994056706, 1.498434915, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_03", "jelly_bag", (-1.994056706, 1.614825097, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_04", "jelly_bag", (-1.994056706, 1.745269861, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_05", "jelly_bag", (-1.994056706, 1.866684014, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_06", "jelly_bag", (-1.994056706, 1.980108453, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_07", "jelly_bag", (-2.162688506, 1.386126380, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_08", "jelly_bag", (-2.162688506, 1.498434915, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_09", "jelly_bag", (-2.162688506, 1.614825097, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_10", "jelly_bag", (-2.162688506, 1.745269861, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_11", "jelly_bag", (-2.162688506, 1.866684014, 1.058593061), _OBJECT_ROT_X_90),
    ("jelly_bag_12", "jelly_bag", (-2.162688506, 1.980108453, 1.058593061), _OBJECT_ROT_X_90),
    ("peanut_mix_bag", "peanut_mix_bag", (-2.014346251, 1.762767120, 1.345386305), _OBJECT_ROT_X_90),
    ("peanut_mix_bag_01", "peanut_mix_bag", (-2.014346251, 1.879918935, 1.345386305), _OBJECT_ROT_X_90),
    ("peanut_mix_bag_02", "peanut_mix_bag", (-2.014346251, 1.991956556, 1.345386305), _OBJECT_ROT_X_90),
    ("peanut_mix_bag_03", "peanut_mix_bag", (-2.159416301, 1.879918935, 1.345386305), _OBJECT_ROT_X_90),
    ("peanut_mix_bag_04", "peanut_mix_bag", (-2.159416301, 1.762767120, 1.345386305), _OBJECT_ROT_X_90),
    ("peanut_mix_bag_05", "peanut_mix_bag", (-2.159416301, 1.991956556, 1.345386305), _OBJECT_ROT_X_90),
    ("roasted_chestnut_bag", "roasted_chestnut_bag", (-2.008060016, 1.612270777, 1.340930951), _OBJECT_ROT_X_90),
    ("roasted_chestnut_bag_01", "roasted_chestnut_bag", (-2.008060016, 1.497312938, 1.340930951), _OBJECT_ROT_X_90),
    ("roasted_chestnut_bag_02", "roasted_chestnut_bag", (-2.008060016, 1.395843257, 1.340930951), _OBJECT_ROT_X_90),
    ("roasted_chestnut_bag_03", "roasted_chestnut_bag", (-2.159459387, 1.395843257, 1.340930951), _OBJECT_ROT_X_90),
    ("roasted_chestnut_bag_04", "roasted_chestnut_bag", (-2.159459387, 1.612270777, 1.340930951), _OBJECT_ROT_X_90),
    ("roasted_chestnut_bag_05", "roasted_chestnut_bag", (-2.159459387, 1.497312938, 1.340930951), _OBJECT_ROT_X_90),
    ("plastic_basket", "plastic_basket", (-2.093242414, 1.904937999, 0.818692574), _OBJECT_ROT_X_90),
    ("plastic_basket_01", "plastic_basket", (-2.093242414, 1.658394973, 0.818692574), _OBJECT_ROT_X_90),
)

SHOWROOM_OBJECT_CFGS = {
    "jelly_bag": JELLY_BAG_CFG,
    "peanut_mix_bag": PEANUT_MIX_BAG_CFG,
    "roasted_chestnut_bag": ROASTED_CHESTNUT_BAG_CFG,
    "plastic_basket": PLASTIC_BASKET_CFG,
}


def _object_type_for_name(object_name: str) -> str | None:
    for object_type in SHOWROOM_OBJECT_CFGS:
        if object_name == object_type or object_name.startswith(f"{object_type}_"):
            return object_type
    return None


def read_showroom_object_placements():
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(ROBOTIS_SHOWROOM_OBJECTS_USD_PATH)
        if stage is None:
            raise RuntimeError(f"Failed to open {ROBOTIS_SHOWROOM_OBJECTS_USD_PATH}")

        object_parent = None
        for prim in stage.Traverse():
            if prim.GetName() == "robotis_showroom_objects":
                object_parent = prim
                break
        if object_parent is None:
            raise RuntimeError("Could not find robotis_showroom_objects prim")

        placements = []
        for prim in object_parent.GetChildren():
            object_name = prim.GetName()
            object_type = _object_type_for_name(object_name)
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
                rot = _OBJECT_ROT_X_90
            placements.append((object_name, object_type, pos, rot or _OBJECT_ROT_X_90))

        if not placements:
            raise RuntimeError("No showroom object placements found")
        return tuple(placements)
    except Exception as exc:
        print(f"[WARN] Failed to read showroom object placements from USD, using fallback: {exc}")
        return SHOWROOM_OBJECT_PLACEMENTS_FALLBACK


@configclass
class ShowroomSceneCfg(InteractiveSceneCfg):
    """Showroom scene with free SG2, static furniture, registered objects, and cameras."""

    robot: ArticulationCfg = MISSING
    environment: AssetBaseCfg = MISSING
    cam_head: CameraCfg = MISSING
    cam_wrist_left: CameraCfg = MISSING
    cam_wrist_right: CameraCfg = MISSING

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Showroom recording actions: 19 SG2 joints followed by 3D base velocity."""

    arm_l_action: mdp.JointPositionActionCfg = MISSING
    gripper_l_action: mdp.JointPositionActionCfg = MISSING
    arm_r_action: mdp.JointPositionActionCfg = MISSING
    gripper_r_action: mdp.JointPositionActionCfg = MISSING
    lift_action: mdp.JointPositionActionCfg = MISSING
    head_action: mdp.JointPositionActionCfg = MISSING
    base_action: SwerveBaseVelocityActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Policy observations written into the HDF5 ``obs`` group."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_name,
            params={"joint_names": FFW_SG2_PUBLISHED_JOINT_NAMES, "asset_name": "robot"},
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target_name,
            params={"joint_names": FFW_SG2_PUBLISHED_JOINT_NAMES, "asset_name": "robot"},
        )
        base_twist = ObsTerm(func=mdp.base_twist, params={"asset_name": "robot"})
        timestamp = ObsTerm(func=mdp.wall_time)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = None


@configclass
class CommandsCfg:
    pass


@configclass
class RewardsCfg:
    pass


@configclass
class EventsCfg:
    pass


@configclass
class CurriculumCfg:
    pass


@configclass
class ShowroomEnvCfg(ManagerBasedRLEnvCfg):
    """Base SG2 showroom env used by task-specific HDF5 collection configs."""

    env_name: str = "Cyclo-Real-Showroom-FFW-SG2-v0"
    scene: ShowroomSceneCfg = ShowroomSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    recorders: mdp.ShowroomRecorderManagerCfg = mdp.ShowroomRecorderManagerCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

    def set_camera_set(self, camera_set: str):
        """Enable a subset of showroom camera sensors and HDF5 camera recorder terms."""
        if camera_set == "all":
            camera_names = SHOWROOM_CAMERA_NAMES
        elif camera_set == "head":
            camera_names = ("cam_head",)
        elif camera_set == "none":
            camera_names = ()
        else:
            raise ValueError(f"Unsupported showroom camera set: {camera_set}")

        enabled = set(camera_names)
        if "cam_head" not in enabled:
            self.scene.cam_head = None
        if "cam_wrist_left" not in enabled:
            self.scene.cam_wrist_left = None
        if "cam_wrist_right" not in enabled:
            self.scene.cam_wrist_right = None

        if camera_names:
            self.recorders.record_pre_step_camera_observations.camera_names = tuple(camera_names)
        else:
            self.recorders.record_pre_step_camera_observations = None

    def set_camera_resolution(self, width: int, height: int):
        """Set all enabled showroom camera sensors to the same resolution."""
        for camera_name in SHOWROOM_CAMERA_NAMES:
            camera_cfg = getattr(self.scene, camera_name, None)
            if camera_cfg is None:
                continue
            camera_cfg.width = int(width)
            camera_cfg.height = int(height)

    def init_action_cfg(self, mode: str):
        if mode not in ("record", "inference"):
            raise ValueError(f"Unsupported SG2 showroom action mode: {mode}")

        self.actions.arm_l_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(FFW_SG2_LEFT_ARM_JOINT_NAMES),
            preserve_order=True,
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.gripper_l_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(FFW_SG2_LEFT_GRIPPER_JOINT_NAMES),
            preserve_order=True,
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.arm_r_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(FFW_SG2_RIGHT_ARM_JOINT_NAMES),
            preserve_order=True,
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.gripper_r_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES),
            preserve_order=True,
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.lift_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(FFW_SG2_LIFT_JOINT_NAMES),
            preserve_order=True,
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.head_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(FFW_SG2_ACTION_JOINT_NAMES[-2:]),
            preserve_order=True,
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.base_action = SwerveBaseVelocityActionCfg(
            asset_name="robot",
            steering_joint_names=tuple(SG2_SWERVE_STEERING_JOINTS),
            wheel_joint_names=tuple(SG2_SWERVE_WHEEL_JOINTS),
            module_x_offsets=tuple(SG2_SWERVE_MODULE_X_OFFSETS),
            module_y_offsets=tuple(SG2_SWERVE_MODULE_Y_OFFSETS),
            module_angle_offsets=tuple(SG2_SWERVE_MODULE_ANGLE_OFFSETS),
            wheel_radius=SG2_SWERVE_WHEEL_RADIUS,
            steering_limit_lower=FFW_SG2_SWERVE_STEERING_LIMIT_LOWER,
            steering_limit_upper=FFW_SG2_SWERVE_STEERING_LIMIT_UPPER,
            wheel_speed_limit_lower=FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
            wheel_speed_limit_upper=FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
            steering_angular_velocity_limit=FFW_SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
            linear_acceleration_limit=FFW_SG2_SWERVE_LINEAR_ACCELERATION_LIMIT,
            angular_acceleration_limit=FFW_SG2_SWERVE_ANGULAR_ACCELERATION_LIMIT,
            drive_speed_scale=FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
        )


def make_showroom_object_cfg(
    object_name: str,
    base_cfg: RigidObjectCfg,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
) -> RigidObjectCfg:
    cfg = base_cfg.replace(prim_path=f"{{ENV_REGEX_NS}}/{object_name}")
    cfg.init_state.pos = list(pos)
    cfg.init_state.rot = list(rot)
    return cfg
