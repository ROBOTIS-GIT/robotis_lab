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

"""FFW-SG2 embodiments for Cyclo Arena policy evaluation."""

from copy import deepcopy

import isaaclab.envs.mdp as mdp
import torch
from cyclo_lab.assets.robots import FFW_SG2_PHYSICS_CFG
from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    make_ffw_sg2_head_camera_cfg,
    make_ffw_sg2_wrist_camera_cfg,
)
from cyclo_lab.manager_based.actions import SwerveBaseVelocityActionCfg
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_HEAD_JOINT_NAMES,
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_LIFT_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    FFW_SG2_SWERVE_ANGULAR_ACCELERATION_LIMIT,
    FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
    FFW_SG2_SWERVE_ENABLED_SPEED_LIMITS,
    FFW_SG2_SWERVE_ENABLED_WHEEL_SATURATION_SCALING,
    FFW_SG2_SWERVE_LINEAR_ACCELERATION_LIMIT,
    FFW_SG2_SWERVE_STEERING_ALIGNMENT_ANGLE_ERROR_THRESHOLD,
    FFW_SG2_SWERVE_STEERING_ALIGNMENT_START_ANGLE_ERROR_THRESHOLD,
    FFW_SG2_SWERVE_STEERING_ALIGNMENT_START_SPEED_ERROR_THRESHOLD,
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
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose


def _make_robot_cfg() -> ArticulationCfg:
    robot_cfg = deepcopy(FFW_SG2_PHYSICS_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = False
    return robot_cfg


def _joint_position_action(joint_names: tuple[str, ...]) -> JointPositionActionCfg:
    return JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(joint_names),
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
    )


def _base_twist(env, asset_name: str) -> torch.Tensor:
    """Return FFW-SG2 body-frame ``[vx, vy, wz]`` state."""
    asset = env.scene[asset_name]
    return torch.cat(
        (asset.data.root_lin_vel_b[:, :2], asset.data.root_ang_vel_b[:, 2:3]),
        dim=-1,
    )


def _base_velocity_action() -> SwerveBaseVelocityActionCfg:
    """Create the shared FFW-SG2 three-axis swerve command term."""
    return SwerveBaseVelocityActionCfg(
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
        steering_angular_velocity_limit=(FFW_SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT),
        enabled_speed_limits=FFW_SG2_SWERVE_ENABLED_SPEED_LIMITS,
        linear_acceleration_limit=FFW_SG2_SWERVE_LINEAR_ACCELERATION_LIMIT,
        angular_acceleration_limit=FFW_SG2_SWERVE_ANGULAR_ACCELERATION_LIMIT,
        steering_alignment_angle_error_threshold=(FFW_SG2_SWERVE_STEERING_ALIGNMENT_ANGLE_ERROR_THRESHOLD),
        steering_alignment_start_angle_error_threshold=(FFW_SG2_SWERVE_STEERING_ALIGNMENT_START_ANGLE_ERROR_THRESHOLD),
        steering_alignment_start_speed_error_threshold=(FFW_SG2_SWERVE_STEERING_ALIGNMENT_START_SPEED_ERROR_THRESHOLD),
        enabled_wheel_saturation_scaling=(FFW_SG2_SWERVE_ENABLED_WHEEL_SATURATION_SCALING),
        drive_speed_scale=FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
    )


@configclass
class FFWSG2SceneCfg:
    """FFW-SG2 articulation and end-effector frames."""

    robot: ArticulationCfg = _make_robot_cfg()
    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_l_link7",
                name="left_end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.2)),
            )
        ],
    )
    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_r_link7",
                name="right_end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.2)),
            )
        ],
    )


@configclass
class FFWSG2CameraCfg:
    """Head and wrist RGB cameras."""

    cam_head: CameraCfg = make_ffw_sg2_head_camera_cfg(height=480, width=640)
    cam_wrist_left: CameraCfg = make_ffw_sg2_wrist_camera_cfg("left", height=640, width=480)
    cam_wrist_right: CameraCfg = make_ffw_sg2_wrist_camera_cfg("right", height=640, width=480)


@configclass
class FFWSG2ShowroomCameraCfg:
    """Three RGB streams with the showroom demonstration aspect ratios."""

    cam_head: CameraCfg = make_ffw_sg2_head_camera_cfg()
    cam_wrist_left: CameraCfg = make_ffw_sg2_wrist_camera_cfg("left")
    cam_wrist_right: CameraCfg = make_ffw_sg2_wrist_camera_cfg("right")


@configclass
class FFWSG2ObservationsCfg:
    """Policy observations in the real-robot joint order."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(FFW_SG2_ACTION_JOINT_NAMES),
                    preserve_order=True,
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(FFW_SG2_ACTION_JOINT_NAMES),
                    preserve_order=True,
                )
            },
        )
        base_twist = ObsTerm(func=_base_twist, params={"asset_name": "robot"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FFWSG2AbsoluteJointActionsCfg:
    """Absolute 19-value action surface used by Cyclo training and inference."""

    arm_l_action: ActionTermCfg = _joint_position_action(FFW_SG2_LEFT_ARM_JOINT_NAMES)
    gripper_l_action: ActionTermCfg = _joint_position_action(FFW_SG2_LEFT_GRIPPER_JOINT_NAMES)
    arm_r_action: ActionTermCfg = _joint_position_action(FFW_SG2_RIGHT_ARM_JOINT_NAMES)
    gripper_r_action: ActionTermCfg = _joint_position_action(FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES)
    lift_action: ActionTermCfg = _joint_position_action(FFW_SG2_LIFT_JOINT_NAMES)
    head_action: ActionTermCfg = _joint_position_action(FFW_SG2_HEAD_JOINT_NAMES)


@configclass
class FFWSG2MobileAbsoluteJointActionsCfg(FFWSG2AbsoluteJointActionsCfg):
    """Absolute joint positions followed by ``[vx, vy, wz]`` base velocity."""

    base_action: ActionTermCfg = _base_velocity_action()


class FFWSG2EmbodimentBase(EmbodimentBase):
    """Common FFW-SG2 scene, observations, and cameras."""

    name = "ffw_sg2"
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = FFWSG2SceneCfg()
        self.camera_config = FFWSG2CameraCfg()
        self.observation_config = FFWSG2ObservationsCfg()
        self.observation_config.policy.concatenate_terms = concatenate_observation_terms

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        assert arm_mode in (
            ArmMode.LEFT,
            ArmMode.RIGHT,
        ), "An individual arm is required for an end-effector frame"
        return "left_ee_frame" if arm_mode == ArmMode.LEFT else "right_ee_frame"

    def get_command_body_name(self) -> str:
        return "arm_r_link7"


@register_asset
class FFWSG2AbsoluteJointPositionEmbodiment(FFWSG2EmbodimentBase):
    """FFW-SG2 with absolute joint-position actions."""

    name = "ffw_sg2_abs_joint_pos"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_config = FFWSG2AbsoluteJointActionsCfg()


@register_asset
class FFWSG2MobileAbsoluteJointPositionEmbodiment(FFWSG2EmbodimentBase):
    """FFW-SG2 with 19 joint-position and 3 base-velocity actions."""

    name = "ffw_sg2_mobile_abs_joint_pos"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera_config = FFWSG2ShowroomCameraCfg()
        self.action_config = FFWSG2MobileAbsoluteJointActionsCfg()
