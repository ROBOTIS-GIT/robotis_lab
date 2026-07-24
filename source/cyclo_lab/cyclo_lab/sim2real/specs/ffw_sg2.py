"""FFW SG2 sim2real topic, joint-order, and runtime defaults."""

from __future__ import annotations

import math


# AI Worker command topics.
AI_WORKER_RIGHT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"
AI_WORKER_LEFT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"
HEAD_TOPIC = "/leader/joystick_controller_left/joint_trajectory"
LIFT_TOPIC = "/leader/joystick_controller_right/joint_trajectory"
CMD_VEL_TOPIC = "/cmd_vel"

# ROS2 output topics and frames.
JOINT_STATES_TOPIC = "/joint_states"
ODOM_TOPIC = "/odom"
TF_TOPIC = "/tf"
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"

# Runtime timing.
PUBLISH_HZ = 30.0
RENDER_INTERVAL = 1
CMD_VEL_TIMEOUT = 0.1

# SG2 joint groups.
FFW_SG2_LEFT_ARM_JOINT_NAMES = tuple(f"arm_l_joint{index}" for index in range(1, 8))
FFW_SG2_RIGHT_ARM_JOINT_NAMES = tuple(f"arm_r_joint{index}" for index in range(1, 8))
FFW_SG2_LEFT_GRIPPER_JOINT_NAMES = ("gripper_l_joint1",)
FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES = ("gripper_r_joint1",)
FFW_SG2_HEAD_JOINT_NAMES = ("head_joint1", "head_joint2")
FFW_SG2_LIFT_JOINT_NAME = "lift_joint"
FFW_SG2_LIFT_JOINT_NAMES = (FFW_SG2_LIFT_JOINT_NAME,)
LIFT_JOINT_NAME = FFW_SG2_LIFT_JOINT_NAME

# ROS joint_states order. This matches the real robot observation surface and
# keeps mimic gripper joints filtered out.
FFW_SG2_PUBLISHED_JOINT_NAMES = (
    *FFW_SG2_LEFT_ARM_JOINT_NAMES,
    *FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    *FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    *FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    *FFW_SG2_HEAD_JOINT_NAMES,
    *FFW_SG2_LIFT_JOINT_NAMES,
)
FFW_SG2_JOINT_NAMES = FFW_SG2_PUBLISHED_JOINT_NAMES

# Isaac Lab action tensor order for Cyclo-Real-Pick-Place-FFW-SG2-v0.
# The ActionCfg dataclass declares lift before head, so this must stay separate
# from the joint_states publication order above.
FFW_SG2_ACTION_JOINT_NAMES = (
    *FFW_SG2_LEFT_ARM_JOINT_NAMES,
    *FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    *FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    *FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    *FFW_SG2_LIFT_JOINT_NAMES,
    *FFW_SG2_HEAD_JOINT_NAMES,
)

FFW_SG2_ACTION_TOPICS = {
    "left_arm": AI_WORKER_LEFT_ARM_TOPIC,
    "right_arm": AI_WORKER_RIGHT_ARM_TOPIC,
    "head": HEAD_TOPIC,
    "lift": LIFT_TOPIC,
    "mobile": CMD_VEL_TOPIC,
}
FFW_SG2_JOYSTICK_TRIGGER_TOPIC = "/leader/joystick_controller/tact_trigger"
FFW_SG2_JOINT_STATES_TOPIC = JOINT_STATES_TOPIC
FFW_SG2_CAMERA_TOPICS = {
    "cam_head": "/zed/zed_node/left/image_rect_color/compressed",
    "cam_wrist_left": "/camera_left/camera_left/color/image_rect_raw/compressed",
    "cam_wrist_right": "/camera_right/camera_right/color/image_rect_raw/compressed",
}

# SG2 runtime defaults.
FFW_SG2_ROBOT_POS = (-1.316, 1.681, 0.0)
FFW_SG2_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
FFW_SG2_STEP_HZ = 30.0
FFW_SG2_ENVIRONMENT_GROUND_Z = 0.0
FFW_SG2_OVERVIEW_CAMERA_EYE = (2.2, -2.0, 1.6)
FFW_SG2_OVERVIEW_CAMERA_TARGET = (0.0, 0.0, 0.8)

# SG2 swerve runtime tuning.
FFW_SG2_SWERVE_STEERING_LIMIT_LOWER = -math.pi
FFW_SG2_SWERVE_STEERING_LIMIT_UPPER = math.pi
FFW_SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT = 4.0
FFW_SG2_SWERVE_DRIVE_SPEED_SCALE = 1.5
FFW_SG2_SWERVE_LINEAR_ACCELERATION_LIMIT = 0.6
FFW_SG2_SWERVE_ANGULAR_ACCELERATION_LIMIT = 1.2
FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_LOWER = -50.0
FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_UPPER = 50.0

# SG2 lift and initial manipulator pose.
FFW_SG2_LIFT_POSITION_LOWER = -0.5
FFW_SG2_LIFT_POSITION_UPPER = 0.0
FFW_SG2_LEFT_INITIAL_JOINT_POSITIONS = (0.0659, 0.3421, 0.5123, -2.4973, 0.612, 0.8882, -0.6281, 0.0)
FFW_SG2_RIGHT_INITIAL_JOINT_POSITIONS = (0.0659, -0.3421, -0.5123, -2.4973, -0.612, 0.8882, 0.6281, 0.0)
FFW_SG2_INITIAL_JOINT_POSITIONS = {
    **dict(zip((*FFW_SG2_LEFT_ARM_JOINT_NAMES, *FFW_SG2_LEFT_GRIPPER_JOINT_NAMES), FFW_SG2_LEFT_INITIAL_JOINT_POSITIONS)),
    **dict(zip((*FFW_SG2_RIGHT_ARM_JOINT_NAMES, *FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES), FFW_SG2_RIGHT_INITIAL_JOINT_POSITIONS)),
}


def clamp_lift_position(position: float) -> float:
    return max(FFW_SG2_LIFT_POSITION_LOWER, min(float(position), FFW_SG2_LIFT_POSITION_UPPER))
