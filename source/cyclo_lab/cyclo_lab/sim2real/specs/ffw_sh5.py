"""FFW SH5 sim2real topic, frame, and runtime defaults."""


# AI Worker command topics.
AI_WORKER_RIGHT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"
AI_WORKER_LEFT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"
FFW_SH5_RIGHT_HAND_TOPIC = "/leader/joint_trajectory_command_broadcaster_right_hand/joint_trajectory"
FFW_SH5_LEFT_HAND_TOPIC = "/leader/joint_trajectory_command_broadcaster_left_hand/joint_trajectory"
SH5_RIGHT_HAND_TOPIC = FFW_SH5_RIGHT_HAND_TOPIC
SH5_LEFT_HAND_TOPIC = FFW_SH5_LEFT_HAND_TOPIC
HEAD_TOPIC = "/leader/joystick_controller_left/joint_trajectory"
LIFT_TOPIC = "/leader/joystick_controller_right/joint_trajectory"
CMD_VEL_TOPIC = "/cmd_vel"

# ROS2 output topics and frames.
JOINT_STATES_TOPIC = "/joint_states"
ODOM_TOPIC = "/odom"
TF_TOPIC = "/tf"
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"

# SH5 lift command conversion.
LIFT_JOINT_NAME = "lift_joint"
LIFT_POSITION_SCALE = 0.5

# Runtime timing and robot spawn pose.
PUBLISH_HZ = 30.0
STEP_HZ = 30.0
RENDER_INTERVAL = 1
ROBOT_POS = (0.0, 0.0, -0.18)
CMD_VEL_TIMEOUT = 0.1

# SH5 swerve runtime limits.
FFW_SH5_SWERVE_STEERING_LIMIT_LOWER = -1.570796
FFW_SH5_SWERVE_STEERING_LIMIT_UPPER = 1.570796
FFW_SH5_SWERVE_WHEEL_SPEED_LIMIT_LOWER = -50.0
FFW_SH5_SWERVE_WHEEL_SPEED_LIMIT_UPPER = 50.0
AI_WORKER_SWERVE_STEERING_LIMIT_LOWER = FFW_SH5_SWERVE_STEERING_LIMIT_LOWER
AI_WORKER_SWERVE_STEERING_LIMIT_UPPER = FFW_SH5_SWERVE_STEERING_LIMIT_UPPER
AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_LOWER = FFW_SH5_SWERVE_WHEEL_SPEED_LIMIT_LOWER
AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_UPPER = FFW_SH5_SWERVE_WHEEL_SPEED_LIMIT_UPPER

# Isaac Sim overview viewport placement.
ISAAC_SIM_OVERVIEW_CAMERA_EYE = (2.8, -2.2, 1.8)
ISAAC_SIM_OVERVIEW_CAMERA_TARGET = (0.0, 0.0, 0.8)

# AI Worker camera prim names used for optional SH5 viewport windows.
AI_WORKER_CAMERA_CENTER_NAME = "Head_Camera"
AI_WORKER_CAMERA_LEFT_NAME = "Left_Camera"
AI_WORKER_CAMERA_RIGHT_NAME = "Right_Camera"

FFW_SH5_ACTION_TOPICS = {
    "right_arm": AI_WORKER_RIGHT_ARM_TOPIC,
    "right_hand": FFW_SH5_RIGHT_HAND_TOPIC,
    "left_arm": AI_WORKER_LEFT_ARM_TOPIC,
    "left_hand": FFW_SH5_LEFT_HAND_TOPIC,
    "head": HEAD_TOPIC,
    "lift": LIFT_TOPIC,
    "mobile": CMD_VEL_TOPIC,
}
