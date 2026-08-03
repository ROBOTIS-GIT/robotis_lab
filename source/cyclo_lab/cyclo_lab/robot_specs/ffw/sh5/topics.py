"""FFW SH5 ROS2-compatible topic, frame, and camera names."""

AI_WORKER_LEFT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"
AI_WORKER_RIGHT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"
HEAD_TOPIC = "/leader/joystick_controller_left/joint_trajectory"
LIFT_TOPIC = "/leader/joystick_controller_right/joint_trajectory"
CMD_VEL_TOPIC = "/cmd_vel"
JOINT_STATES_TOPIC = "/joint_states"
ODOM_TOPIC = "/odom"
TF_TOPIC = "/tf"
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"

FFW_SH5_RIGHT_HAND_TOPIC = "/leader/joint_trajectory_command_broadcaster_right_hand/joint_trajectory"
FFW_SH5_LEFT_HAND_TOPIC = "/leader/joint_trajectory_command_broadcaster_left_hand/joint_trajectory"

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
