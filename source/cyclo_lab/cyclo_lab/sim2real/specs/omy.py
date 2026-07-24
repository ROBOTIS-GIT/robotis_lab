"""OMY sim2real topic and joint-order constants."""

OMY_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "rh_r1_joint",
)

OMY_JOINT_TRAJECTORY_TOPIC = "/leader/joint_trajectory"
OMY_JOINT_STATES_TOPIC = "/joint_states"
OMY_CAMERA_TOPICS = {
    "cam_top": "/camera/cam_top/color/image_rect_raw/compressed",
    "cam_wrist": "/camera/cam_wrist/color/image_rect_raw/compressed",
}
