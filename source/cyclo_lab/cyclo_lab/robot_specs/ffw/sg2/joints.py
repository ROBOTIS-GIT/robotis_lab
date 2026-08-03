"""FFW SG2 joint groups and public joint orders."""

FFW_SG2_LEFT_ARM_JOINT_NAMES = tuple(f"arm_l_joint{index}" for index in range(1, 8))
FFW_SG2_RIGHT_ARM_JOINT_NAMES = tuple(f"arm_r_joint{index}" for index in range(1, 8))
FFW_SG2_LEFT_GRIPPER_JOINT_NAMES = ("gripper_l_joint1",)
FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES = ("gripper_r_joint1",)
FFW_SG2_HEAD_JOINT_NAMES = ("head_joint1", "head_joint2")
FFW_SG2_LIFT_JOINT_NAME = "lift_joint"
FFW_SG2_LIFT_JOINT_NAMES = (FFW_SG2_LIFT_JOINT_NAME,)

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
