# Copyright 2025 ROBOTIS CO., LTD.
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
# Author: Taehyeong Kim

from copy import deepcopy

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR


SG2_SWERVE_STEERING_JOINTS = ("left_wheel_steer", "right_wheel_steer", "rear_wheel_steer")
SG2_SWERVE_WHEEL_JOINTS = ("left_wheel_drive", "right_wheel_drive", "rear_wheel_drive")
SG2_SWERVE_MODULE_X_OFFSETS = (0.1371, 0.1371, -0.2899)
SG2_SWERVE_MODULE_Y_OFFSETS = (0.2554, -0.2554, 0.0)
SG2_SWERVE_MODULE_ANGLE_OFFSETS = (0.0, 0.0, 0.0)
SG2_SWERVE_WHEEL_RADIUS = 0.0865
SG2_BRINGUP_LIFT_EFFORT_LIMIT = 5_000_000.0
SG2_BRINGUP_LIFT_STIFFNESS = 250_000.0
SG2_BRINGUP_LIFT_DAMPING = 5_000.0


FFW_SG2_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_SG2.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Left arm joints
            **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
            # Right arm joints
            **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},

            # Left and right gripper joints
            **{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
            **{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},

            # Head joints
            "head_joint1": 0.0,
            "head_joint2": 0.0,

            # Lift joint
            "lift_joint": 0.0,
        },
    ),
    actuators={
        # Actuator for vertical lift joint
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,
            effort_limit_sim=1_000_000.0,
            stiffness=10_000.0,
            damping=100.0,
        ),

        # Actuators for both arms
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[1-2]",
                "arm_r_joint[1-2]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=61.4,
            stiffness=600.0,
            damping=30.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[3-6]",
                "arm_r_joint[3-6]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=31.7,
            stiffness=600.0,
            damping=20.0,
        ),
        "DP-42": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint7",
                "arm_r_joint7",
            ],
            velocity_limit_sim=6.0,
            effort_limit_sim=5.1,
            stiffness=200.0,
            damping=3.0,
        ),

        # Actuators for grippers
        "gripper_master": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint1", "gripper_r_joint1"],
            velocity_limit_sim=2.2,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
        ),
        "gripper_slave": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
            effort_limit_sim=20.0,
            stiffness=2.0,
            damping=0.5,
        ),

        # Actuators for head joints
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
    }
)


def _tune_sg2_bringup_lift(robot_cfg: ArticulationCfg) -> None:
    lift_actuator = robot_cfg.actuators["lift"]
    lift_actuator.effort_limit_sim = SG2_BRINGUP_LIFT_EFFORT_LIMIT
    lift_actuator.stiffness = SG2_BRINGUP_LIFT_STIFFNESS
    lift_actuator.damping = SG2_BRINGUP_LIFT_DAMPING


def _enable_sg2_swerve_actuators(robot_cfg: ArticulationCfg) -> None:
    robot_cfg.init_state.joint_pos.update(
        {steering_joint: 0.0 for steering_joint in SG2_SWERVE_STEERING_JOINTS}
    )
    robot_cfg.init_state.joint_pos.update(
        {wheel_joint: 0.0 for wheel_joint in SG2_SWERVE_WHEEL_JOINTS}
    )
    robot_cfg.actuators = {
        "base_steer": ImplicitActuatorCfg(
            joint_names_expr=list(SG2_SWERVE_STEERING_JOINTS),
            velocity_limit_sim=10.0,
            effort_limit_sim=100000.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "base_drive": ImplicitActuatorCfg(
            joint_names_expr=list(SG2_SWERVE_WHEEL_JOINTS),
            velocity_limit_sim=50.0,
            effort_limit_sim=100000.0,
            stiffness=0.0,
            damping=100.0,
        ),
        **robot_cfg.actuators,
    }


# Keeps the SG2 USD fixed-root and lets bringup integrate /cmd_vel into root pose.
FFW_SG2_KINEMATIC_CFG = deepcopy(FFW_SG2_CFG)
_tune_sg2_bringup_lift(FFW_SG2_KINEMATIC_CFG)
_enable_sg2_swerve_actuators(FFW_SG2_KINEMATIC_CFG)
