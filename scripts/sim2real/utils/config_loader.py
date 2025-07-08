# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import fnmatch
import io
from typing import List, Tuple

import yaml

def parse_env_config(env_config_path: str = "env.yaml") -> dict:
    """
    Parses the environment configuration file from a local path or an Omniverse URL.

    Args:
        env_config_path (str, optional): The path to the environment configuration file. Can be local or an Omniverse URL.

    Returns:
        dict: The parsed environment configuration data.
    """

    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node) -> None:
            return None

        def tuple_constructor(loader, node) -> tuple:
            return tuple(loader.construct_sequence(node))

    SafeLoaderIgnoreUnknown.add_constructor("tag:yaml.org,2002:python/tuple", SafeLoaderIgnoreUnknown.tuple_constructor)
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(env_config_path, "rb") as f:
        file = io.BytesIO(f.read())

    data = yaml.load(file, Loader=SafeLoaderIgnoreUnknown)
    return data


def get_robot_joint_properties(
    data: dict, joint_names: List[str]
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Gets the robot joint properties from the environment configuration data.

    Args:
        data (dict): The environment configuration data.
        joint_names (List[str]): The list of joint names in the expected order.

    Returns:
        tuple: A tuple containing the effort limits, velocity limits, stiffness, damping, default positions, and default velocities.
    """
    actuator_data = data.get("scene").get("robot").get("actuators")
    stiffness = {}
    damping = {}
    effort_limits = {}
    velocity_limits = {}
    default_pos = {}
    default_vel = {}
    joint_names_expr_list = []

    for actuator in actuator_data:
        actuator_config = actuator_data.get(actuator)
        joint_names_expr = actuator_config.get("joint_names_expr")
        joint_names_expr_list.extend(joint_names_expr)

        effort_limit = actuator_config.get("effort_limit")
        velocity_limit = actuator_config.get("velocity_limit")
        joint_stiffness = actuator_config.get("stiffness")
        joint_damping = actuator_config.get("damping")

        if isinstance(effort_limit, (float, int)) or effort_limit is None:
            if effort_limit is None or effort_limit == float("inf"):
                effort_limit = float(sys.maxsize)
            for names in joint_names_expr:
                effort_limits[names] = float(effort_limit)
        elif isinstance(effort_limit, dict):
            effort_limits.update(effort_limit)
        else:
            print(f"Failed to parse effort limit, expected float, int, or dict, got: {type(effort_limit)}")

        if isinstance(velocity_limit, (float, int)) or velocity_limit is None:
            if velocity_limit is None or velocity_limit == float("inf"):
                velocity_limit = float(sys.maxsize)
            for names in joint_names_expr:
                velocity_limits[names] = float(velocity_limit)
        elif isinstance(velocity_limit, dict):
            velocity_limits.update(velocity_limit)
        else:
            print(f"Failed to parse velocity limit, expected float, int, or dict, got: {type(velocity_limit)}")

        if isinstance(joint_stiffness, (float, int)) or joint_stiffness is None:
            if joint_stiffness is None:
                joint_stiffness = 0
            for names in joint_names_expr:
                stiffness[names] = float(joint_stiffness)
        elif isinstance(joint_stiffness, dict):
            stiffness.update(joint_stiffness)
        else:
            print(f"Failed to parse stiffness, expected float, int, or dict, got: {type(joint_stiffness)}")

        if isinstance(joint_damping, (float, int)) or joint_damping is None:
            if joint_damping is None:
                joint_damping = 0
            for names in joint_names_expr:
                damping[names] = float(joint_damping)
        elif isinstance(joint_damping, dict):
            damping.update(joint_damping)
        else:
            print(f"Failed to parse damping, expected float, int, or dict, got: {type(joint_damping)}")

    # parse default joint position
    init_joint_pos = data.get("scene").get("robot").get("init_state").get("joint_pos")
    if isinstance(init_joint_pos, (float, int)):
        for names in joint_names_expr:
            default_pos[names] = float(init_joint_pos)
    elif isinstance(init_joint_pos, dict):
        default_pos.update(init_joint_pos)
    else:
        print(
            f"Failed to parse init state joint position, expected float, int, or dict, got: {type(init_joint_pos)}"
        )

    # parse default joint velocity
    init_joint_vel = data.get("scene").get("robot").get("init_state").get("joint_vel")
    if isinstance(init_joint_vel, (float, int)):
        for names in joint_names_expr:
            default_vel[names] = float(init_joint_vel)
    elif isinstance(init_joint_vel, dict):
        default_vel.update(init_joint_vel)
    else:
        print(
            f"Failed to parse init state vel position, expected float, int, or dict, got: {type(init_joint_vel)}"
        )

    stiffness_inorder = []
    damping_inorder = []
    effort_limits_inorder = []
    velocity_limits_inorder = []
    default_pos_inorder = []
    default_vel_inorder = []

    for joint in joint_names:
        for pattern in joint_names_expr_list:
            if fnmatch.fnmatch(joint, pattern.replace(".", "*") + "*"):
                if pattern in stiffness:
                    stiffness_inorder.append(stiffness[pattern])
                else:
                    stiffness_inorder.append(0)
                    print(f"{joint} stiffness not found, setting to 0")
                if pattern in damping:
                    damping_inorder.append(damping[pattern])
                else:
                    damping_inorder.append(0)
                    print(f"{joint} damping not found, setting to 0")
                if pattern in effort_limits:
                    effort_limits_inorder.append(effort_limits[pattern])
                else:
                    effort_limits_inorder.append(0)
                    print(f"{joint} effort limit not found, setting to 0")
                if pattern in velocity_limits:
                    velocity_limits_inorder.append(velocity_limits[pattern])
                else:
                    velocity_limits_inorder.append(0)
                    print(f"{joint} velocity limit not found, setting to 0")
                break

        default_position_found = False
        for pattern in default_pos:
            if fnmatch.fnmatch(joint, pattern.replace(".", "*") + "*"):
                default_pos_inorder.append(default_pos[pattern])
                default_position_found = True
                break
        if not default_position_found:
            default_pos_inorder.append(0)
            print(f"{joint} default position not found, setting to 0")

        default_velocity_found = False
        for pattern in default_vel:
            if fnmatch.fnmatch(joint, pattern.replace(".", "*") + "*"):
                default_vel_inorder.append(default_vel[pattern])
                default_velocity_found = True
                break
        if not default_velocity_found:
            default_vel_inorder.append(0)
            print(f"{joint} default velocity not found, setting to 0")

    return (
        effort_limits_inorder,
        velocity_limits_inorder,
        stiffness_inorder,
        damping_inorder,
        default_pos_inorder,
        default_vel_inorder,
    )

def get_physics_properties(data: dict) -> dict:
    """
    Gets the physics properties from the environment configuration data.

    Args:
        data (dict): The environment configuration data.

    Returns:
        tuple: A tuple containing the decimation, dt, and render interval.
    """
    return data.get("decimation"), data.get("sim").get("dt"), data.get("sim").get("render_interval")