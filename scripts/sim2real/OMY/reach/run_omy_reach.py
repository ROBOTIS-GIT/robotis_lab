#!/usr/bin/env python3

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

# Modified from original code by Louis Le Lay
# Original source: https://github.com/louislelay/kinova_isaaclab_sim2real

import argparse
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from utils.policy_executor import PolicyExecutor


class OMYReachPolicy(PolicyExecutor):
    def __init__(self, model_dir: str, debug: bool) -> None:
        super().__init__()
        self.joint_names = [f"joint{i}" for i in range(1, 7)]

        repo_root = Path(__file__).resolve().parents[4]
        model_dir = repo_root / "logs/rsl_rl/reach_omy" / model_dir

        self.load_policy(
            model_dir / "exported/policy.pt",
            model_dir / "params/env.yaml"
        )

        self.debug = debug
        self.has_joint_data = False
        self.previous_action = np.zeros(6)
        self.current_joint_positions = np.zeros(6)
        self.current_joint_velocities = np.zeros(6)


    def update_joint_state(self, position, velocity) -> None:
        self.current_joint_positions = np.array(position[:self.num_joints], dtype=np.float32)
        self.current_joint_velocities = np.array(velocity[:self.num_joints], dtype=np.float32)
        self.has_joint_data = True

    def compute_observation(self, command: np.ndarray) -> np.ndarray:
        if not self.has_joint_data:
            return None

        obs = np.zeros(19, dtype=np.float32)
        obs[:6] = self.current_joint_positions - self.default_pos
        obs[6:13] = command
        obs[13:19] = self.previous_action
        return obs

    def forward(self, command: np.ndarray) -> np.ndarray:
        if not self.has_joint_data:
            return None

        observation = self.compute_observation(command)
        if observation is None:
            return None

        self.action = self.compute_action(observation)
        self.previous_action = self.action.copy()
        joint_positions = self.default_pos + (self.action * self.action_scale)

        if self.debug:
            print("\n=== Policy Step ===")
            print(f"{'Command:':<20} {np.round(command, 4)}")
            print(f"{'Δ Joint Positions:':<20} {np.round(observation[:6], 4)}")
            print(f"{'Previous Action:':<20} {np.round(observation[13:19], 4)}")
            print(f"{'Raw Action:':<20} {np.round(self.action, 4)}")
            print(f"{'Processed Action:':<20} {np.round(joint_positions, 4)}")

        return joint_positions


class ReachPolicy(Node):
    """ROS2 node for executing reach policy on OMY robot."""

    def __init__(self, model_dir: str, debug: bool):
        super().__init__('omy_reach_policy_node')

        self.robot = OMYReachPolicy(model_dir, debug)
        self.target_command = np.zeros(6)
        self.step_size = 1.0 / 200  # 200Hz
        self.trajectory_time_from_start = 0.03

        self.iteration = 0
        self.br = TransformBroadcaster(self)

        self.create_subscription(
            JointTrajectoryControllerState,
            '/arm_controller/controller_state',
            self.joint_state_callback,
            10
        )
        self.joint_trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )
        self.timer = self.create_timer(self.step_size, self.step_callback)

        self.get_logger().info("ReachPolicy node initialized.")

    def joint_state_callback(self, msg: JointTrajectoryControllerState):
        self.robot.update_joint_state(msg.feedback.positions, msg.feedback.velocities)

    def sample_random_pose(self) -> np.ndarray:
        pos = np.random.uniform([0.25, -0.2, 0.3], [0.45, 0.2, 0.45])
        roll = np.random.uniform(0.0, math.pi)
        pitch = np.random.uniform(0.0, 0.0)
        yaw = np.random.uniform(0.0, math.pi)
        quat = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()  # [w, x, y, z]

        return np.concatenate([pos, quat])

    def create_trajectory_command(self, joint_positions: np.ndarray) -> JointTrajectory:
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = Duration(
            sec=0,
            nanosec=int(self.trajectory_time_from_start * 1e9)
        )

        joint_trajectory = JointTrajectory()
        joint_trajectory.joint_names = self.robot.joint_names
        joint_trajectory.points.append(point)
        return joint_trajectory

    def broadcast_target_pose_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = "target_pose"

        t.transform.translation.x = self.target_command[0]
        t.transform.translation.y = self.target_command[1]
        t.transform.translation.z = self.target_command[2]
        t.transform.rotation.x = self.target_command[4]
        t.transform.rotation.y = self.target_command[5]
        t.transform.rotation.z = self.target_command[6]
        t.transform.rotation.w = self.target_command[3]

        self.br.sendTransform(t)

    def step_callback(self):
        if self.iteration % 400 == 0:
            self.target_command = self.sample_random_pose()
            self.broadcast_target_pose_tf()
            self.get_logger().info(f"New target command: {np.round(self.target_command, 4)}")

        joint_positions = self.robot.forward(self.target_command)
        if joint_positions is not None:
            if len(joint_positions) != 6:
                raise ValueError(f"Expected 6 joint positions, got {len(joint_positions)}")
            joint_trajectory_msg = self.create_trajectory_command(joint_positions)
            self.joint_trajectory_publisher.publish(joint_trajectory_msg)

        self.iteration += 1


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Relative path to the trained policy directory under logs/rsl_rl/reach_omy/"
    )
    parser.add_argument(
        "--debug", type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="Enable debug print"
    )
    parsed_args, remaining_args = parser.parse_known_args(args)

    rclpy.init(args=remaining_args)
    node = ReachPolicy(model_dir=parsed_args.model_dir, debug=parsed_args.debug)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
