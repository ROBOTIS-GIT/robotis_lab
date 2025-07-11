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

import numpy as np
import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from reach_env_cfg import ReachEnvConfig
from utils.policy_executor import PolicyExecutor


class ReachPolicy(Node, PolicyExecutor):
    """ROS2 node for executing a reach policy on the OMY robot."""

    def __init__(self, model_dir: str, debug: bool):
        super().__init__('omy_reach_policy_node')

        self.cfg = ReachEnvConfig(model_dir=model_dir)
        self.load_policy(
            self.cfg.policy_path,
            self.cfg.env_yaml_path,
            self.cfg.joint_names
        )

        self.debug = debug
        self.br = TransformBroadcaster(self)
        self.iteration = 0
        self.has_joint_data = False

        self.target_command = np.zeros(7)  # [x, y, z, qw, qx, qy, qz]
        self.previous_action = np.zeros(self.num_joints)
        self.current_joint_positions = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)

        self.joint_state_subscriber = self.create_subscription(
            JointTrajectoryControllerState,
            self.cfg.joint_state_topic,
            self.joint_state_callback,
            10
        )
        self.joint_trajectory_publisher = self.create_publisher(
            JointTrajectory,
            self.cfg.joint_trajectory_topic,
            10
        )
        self.joint_command_timer = self.create_timer(self.cfg.step_size, self.timer_callback)

        self.get_logger().info("ReachPolicy node initialized.")

    def joint_state_callback(self, msg: JointTrajectoryControllerState):
        """Callback to update current joint state from the controller feedback."""
        self.update_joint_state(msg.feedback.positions, msg.feedback.velocities)
        self.has_joint_data = True

    def create_trajectory_command(self, joint_positions: np.ndarray) -> JointTrajectory:
        """Creates a JointTrajectory message from joint positions."""
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = Duration(
            sec=0,
            nanosec=int(self.cfg.trajectory_time_from_start * 1e9)
        )

        joint_trajectory = JointTrajectory()
        joint_trajectory.joint_names = self.cfg.joint_names
        joint_trajectory.points.append(point)
        return joint_trajectory

    def broadcast_target_pose_tf(self):
        """Publishes a TF transform for the target pose."""
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

    def timer_callback(self):
        """Main control loop: samples target, computes action, and publishes joint commands."""
        if not self.has_joint_data:
            return

        command_interval = int(self.cfg.send_command_interval / self.cfg.step_size) # interval in steps
        phase = self.iteration % (2 * command_interval)

        if phase == 0:
            self.target_command = self.cfg.sample_random_pose()
            self.broadcast_target_pose_tf()
            self.get_logger().info(f"New target command: {np.round(self.target_command, 4)}")

        if phase < command_interval:
            joint_trajectory_msg = self.create_trajectory_command(self.default_pos)
            self.joint_trajectory_publisher.publish(joint_trajectory_msg)
        else:
            joint_positions = self.forward(self.target_command)
            if joint_positions is not None:
                if len(joint_positions) != self.num_joints:
                    raise ValueError(f"Expected {self.num_joints} joint positions, got {len(joint_positions)}")
                joint_trajectory_msg = self.create_trajectory_command(joint_positions)
                self.joint_trajectory_publisher.publish(joint_trajectory_msg)

        self.iteration += 1

    def update_joint_state(self, position, velocity) -> None:
        """Updates internal joint state variables."""
        self.current_joint_positions = np.array(position[:self.num_joints], dtype=np.float32)
        self.current_joint_velocities = np.array(velocity[:self.num_joints], dtype=np.float32)

    def compute_observation(self, command: np.ndarray) -> np.ndarray:
        """Builds the observation vector for the policy."""
        obs = np.concatenate([
            self.current_joint_positions - self.default_pos,
            self.current_joint_velocities,
            command,
            self.previous_action,
        ]).astype(np.float32)

        return obs

    def forward(self, command: np.ndarray) -> np.ndarray:
        """Runs policy forward to compute joint positions for a given command."""
        observation = self.compute_observation(command)
        if observation is None:
            return None

        self.action = self.compute_action(observation)
        self.previous_action = self.action.copy()
        joint_positions = self.default_pos + (self.action * self.action_scale)

        if self.debug:
            self.get_logger().info("\n=== Policy Step ===")
            self.get_logger().info(f"{'Command:':<20} {np.round(command, 4)}")
            self.get_logger().info(f"{'Δ Joint Positions:':<20} {np.round(observation[:6], 4)}")
            self.get_logger().info(f"{'Joint Velocities:':<20} {np.round(observation[6:12], 4)}")
            self.get_logger().info(f"{'Previous Action:':<20} {np.round(observation[19:25], 4)}")
            self.get_logger().info(f"{'Raw Action:':<20} {np.round(self.action, 4)}")
            self.get_logger().info(f"{'Processed Action:':<20} {np.round(joint_positions, 4)}")

        return joint_positions

def main(args=None):
    """Entry point to initialize ROS2 and run the reach policy node."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Relative path to the trained policy directory under logs/rsl_rl/reach_omy/"
    )
    parser.add_argument(
        "--debug", type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=False,
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
