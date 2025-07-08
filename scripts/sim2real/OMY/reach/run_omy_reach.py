#!/usr/bin/env python3

# Copyright 2025 ROBOTIS CO., LTD.
# Licensed under the Apache License, Version 2.0

import math
import numpy as np
import rclpy
from rclpy.node import Node
from pathlib import Path
from scipy.spatial.transform import Rotation

from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from utils.policy_controller import PolicyExecutor


class OMYReachPolicy(PolicyExecutor):
    """Policy controller for OMY Reach using a pre-trained policy model."""

    def __init__(self) -> None:
        super().__init__()
        self.dof_names = [f"joint{i}" for i in range(1, 7)]

        repo_root = Path(__file__).resolve().parents[4]
        model_dir = repo_root / "logs/rsl_rl/reach_omy/2025-07-08_06-25-57"

        self.load_policy(
            model_dir / "exported/policy.pt",
            model_dir / "params/env.yaml",
        )

        self.debug = False

        self._action_scale = 0.5

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

        obs = np.zeros(25)
        obs[:6] = self.current_joint_positions - self.default_pos
        obs[6:12] = self.current_joint_velocities
        obs[12:19] = command
        obs[19:25] = self.previous_action
        return obs

    def forward(self, dt: float, command: np.ndarray) -> np.ndarray:
        if not self.has_joint_data:
            return None

        obs = self.compute_observation(command)
        if obs is None:
            return None

        self.action = self._compute_action(obs)
        self.previous_action = self.action.copy()
        joint_positions = self.default_pos + (self.action * self._action_scale)

        # Debug Logging
        if self.debug:  # Change to True to enable debug print
            print("\n=== Policy Step ===")
            print(f"{'Command:':<20} {np.round(command, 4)}")
            print(f"{'Δ Joint Positions:':<20} {np.round(obs[:6], 4)}")
            print(f"{'Joint Velocities:':<20} {np.round(obs[6:12], 4)}")
            print(f"{'Previous Action:':<20} {np.round(obs[19:25], 4)}")
            print(f"{'Raw Action:':<20} {np.round(self.action, 4)}")
            print(f"{'Processed Action:':<20} {np.round(joint_positions, 4)}")

        return joint_positions


class ReachPolicy(Node):
    """ROS2 node for executing reach policy on OMY robot."""

    def __init__(self):
        super().__init__('reach_policy_node')

        self.robot = OMYReachPolicy()
        self.target_command = np.zeros(6)
        self.step_size = 1.0 / 100  # 100Hz
        self.timer = self.create_timer(self.step_size, self.step_callback)
        self.iteration = 0

        self.create_subscription(
            JointTrajectoryControllerState,
            '/arm_controller/controller_state',
            self.sub_callback,
            10
        )

        self.pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10
        )
        self.min_traj_dur = 0.05  # in seconds

        self.joint_names = [f"joint{i}" for i in range(1, 7)]

        self.get_logger().info("ReachPolicy node initialized.")

    def sub_callback(self, msg: JointTrajectoryControllerState):
        self.robot.update_joint_state(msg.feedback.positions, msg.feedback.velocities)

    def sample_random_pose(self) -> np.ndarray:
        pos = np.random.uniform([0.25, -0.2, 0.3], [0.55, 0.2, 0.5])
        euler = np.random.uniform(
            [math.pi/2 - math.pi/8, -math.pi/8, math.pi/2 - math.pi/8],
            [math.pi/2 + math.pi/8, math.pi/8, math.pi/2 + math.pi/8]
        )
        quat = Rotation.from_euler("xyz", euler).as_quat()
        return np.concatenate([pos, quat])

    def create_trajectory_command(self, joint_pos: np.ndarray) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = joint_pos
        point.time_from_start = Duration(sec=0, nanosec=int(self.min_traj_dur * 1e9))
        traj.points.append(point)
        return traj

    def step_callback(self):
        if self.iteration % 500 == 0:
            self.target_command = self.sample_random_pose()

        joint_pos = self.robot.forward(self.step_size, self.target_command)

        if joint_pos is not None:
            if len(joint_pos) != 6:
                raise ValueError(f"Expected 6 joint positions, got {len(joint_pos)}")
            traj_msg = self.create_trajectory_command(joint_pos)
            self.pub.publish(traj_msg)

        self.iteration += 1


def main(args=None):
    rclpy.init(args=args)
    node = ReachPolicy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
