#!/usr/bin/env python3
"""
run_task_reach.py
--------------------

ROS 2 node that drives a Kinova OMY arm with a simple
open-loop “reach” policy.

* Runs at      : 100 Hz

Author: Louis Le Lay
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial.transform import Rotation

from robots.omy import OMYReachPolicy


class ReachPolicy(Node):
    """ROS2 node for controlling a OMY robot's reach policy."""
    
    # Define simulation degree-of-freedom angle limits: (Lower limit, Upper limit, Inversed flag)
    SIM_DOF_ANGLE_LIMITS = [
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
    ]
    
    # Define servo angle limits (in radians)
    PI = math.pi
    SERVO_ANGLE_LIMITS = [
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
    ]
    
    # ROS topics and joint names
    STATE_TOPIC = '/arm_controller/controller_state'
    CMD_TOPIC = '/arm_controller/joint_trajectory'



    JOINT_NAMES = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
    ]
    
    # Mapping from joint name to simulation action index
    JOINT_NAME_TO_IDX = {
        'joint1': 0,
        'joint2': 1,
        'joint3': 2,
        'joint4': 3,
        'joint5': 4,
        'joint6': 5,
    }

    def __init__(self, fail_quietly: bool = False, verbose: bool = False):
        """Initialize the ReachPolicy node."""
        super().__init__('reach_policy_node')
        
        self.robot = OMYReachPolicy()
        self.target_command = np.zeros(6)
        self.step_size = 1.0 / 100 # 10 ms period = 100 Hz
        self.timer = self.create_timer(self.step_size, self.step_callback)
        self.i = 0
        self.fail_quietly = fail_quietly
        self.verbose = verbose
        # self.pub_freq = 1.0  # Hz
        self.current_pos = None  # Dictionary of current joint positions
        self.target_pos = None   # List of target joint positions

        # Subscriber for controller state messages
        self.create_subscription(
            JointTrajectoryControllerState,
            self.STATE_TOPIC,
            self.sub_callback,
            10
        )
        
        # Publisher for joint trajectory commands
        self.pub = self.create_publisher(JointTrajectory, self.CMD_TOPIC, 10)
        self.min_traj_dur = 1.0  # Minimum trajectory duration in seconds
        
        self.get_logger().info("ReachPolicy node initialized.")

    def sub_callback(self, msg: JointTrajectoryControllerState):
        """
        Callback for receiving controller state messages.
        Updates the current joint positions and passes the state to the robot model.
        """
        feedback_pos = {}
        for i, joint_name in enumerate(msg.joint_names):
            joint_pos = msg.feedback.positions[i]
            feedback_pos[joint_name] = joint_pos
        self.current_pos = feedback_pos
        
        # Update the robot's state with current joint positions and velocities.
        self.robot.update_joint_state(msg.feedback.positions, msg.feedback.velocities)

    def sample_random_pose(self) -> np.ndarray:
        # Position range
        pos_x = np.random.uniform(0.25, 0.55)
        pos_y = np.random.uniform(-0.2, 0.2)
        pos_z = np.random.uniform(0.3, 0.5)

        # Orientation range (in radians)
        roll  = np.random.uniform(math.pi / 2 - math.pi / 8, math.pi / 2 + math.pi / 8)
        pitch = np.random.uniform(-math.pi / 8, math.pi / 8)
        yaw   = np.random.uniform(math.pi / 2 - math.pi / 8, math.pi / 2 + math.pi / 8)

        # Convert to quaternion
        quat = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()  # [x, y, z, w]

        # Return full 7D pose: position + quaternion
        return np.array([pos_x, pos_y, pos_z, quat[0], quat[1], quat[2], quat[3]])


    def step_callback(self):
        """
        Timer callback to compute and publish the next joint trajectory command.
        """
        # Set a constant target command for the robot (example values)
        if self.i % 500 == 0:
            self.target_command = self.sample_random_pose()

        # Get simulation joint positions from the robot's forward model
        joint_pos = self.robot.forward(self.step_size, self.target_command)
        
        if joint_pos is not None:
            if len(joint_pos) != 6:
                raise Exception(f"Expected 6 joint positions, got {len(joint_pos)}!")
            
            traj = JointTrajectory()
            traj.joint_names = self.JOINT_NAMES

            point = JointTrajectoryPoint()
            point.positions = joint_pos
            point.time_from_start = Duration(sec=0, nanosec=50000000)  # Temps pour atteindre la position

            traj.points.append(point)
            
            self.pub.publish(traj)
            
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = ReachPolicy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()