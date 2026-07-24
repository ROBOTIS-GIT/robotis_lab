# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import threading
import time

import isaaclab.utils.math as math_utils

from cyclo_lab.sim2real.controllers.odometry import SwerveOdometry, yaw_to_quaternion
from cyclo_lab.sim2real.controllers.swerve_drive import SpeedLimiter, SwerveControllerConfig, SwerveDriveController, SwerveModule
from cyclo_lab.sim2real.runtime.camera_publishers import publish_compressed_camera
from cyclo_lab.sim2real.specs.ffw_sg2 import (
    FFW_SG2_LIFT_JOINT_NAME,
    FFW_SG2_LIFT_POSITION_LOWER as SG2_LIFT_POSITION_LOWER,
    FFW_SG2_LIFT_POSITION_UPPER as SG2_LIFT_POSITION_UPPER,
    FFW_SG2_PUBLISHED_JOINT_NAMES as SG2_PUBLISHED_JOINT_NAMES,
    FFW_SG2_SWERVE_ANGULAR_ACCELERATION_LIMIT as SG2_SWERVE_ANGULAR_ACCELERATION_LIMIT,
    FFW_SG2_SWERVE_DRIVE_SPEED_SCALE as SG2_SWERVE_DRIVE_SPEED_SCALE,
    FFW_SG2_SWERVE_LINEAR_ACCELERATION_LIMIT as SG2_SWERVE_LINEAR_ACCELERATION_LIMIT,
    FFW_SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT as SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
    clamp_lift_position,
)
from cyclo_lab.sim2real.transport.ros2_zenoh import (
    COMPRESSED_IMAGE,
    JOINT_STATE,
    JOINT_TRAJECTORY,
    ODOMETRY,
    TF_MESSAGE,
    TWIST,
    close_endpoints,
    create_publisher,
    create_subscriber,
    make_joint_state_kwargs,
    make_odometry_kwargs,
    make_tf_message_kwargs,
    now_time_msg,
    transform_stamped_msg,
)


def _now_stamp():
    return now_time_msg()


def _clamp_lift_position(position: float) -> float:
    return clamp_lift_position(position)


class SG2ZenohRos2Bridge:
    def __init__(
        self,
        robot,
        topic_names: dict[str, tuple[str, tuple[str, ...]]],
        joint_states_topic: str,
        odom_topic: str,
        tf_topic: str | None,
        base_frame: str,
        base_body: str,
        odom_frame: str,
        trajectory_qos,
        cmd_vel_topic: str | None,
        swerve_modules: list[SwerveModule],
        wheel_radius: float,
        cmd_vel_timeout: float,
        publish_lift_target_state: bool,
        cameras: dict[str, object] | None = None,
        camera_topics: dict[str, str] | None = None,
    ):
        self.robot = robot
        self.cameras = cameras or {}
        self.base_frame = base_frame
        self.odom_frame = odom_frame
        self.swerve_modules = swerve_modules
        self.cmd_vel_timeout = cmd_vel_timeout
        self.publish_lift_target_state = publish_lift_target_state
        self._warned_camera_publish_errors: set[str] = set()
        self.swerve_controller = (
            SwerveDriveController(
                swerve_modules,
                wheel_radius,
                config=SwerveControllerConfig(
                    enabled_speed_limits=True,
                    linear_x_limiter=SpeedLimiter(
                        has_acceleration_limits=True,
                        max_acceleration=SG2_SWERVE_LINEAR_ACCELERATION_LIMIT * SG2_SWERVE_DRIVE_SPEED_SCALE,
                    ),
                    linear_y_limiter=SpeedLimiter(
                        has_acceleration_limits=True,
                        max_acceleration=SG2_SWERVE_LINEAR_ACCELERATION_LIMIT * SG2_SWERVE_DRIVE_SPEED_SCALE,
                    ),
                    angular_z_limiter=SpeedLimiter(
                        has_acceleration_limits=True,
                        max_acceleration=SG2_SWERVE_ANGULAR_ACCELERATION_LIMIT * SG2_SWERVE_DRIVE_SPEED_SCALE,
                    ),
                    steering_angular_velocity_limit=SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
                ),
            )
            if swerve_modules
            else None
        )
        self.odometry = (
            SwerveOdometry(
                [module.x_offset for module in swerve_modules],
                [module.y_offset for module in swerve_modules],
                wheel_radius,
            )
            if swerve_modules
            else None
        )
        self._last_swerve_update_time = time.monotonic()
        self.running = True
        self.lock = threading.Lock()
        self.pending_positions: dict[str, float] = {}
        self.latest_cmd_vel = (0.0, 0.0, 0.0)
        self.last_cmd_vel_time = 0.0
        self.unknown_joints: set[str] = set()
        self._warned_unexpected_topic_joints: set[tuple[str, str]] = set()
        self._warned_lift_command_clamped = False
        self._joint_name_to_index = {
            name: index for index, name in enumerate(self.robot.data.joint_names)
        }
        self._lift_joint_id = self._joint_name_to_index.get(FFW_SG2_LIFT_JOINT_NAME)
        self._published_joint_state_names = [
            joint_name for joint_name in SG2_PUBLISHED_JOINT_NAMES if joint_name in self._joint_name_to_index
        ]
        self._published_joint_state_ids = [
            self._joint_name_to_index[joint_name] for joint_name in self._published_joint_state_names
        ]
        self._missing_published_joint_state_names = [
            joint_name for joint_name in SG2_PUBLISHED_JOINT_NAMES if joint_name not in self._joint_name_to_index
        ]
        if self._missing_published_joint_state_names:
            print(
                "[Zenoh ROS2] Missing configured joint_states joints: "
                f"{self._missing_published_joint_state_names}"
            )
        self._missing_swerve_joints = [
            joint_name
            for module in self.swerve_modules
            for joint_name in (module.steering_joint, module.wheel_joint)
            if joint_name not in self._joint_name_to_index
        ]
        self._swerve_steering_joint_ids = [
            self._joint_name_to_index[module.steering_joint]
            for module in self.swerve_modules
            if module.steering_joint in self._joint_name_to_index
        ]
        self._swerve_wheel_joint_ids = [
            self._joint_name_to_index[module.wheel_joint]
            for module in self.swerve_modules
            if module.wheel_joint in self._joint_name_to_index
        ]
        self._body_names = list(self.robot.data.body_names)
        self.base_body = self._resolve_base_body(base_body)
        self._base_id = self._body_names.index(self.base_body) if self.base_body in self._body_names else None
        self._warned_missing_base_frame = False
        self._warned_missing_swerve_joints: set[str] = set()

        self.subscribers = []
        self.publishers = []
        self.joint_state_writer = create_publisher(joint_states_topic, JOINT_STATE)
        self.publishers.append(self.joint_state_writer)
        self.odom_writer = None
        if self.odometry is not None:
            self.odom_writer = create_publisher(odom_topic, ODOMETRY)
            self.publishers.append(self.odom_writer)
        self.tf_writer = None
        if tf_topic:
            self.tf_writer = create_publisher(tf_topic, TF_MESSAGE)
            self.publishers.append(self.tf_writer)
        self.camera_writers = {}
        camera_topics = camera_topics or {}
        for camera_name, camera in self.cameras.items():
            topic = camera_topics.get(camera_name)
            if camera is None or not topic:
                continue
            writer = create_publisher(topic, COMPRESSED_IMAGE)
            self.camera_writers[camera_name] = writer
            self.publishers.append(writer)

        for label, (topic_name, fallback_joint_names) in topic_names.items():
            if not topic_name:
                continue
            subscriber = create_subscriber(
                topic=topic_name,
                msg_type=JOINT_TRAJECTORY,
                callback=lambda msg, label=label, fallback_joint_names=fallback_joint_names: self._store_trajectory(
                    label,
                    msg,
                    fallback_joint_names,
                ),
                qos=trajectory_qos,
            )
            self.subscribers.append(subscriber)
            print(f"[Zenoh ROS2] Subscribing {label}: {topic_name}")

        if cmd_vel_topic:
            cmd_vel_subscriber = create_subscriber(
                topic=cmd_vel_topic,
                msg_type=TWIST,
                callback=self._store_cmd_vel,
                qos=trajectory_qos,
            )
            self.subscribers.append(cmd_vel_subscriber)
            print(f"[Zenoh ROS2] Subscribing cmd_vel: {cmd_vel_topic}")

    def _resolve_base_body(self, requested_base_body: str) -> str:
        if requested_base_body in self._body_names:
            return requested_base_body

        for candidate in ("base_link", "world", "arm_base_link"):
            if candidate in self._body_names:
                print(
                    f"[Zenoh ROS2] Requested base body '{requested_base_body}' was not found; "
                    f"using '{candidate}' for base pose."
                )
                return candidate

        if self._body_names:
            fallback = self._body_names[0]
            print(
                f"[Zenoh ROS2] Requested base body '{requested_base_body}' was not found; "
                f"using first body '{fallback}' for base pose."
            )
            return fallback
        return requested_base_body

    # Parse trajectory topics and match joints
    def _store_trajectory(self, label: str, msg, fallback_joint_names: tuple[str, ...]):
        if msg is None or not msg.points:
            return

        point = msg.points[-1]
        positions = list(point.positions)
        joint_names = list(msg.joint_names)

        if label == "lift":
            lift_position = None
            if len(joint_names) == len(positions) and FFW_SG2_LIFT_JOINT_NAME in joint_names:
                lift_position = positions[joint_names.index(FFW_SG2_LIFT_JOINT_NAME)]
            elif not joint_names and len(positions) == 1:
                lift_position = positions[0]
            if lift_position is None:
                print(
                    f"[Zenoh ROS2] Ignoring lift message: '{FFW_SG2_LIFT_JOINT_NAME}' "
                    f"not found in joint_names={joint_names}"
                )
                return

            lift_position = float(lift_position)
            clamped_lift_position = _clamp_lift_position(lift_position)
            if clamped_lift_position != lift_position and not self._warned_lift_command_clamped:
                self._warned_lift_command_clamped = True
                print(
                    f"[Zenoh ROS2] Clamping lift_joint command to "
                    f"[{SG2_LIFT_POSITION_LOWER}, {SG2_LIFT_POSITION_UPPER}]."
                )

            with self.lock:
                self.pending_positions[FFW_SG2_LIFT_JOINT_NAME] = clamped_lift_position
            return

        if not joint_names and len(positions) <= len(fallback_joint_names):
            joint_names = list(fallback_joint_names[: len(positions)])
        if len(joint_names) != len(positions):
            print(
                f"[Zenoh ROS2] Ignoring {label} message: joint_names={len(joint_names)} "
                f"positions={len(positions)}"
            )
            return

        allowed_joint_names = set(fallback_joint_names)
        commands = {}
        for name, position in zip(joint_names, positions):
            if name in allowed_joint_names:
                commands[name] = float(position)
                continue

            warning_key = (label, name)
            if warning_key not in self._warned_unexpected_topic_joints:
                self._warned_unexpected_topic_joints.add(warning_key)
                print(f"[Zenoh ROS2] Ignoring {label} topic joint '{name}'; expected one of {fallback_joint_names}.")

        if not commands:
            return

        with self.lock:
            self.pending_positions.update(commands)

    # Apply swerve drive mobile base command
    def _store_cmd_vel(self, msg):
        if msg is None:
            return
        with self.lock:
            self.latest_cmd_vel = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
            self.last_cmd_vel_time = time.monotonic()

    def _current_cmd_vel(self) -> tuple[float, float, float]:
        with self.lock:
            command = self.latest_cmd_vel
            last_msg_time = self.last_cmd_vel_time

        if last_msg_time == 0.0:
            return 0.0, 0.0, 0.0
        if self.cmd_vel_timeout > 0.0 and time.monotonic() - last_msg_time > self.cmd_vel_timeout:
            return 0.0, 0.0, 0.0
        return tuple(value * SG2_SWERVE_DRIVE_SPEED_SCALE for value in command)

    def apply_latest_targets(self):
        with self.lock:
            commands = dict(self.pending_positions)

        position_target = self.robot.data.joint_pos_target.clone()
        velocity_target = self.robot.data.joint_vel_target.clone()

        for name, position in commands.items():
            joint_id = self._joint_name_to_index.get(name)
            if joint_id is None:
                if name not in self.unknown_joints:
                    self.unknown_joints.add(name)
                    print(f"[Zenoh ROS2] Joint '{name}' is not in the SG2 USD articulation; ignoring it.")
                continue
            if name == FFW_SG2_LIFT_JOINT_NAME:
                position = _clamp_lift_position(position)
            position_target[:, joint_id] = float(position)

        self._apply_swerve_targets(position_target, velocity_target)

        self.robot.set_joint_position_target(position_target)
        self.robot.set_joint_velocity_target(velocity_target)

    def _apply_swerve_targets(self, position_target, velocity_target):
        if not self.swerve_modules:
            return

        for joint_name in self._missing_swerve_joints:
            if joint_name not in self._warned_missing_swerve_joints:
                self._warned_missing_swerve_joints.add(joint_name)
                print(f"[Zenoh ROS2] Swerve joint '{joint_name}' is not in the SG2 USD articulation; ignoring cmd_vel.")
        if self._missing_swerve_joints:
            return

        current_steering = [
            float(value)
            for value in self.robot.data.joint_pos[0, self._swerve_steering_joint_ids].detach().cpu().tolist()
        ]
        current_wheel_velocities = [
            float(value)
            for value in self.robot.data.joint_vel[0, self._swerve_wheel_joint_ids].detach().cpu().tolist()
        ]
        linear_x, linear_y, angular_z = self._current_cmd_vel()
        now = time.monotonic()
        dt = now - self._last_swerve_update_time
        self._last_swerve_update_time = now

        if self.swerve_controller is None:
            return
        module_commands = self.swerve_controller.compute_commands(
            linear_x,
            linear_y,
            angular_z,
            current_steering_positions=current_steering,
            current_wheel_velocities=current_wheel_velocities,
            dt=dt,
        )
        for module_command, steering_id, wheel_id in zip(
            module_commands,
            self._swerve_steering_joint_ids,
            self._swerve_wheel_joint_ids,
        ):
            position_target[:, steering_id] = module_command.steering_position
            velocity_target[:, wheel_id] = module_command.wheel_velocity

    def update_odometry(self, dt: float):
        if self.odometry is None or not self.swerve_modules or self._missing_swerve_joints:
            return

        steering_positions = [
            float(value) + module.angle_offset
            for value, module in zip(
                self.robot.data.joint_pos[0, self._swerve_steering_joint_ids].detach().cpu().tolist(),
                self.swerve_modules,
            )
        ]
        wheel_velocities = [
            float(value)
            for value in self.robot.data.joint_vel[0, self._swerve_wheel_joint_ids].detach().cpu().tolist()
        ]
        self.odometry.update(steering_positions, wheel_velocities, dt)

    # Publish robot state and close ROS2 resources
    def publish_joint_states(self):
        all_positions = self.robot.data.joint_pos.squeeze(0).detach().cpu().tolist()
        all_velocities = self.robot.data.joint_vel.squeeze(0).detach().cpu().tolist()

        if self.publish_lift_target_state and self._lift_joint_id is not None:
            all_positions[self._lift_joint_id] = _clamp_lift_position(
                float(self.robot.data.joint_pos_target[0, self._lift_joint_id])
            )
            all_velocities[self._lift_joint_id] = float(self.robot.data.joint_vel_target[0, self._lift_joint_id])

        joint_names = self._published_joint_state_names
        positions = [all_positions[joint_id] for joint_id in self._published_joint_state_ids]
        velocities = [all_velocities[joint_id] for joint_id in self._published_joint_state_ids]
        efforts = [0.0] * len(joint_names)

        try:
            self.joint_state_writer.publish(
                **make_joint_state_kwargs(
                    names=joint_names,
                    positions=positions,
                    velocities=velocities,
                    efforts=efforts,
                    frame_id=self.base_frame,
                    stamp=_now_stamp(),
                )
            )
        except Exception as exc:
            print(f"[Zenoh ROS2] joint_states publish error: {exc}")

    def publish_odometry(self):
        if self.odom_writer is None or self.odometry is None:
            return

        state = self.odometry.state()
        quat_x, quat_y, quat_z, quat_w = yaw_to_quaternion(state.yaw)
        covariance = [0.0] * 36
        for index in (0, 7, 14, 21, 28, 35):
            covariance[index] = 0.001

        try:
            self.odom_writer.publish(
                **make_odometry_kwargs(
                    frame_id=self.odom_frame,
                    child_frame_id=self.base_frame,
                    position_xyz=(state.x, state.y, 0.0),
                    orientation_xyzw=(quat_x, quat_y, quat_z, quat_w),
                    linear_xyz=(state.vx, state.vy, 0.0),
                    angular_xyz=(0.0, 0.0, state.wz),
                    covariance=covariance,
                    stamp=_now_stamp(),
                )
            )
        except Exception as exc:
            print(f"[Zenoh ROS2] odom publish error: {exc}")

    def publish_tf(self):
        if self.tf_writer is None:
            return
        if self._base_id is None:
            if not self._warned_missing_base_frame:
                self._warned_missing_base_frame = True
                print(
                    f"[Zenoh ROS2] Cannot publish TF: base body '{self.base_body}' is not in SG2 body names. "
                    f"Available bodies: {self._body_names}"
                )
            return

        stamp = _now_stamp()
        body_pose_w = self.robot.data.body_link_state_w[0, :, :7]
        base_pose_w = body_pose_w[self._base_id]
        base_pos_w = base_pose_w[:3].unsqueeze(0)
        base_quat_w = base_pose_w[3:7].unsqueeze(0)

        transforms = []
        for body_id, child_frame in enumerate(self._body_names):
            if child_frame in (self.base_body, self.base_frame):
                continue

            child_pose_w = body_pose_w[body_id]
            child_pos_b, child_quat_b = math_utils.subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                child_pose_w[:3].unsqueeze(0),
                child_pose_w[3:7].unsqueeze(0),
            )
            pos = child_pos_b.squeeze(0).detach().cpu().tolist()
            quat_wxyz = child_quat_b.squeeze(0).detach().cpu().tolist()

            transforms.append(
                transform_stamped_msg(
                    parent_frame=self.base_frame,
                    child_frame=child_frame,
                    translation=(float(pos[0]), float(pos[1]), float(pos[2])),
                    rotation_xyzw=(
                        float(quat_wxyz[1]),
                        float(quat_wxyz[2]),
                        float(quat_wxyz[3]),
                        float(quat_wxyz[0]),
                    ),
                    stamp=stamp,
                )
            )

        try:
            self.tf_writer.publish(**make_tf_message_kwargs(transforms))
        except Exception as exc:
            print(f"[Zenoh ROS2] tf publish error: {exc}")

    def publish_cameras(self):
        for camera_name, camera in self.cameras.items():
            writer = self.camera_writers.get(camera_name)
            if camera is None or writer is None:
                continue
            self._publish_camera(camera_name, camera, writer)

    def _publish_camera(self, camera_name: str, camera, writer):
        try:
            publish_compressed_camera(camera_name, camera, writer, stamp_fn=_now_stamp)
        except Exception as exc:
            if camera_name not in self._warned_camera_publish_errors:
                self._warned_camera_publish_errors.add(camera_name)
                print(f"[Zenoh ROS2] {camera_name} publish error: {exc}")

    def shutdown(self):
        self.running = False
        close_endpoints(self.subscribers)
        close_endpoints(self.publishers)

    def reset_runtime_state(self):
        with self.lock:
            self.pending_positions.clear()
            self.latest_cmd_vel = (0.0, 0.0, 0.0)
            self.last_cmd_vel_time = 0.0
        self._last_swerve_update_time = time.monotonic()
        if self.swerve_controller is not None:
            self.swerve_controller.reset()
        if self.odometry is not None:
            self.odometry.reset()
