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
# Author: Howon Kim

import argparse
import os
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import cv2
from isaaclab.app import AppLauncher


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import robotis_config as cfg


# CLI and app launch
parser = argparse.ArgumentParser(description="FFW SG2 DDS runtime for Isaac Sim.")
parser.add_argument("--disable_left_arm", action="store_true", help="Do not subscribe to the left arm topic.")
parser.add_argument("--disable_right_arm", action="store_true", help="Do not subscribe to the right arm topic.")
parser.add_argument("--disable_head", action="store_true", help="Do not subscribe to the head topic.")
parser.add_argument("--disable_lift", action="store_true", help="Do not subscribe to the lift topic.")
parser.add_argument("--disable_tf", action="store_true", help="Do not publish TF.")
parser.add_argument("--disable_cmd_vel", action="store_true", help="Do not subscribe to cmd_vel for the swerve base.")
parser.add_argument(
    "--publish_measured_lift_state",
    action="store_true",
    help="Publish measured lift position in joint_states instead of the held lift target.",
)
parser.add_argument("--enable_camera", action="store_true", help="Enable SG2 head observation camera DDS publishing.")
parser.add_argument("--enable_environment", action="store_true", help="Spawn the Robotis showroom USD.")
parser.add_argument("--environment_usd", default=None, help="Environment USD path. Defaults to the generated Robotis showroom.")
parser.add_argument("--base_frame", default="world", help="Robot body name to use as the TF root and odometry child frame.")
parser.add_argument("--max_runtime", type=float, default=0.0, help="Stop after this many seconds. 0 means run until closed.")
parser.add_argument("--print_robot_info", action="store_true", help="Print SG2 joint and body names after the scene is ready.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.enable_camera:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from cyclonedds.core import Policy, Qos
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from robotis_dds_python.idl.builtin_interfaces.msg import Time_
from robotis_dds_python.idl.geometry_msgs.msg import (
    Point_,
    Pose_,
    PoseWithCovariance_,
    Quaternion_,
    Transform_,
    TransformStamped_,
    Twist_,
    TwistWithCovariance_,
    Vector3_,
)
from robotis_dds_python.idl.nav_msgs.msg import Odometry_
from robotis_dds_python.idl.sensor_msgs.msg import CompressedImage_, JointState_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.tf2_msgs.msg import TFMessage_
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.tools.topic_manager import TopicManager

from cyclo_lab.assets.robots import (
    FFW_SG2_KINEMATIC_CFG,
    SG2_SWERVE_MODULE_ANGLE_OFFSETS,
    SG2_SWERVE_MODULE_X_OFFSETS,
    SG2_SWERVE_MODULE_Y_OFFSETS,
    SG2_SWERVE_STEERING_JOINTS,
    SG2_SWERVE_WHEEL_JOINTS,
    SG2_SWERVE_WHEEL_RADIUS,
)
from common.environment import ROBOTIS_SHOWROOM_USD_PATH, make_robotis_showroom_environment_cfg
from common.odometry import SwerveOdometry, yaw_to_quaternion
from common.swerve_drive import SwerveDriveController, SwerveModule


SG2_ROBOT_POS = (1.6, 2.5, 0.0)
SG2_OVERVIEW_CAMERA_EYE = (2.2, -2.0, 1.6)
SG2_OVERVIEW_CAMERA_TARGET = (0.0, 0.0, 0.8)
SG2_HEAD_CAMERA_NAME = "cam_head"
SG2_HEAD_CAMERA_TOPIC = "/zed/zed_node/left/image_rect_color/compressed"

SG2_LEFT_ARM_JOINT_NAMES = tuple(f"arm_l_joint{index}" for index in range(1, 8))
SG2_RIGHT_ARM_JOINT_NAMES = tuple(f"arm_r_joint{index}" for index in range(1, 8))
SG2_LEFT_GRIPPER_JOINT_NAMES = ("gripper_l_joint1",)
SG2_RIGHT_GRIPPER_JOINT_NAMES = ("gripper_r_joint1",)
SG2_HEAD_JOINT_NAMES = ("head_joint1", "head_joint2")
SG2_LIFT_JOINT_NAMES = ("lift_joint",)


# ========== Scene Setup ==========

@configclass
class SG2BringupSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    robot: ArticulationCfg = None
    environment: AssetBaseCfg = None
    cam_head: CameraCfg = None


def _make_robot_cfg() -> ArticulationCfg:
    robot_cfg = deepcopy(FFW_SG2_KINEMATIC_CFG)
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.init_state.pos = SG2_ROBOT_POS
    return robot_cfg


def _make_head_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/head_link2/zed/cam_head",
        update_period=0.0,
        height=376,
        width=672,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.4,
            focus_distance=200.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.03, 0.0),
            rot=(0.5, 0.5, -0.5, -0.5),
            convention="isaac",
        ),
    )


# ========== DDS Topic Parsing and Matching ==========

def _trajectory_qos() -> Qos:
    return Qos(
        Policy.Reliability.BestEffort,
        Policy.Durability.Volatile,
        Policy.History.KeepLast(10),
    )


def _now_stamp() -> Time_:
    now_ns = time.time_ns()
    return Time_(sec=now_ns // 1_000_000_000, nanosec=now_ns % 1_000_000_000)


def _enabled_topics() -> dict[str, tuple[str, tuple[str, ...]]]:
    topics = {}
    if not args_cli.disable_left_arm:
        topics["left_arm"] = (
            cfg.AI_WORKER_LEFT_ARM_TOPIC,
            (*SG2_LEFT_ARM_JOINT_NAMES, *SG2_LEFT_GRIPPER_JOINT_NAMES),
        )
    if not args_cli.disable_right_arm:
        topics["right_arm"] = (
            cfg.AI_WORKER_RIGHT_ARM_TOPIC,
            (*SG2_RIGHT_ARM_JOINT_NAMES, *SG2_RIGHT_GRIPPER_JOINT_NAMES),
        )
    if not args_cli.disable_head:
        topics["head"] = (cfg.HEAD_TOPIC, SG2_HEAD_JOINT_NAMES)
    if not args_cli.disable_lift:
        topics["lift"] = (cfg.LIFT_TOPIC, SG2_LIFT_JOINT_NAMES)
    return topics


def _mobile_base_enabled() -> bool:
    return not args_cli.disable_cmd_vel


class SG2DdsBridge:
    def __init__(
        self,
        robot,
        topic_manager: TopicManager,
        topic_names: dict[str, tuple[str, tuple[str, ...]]],
        joint_states_topic: str,
        odom_topic: str,
        tf_topic: str | None,
        base_frame: str,
        odom_frame: str,
        trajectory_qos: Qos,
        cmd_vel_topic: str | None,
        swerve_modules: list[SwerveModule],
        wheel_radius: float,
        cmd_vel_timeout: float,
        publish_lift_target_state: bool,
        head_camera=None,
        head_camera_topic: str | None = None,
    ):
        self.robot = robot
        self.head_camera = head_camera
        self.odom_frame = odom_frame
        self.swerve_modules = swerve_modules
        self.cmd_vel_timeout = cmd_vel_timeout
        self.publish_lift_target_state = publish_lift_target_state
        self._warned_camera_publish_error = False
        self.swerve_controller = (
            SwerveDriveController(swerve_modules, wheel_radius) if swerve_modules else None
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
        self._initial_root_state = self.robot.data.root_state_w.clone()
        self.running = True
        self.lock = threading.Lock()
        self.pending_positions: dict[str, float] = {}
        self.latest_cmd_vel = (0.0, 0.0, 0.0)
        self.last_cmd_vel_time = 0.0
        self.unknown_joints: set[str] = set()
        self._warned_unexpected_topic_joints: set[tuple[str, str]] = set()
        self._joint_name_to_index = {
            name: index for index, name in enumerate(self.robot.data.joint_names)
        }
        self._lift_joint_id = self._joint_name_to_index.get(cfg.LIFT_JOINT_NAME)
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
        self.base_frame = self._resolve_base_frame(base_frame)
        self._base_id = self._body_names.index(self.base_frame) if self.base_frame in self._body_names else None
        self._warned_missing_base_frame = False
        self._warned_missing_swerve_joints: set[str] = set()

        self.readers = []
        self.threads = []
        self.joint_state_writer = topic_manager.topic_writer(
            topic_name=joint_states_topic,
            topic_type=JointState_,
        )
        self.odom_writer = None
        if self.odometry is not None:
            self.odom_writer = topic_manager.topic_writer(
                topic_name=odom_topic,
                topic_type=Odometry_,
            )
        self.tf_writer = None
        if tf_topic:
            self.tf_writer = topic_manager.topic_writer(
                topic_name=tf_topic,
                topic_type=TFMessage_,
            )
        self.head_camera_writer = None
        if self.head_camera is not None and head_camera_topic:
            self.head_camera_writer = topic_manager.topic_writer(
                topic_name=head_camera_topic,
                topic_type=CompressedImage_,
            )

        for label, (topic_name, fallback_joint_names) in topic_names.items():
            if not topic_name:
                continue
            reader = topic_manager.topic_reader(
                topic_name=topic_name,
                topic_type=JointTrajectory_,
                qos=trajectory_qos,
            )
            thread = threading.Thread(
                target=self._trajectory_loop,
                args=(label, reader, fallback_joint_names),
                daemon=True,
            )
            self.readers.append(reader)
            self.threads.append(thread)
            thread.start()
            print(f"[DDS] Subscribing {label}: {topic_name}")

        if cmd_vel_topic:
            cmd_vel_reader = topic_manager.topic_reader(
                topic_name=cmd_vel_topic,
                topic_type=Twist_,
                qos=trajectory_qos,
            )
            cmd_vel_thread = threading.Thread(target=self._cmd_vel_loop, args=(cmd_vel_reader,), daemon=True)
            self.readers.append(cmd_vel_reader)
            self.threads.append(cmd_vel_thread)
            cmd_vel_thread.start()
            print(f"[DDS] Subscribing cmd_vel: {cmd_vel_topic}")

    def _resolve_base_frame(self, requested_base_frame: str) -> str:
        if requested_base_frame in self._body_names:
            return requested_base_frame

        for candidate in ("base_link", "world", "arm_base_link"):
            if candidate in self._body_names:
                print(
                    f"[DDS] Requested base frame '{requested_base_frame}' was not found; "
                    f"using '{candidate}' for TF."
                )
                return candidate

        if self._body_names:
            fallback = self._body_names[0]
            print(
                f"[DDS] Requested base frame '{requested_base_frame}' was not found; "
                f"using first body '{fallback}' for TF."
            )
            return fallback
        return requested_base_frame

    # Run DDS reader loops
    def _trajectory_loop(self, label: str, reader, fallback_joint_names: tuple[str, ...]):
        try:
            while self.running:
                for msg in reader.take_iter():
                    self._store_trajectory(label, msg, fallback_joint_names)
                time.sleep(0.001)
        except Exception as exc:
            print(f"[DDS] {label} subscriber exception: {exc}")
        finally:
            try:
                reader.Close()
            except Exception:
                pass

    def _cmd_vel_loop(self, reader):
        try:
            while self.running:
                for msg in reader.take_iter():
                    self._store_cmd_vel(msg)
                time.sleep(0.001)
        except Exception as exc:
            print(f"[DDS] cmd_vel subscriber exception: {exc}")
        finally:
            try:
                reader.Close()
            except Exception:
                pass

    # Parse trajectory topics and match joints
    def _store_trajectory(self, label: str, msg, fallback_joint_names: tuple[str, ...]):
        if msg is None or not msg.points:
            return

        point = msg.points[-1]
        positions = list(point.positions)
        joint_names = list(msg.joint_names)

        if label == "lift":
            lift_position = None
            if len(joint_names) == len(positions) and cfg.LIFT_JOINT_NAME in joint_names:
                lift_position = positions[joint_names.index(cfg.LIFT_JOINT_NAME)]
            elif not joint_names and len(positions) == 1:
                lift_position = positions[0]
            if lift_position is None:
                print(
                    f"[DDS] Ignoring lift message: '{cfg.LIFT_JOINT_NAME}' "
                    f"not found in joint_names={joint_names}"
                )
                return

            with self.lock:
                self.pending_positions[cfg.LIFT_JOINT_NAME] = float(lift_position)
            return

        if not joint_names and len(positions) <= len(fallback_joint_names):
            joint_names = list(fallback_joint_names[: len(positions)])
        if len(joint_names) != len(positions):
            print(
                f"[DDS] Ignoring {label} message: joint_names={len(joint_names)} "
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
                print(f"[DDS] Ignoring {label} topic joint '{name}'; expected one of {fallback_joint_names}.")

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
        return command

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
                    print(f"[DDS] Joint '{name}' is not in the SG2 USD articulation; ignoring it.")
                continue
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
                print(f"[DDS] Swerve joint '{joint_name}' is not in the SG2 USD articulation; ignoring cmd_vel.")
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

        linear_x, linear_y, angular_z = self._current_cmd_vel()
        if self.odometry.update_from_command(linear_x, linear_y, angular_z, dt):
            self._write_kinematic_base_state()

    def _write_kinematic_base_state(self):
        if self.odometry is None:
            return

        state = self.odometry.state()
        root_state = self.robot.data.root_state_w.clone()
        root_state[:, 0] = self._initial_root_state[:, 0] + state.x
        root_state[:, 1] = self._initial_root_state[:, 1] + state.y
        root_state[:, 2] = self._initial_root_state[:, 2]

        quat_x, quat_y, quat_z, quat_w = yaw_to_quaternion(state.yaw)
        root_state[:, 3] = quat_w
        root_state[:, 4] = quat_x
        root_state[:, 5] = quat_y
        root_state[:, 6] = quat_z

        root_state[:, 7:] = 0.0

        self.robot.write_root_state_to_sim(root_state)

    # Publish robot state and close DDS resources
    def publish_joint_states(self):
        stamp = _now_stamp()
        header = Header_(stamp=stamp, frame_id=self.base_frame)

        joint_names = list(self.robot.data.joint_names)
        positions = self.robot.data.joint_pos.squeeze(0).detach().cpu().tolist()
        velocities = self.robot.data.joint_vel.squeeze(0).detach().cpu().tolist()
        efforts = [0.0] * len(joint_names)

        if self.publish_lift_target_state and self._lift_joint_id is not None:
            positions[self._lift_joint_id] = float(self.robot.data.joint_pos_target[0, self._lift_joint_id])
            velocities[self._lift_joint_id] = float(self.robot.data.joint_vel_target[0, self._lift_joint_id])

        msg = JointState_(
            header=header,
            name=joint_names,
            position=positions,
            velocity=velocities,
            effort=efforts,
        )
        try:
            self.joint_state_writer.write(msg)
        except Exception as exc:
            print(f"[DDS] joint_states write error: {exc}")

    def publish_odometry(self):
        if self.odom_writer is None or self.odometry is None:
            return

        state = self.odometry.state()
        quat_x, quat_y, quat_z, quat_w = yaw_to_quaternion(state.yaw)
        covariance = [0.0] * 36
        for index in (0, 7, 14, 21, 28, 35):
            covariance[index] = 0.001

        stamp = _now_stamp()
        msg = Odometry_(
            header=Header_(stamp=stamp, frame_id=self.odom_frame),
            child_frame_id=self.base_frame,
            pose=PoseWithCovariance_(
                pose=Pose_(
                    position=Point_(x=state.x, y=state.y, z=0.0),
                    orientation=Quaternion_(x=quat_x, y=quat_y, z=quat_z, w=quat_w),
                ),
                covariance=covariance,
            ),
            twist=TwistWithCovariance_(
                twist=Twist_(
                    linear=Vector3_(x=state.vx, y=state.vy, z=0.0),
                    angular=Vector3_(x=0.0, y=0.0, z=state.wz),
                ),
                covariance=covariance,
            ),
        )
        try:
            self.odom_writer.write(msg)
        except Exception as exc:
            print(f"[DDS] odom write error: {exc}")

    def publish_tf(self):
        if self.tf_writer is None:
            return
        if self._base_id is None:
            if not self._warned_missing_base_frame:
                self._warned_missing_base_frame = True
                print(
                    f"[DDS] Cannot publish TF: base frame '{self.base_frame}' is not in SG2 body names. "
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
            if child_frame == self.base_frame:
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
                TransformStamped_(
                    header=Header_(stamp=stamp, frame_id=self.base_frame),
                    child_frame_id=child_frame,
                    transform=Transform_(
                        translation=Vector3_(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                        rotation=Quaternion_(
                            x=float(quat_wxyz[1]),
                            y=float(quat_wxyz[2]),
                            z=float(quat_wxyz[3]),
                            w=float(quat_wxyz[0]),
                        ),
                    ),
                )
            )

        try:
            self.tf_writer.write(TFMessage_(transforms=transforms))
        except Exception as exc:
            print(f"[DDS] tf write error: {exc}")

    def publish_head_camera(self):
        if self.head_camera is None or self.head_camera_writer is None:
            return

        try:
            img = self.head_camera.data.output["rgb"][0].detach().cpu().numpy()
            if img.dtype != "uint8":
                max_value = float(img.max()) if img.size else 0.0
                if max_value <= 1.0:
                    img = img * 255.0
                img = img.clip(0, 255).astype("uint8")
            if img.shape[-1] == 4:
                img = img[:, :, :3]

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".jpg", img_bgr)
            if not success:
                raise RuntimeError("cv2.imencode('.jpg', image) failed")

            msg = CompressedImage_(
                header=Header_(stamp=_now_stamp(), frame_id=SG2_HEAD_CAMERA_NAME),
                format="jpeg",
                data=buffer.tobytes(),
            )
            self.head_camera_writer.write(msg)
        except Exception as exc:
            if not self._warned_camera_publish_error:
                self._warned_camera_publish_error = True
                print(f"[DDS] head camera publish error: {exc}")

    def shutdown(self):
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
        for reader in self.readers:
            try:
                reader.Close()
            except Exception:
                pass
        try:
            self.joint_state_writer.Close()
        except Exception:
            pass
        if self.odom_writer is not None:
            try:
                self.odom_writer.Close()
            except Exception:
                pass
        if self.tf_writer is not None:
            try:
                self.tf_writer.Close()
            except Exception:
                pass
        if self.head_camera_writer is not None:
            try:
                self.head_camera_writer.Close()
            except Exception:
                pass


# ========== Robot State ==========

def _swerve_modules() -> list[SwerveModule]:
    return [
        SwerveModule(
            steering_joint=steering_joint,
            wheel_joint=wheel_joint,
            x_offset=SG2_SWERVE_MODULE_X_OFFSETS[index],
            y_offset=SG2_SWERVE_MODULE_Y_OFFSETS[index],
            angle_offset=SG2_SWERVE_MODULE_ANGLE_OFFSETS[index],
            steering_limit_lower=cfg.AI_WORKER_SWERVE_STEERING_LIMIT_LOWER,
            steering_limit_upper=cfg.AI_WORKER_SWERVE_STEERING_LIMIT_UPPER,
            wheel_speed_limit_lower=cfg.AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
            wheel_speed_limit_upper=cfg.AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
        )
        for index, (steering_joint, wheel_joint) in enumerate(
            zip(SG2_SWERVE_STEERING_JOINTS, SG2_SWERVE_WHEEL_JOINTS)
        )
    ]


def _write_default_joint_state(robot):
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)
    robot.set_joint_velocity_target(default_joint_vel)


def _print_robot_info(robot):
    print("[INFO] SG2 joint names:")
    for joint_name in robot.data.joint_names:
        print(f"  - {joint_name}")
    print("[INFO] SG2 body names:")
    for body_name in robot.data.body_names:
        print(f"  - {body_name}")


# ========== Simulation Loop ==========

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, bridge: SG2DdsBridge):
    sim_dt = sim.get_physics_dt()
    step_period = 1.0 / cfg.STEP_HZ if cfg.STEP_HZ > 0 else 0.0
    publish_period = 1.0 / cfg.PUBLISH_HZ if cfg.PUBLISH_HZ > 0 else 0.0
    last_publish = 0.0
    last_step = time.time()
    start_time = time.time()

    while simulation_app.is_running():
        if args_cli.max_runtime > 0.0 and time.time() - start_time >= args_cli.max_runtime:
            print(f"[INFO] max_runtime reached: {args_cli.max_runtime:.3f}s")
            break

        bridge.apply_latest_targets()
        bridge.update_odometry(sim_dt)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        now = time.time()
        if publish_period == 0.0 or now - last_publish >= publish_period:
            bridge.publish_joint_states()
            bridge.publish_odometry()
            bridge.publish_tf()
            bridge.publish_head_camera()
            last_publish = now

        if step_period > 0.0:
            next_step = last_step + step_period
            sleep_time = next_step - time.time()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            last_step = next_step if sleep_time > 0.0 else time.time()


def main():
    camera_enabled = args_cli.enable_camera
    robot_cfg_template = FFW_SG2_KINEMATIC_CFG
    usd_path = robot_cfg_template.spawn.usd_path
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"SG2 USD not found: {usd_path}")

    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / cfg.STEP_HZ,
        render_interval=cfg.RENDER_INTERVAL,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(SG2_OVERVIEW_CAMERA_EYE, SG2_OVERVIEW_CAMERA_TARGET)

    scene_cfg = SG2BringupSceneCfg(num_envs=1, env_spacing=2.0)
    environment_usd_path = args_cli.environment_usd or ROBOTIS_SHOWROOM_USD_PATH
    scene_cfg.robot = _make_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    if args_cli.enable_environment:
        if "://" not in environment_usd_path and not os.path.exists(environment_usd_path):
            raise FileNotFoundError(f"Environment USD not found: {environment_usd_path}")
        scene_cfg.ground.init_state.pos = (0.0, 0.0, -0.03)
        scene_cfg.ground.spawn.color = (0.45, 0.45, 0.45)
        scene_cfg.environment = make_robotis_showroom_environment_cfg(environment_usd_path)
    if camera_enabled:
        scene_cfg.cam_head = _make_head_camera_cfg()
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    _write_default_joint_state(robot)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    if args_cli.print_robot_info:
        _print_robot_info(robot)
    head_camera = scene[SG2_HEAD_CAMERA_NAME] if camera_enabled else None

    domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))
    topic_manager = TopicManager(domain_id=domain_id)
    bridge = SG2DdsBridge(
        robot=robot,
        topic_manager=topic_manager,
        topic_names=_enabled_topics(),
        joint_states_topic=cfg.JOINT_STATES_TOPIC,
        odom_topic=cfg.ODOM_TOPIC,
        tf_topic=None if args_cli.disable_tf else cfg.TF_TOPIC,
        base_frame=args_cli.base_frame,
        odom_frame=cfg.ODOM_FRAME,
        trajectory_qos=_trajectory_qos(),
        cmd_vel_topic=cfg.CMD_VEL_TOPIC if _mobile_base_enabled() else None,
        swerve_modules=_swerve_modules() if _mobile_base_enabled() else [],
        wheel_radius=SG2_SWERVE_WHEEL_RADIUS,
        cmd_vel_timeout=cfg.CMD_VEL_TIMEOUT,
        publish_lift_target_state=not args_cli.publish_measured_lift_state,
        head_camera=head_camera,
        head_camera_topic=SG2_HEAD_CAMERA_TOPIC if camera_enabled else None,
    )

    print(f"[INFO] FFW SG2 DDS runtime ready. ROS_DOMAIN_ID={domain_id}")
    print(f"[INFO] SG2 USD: {usd_path}")
    if args_cli.enable_environment:
        print(f"[INFO] Environment: {environment_usd_path}")
    print("[DDS] JointTrajectory subscriber reliability: best_effort")
    print(f"[DDS] Publishing joint states: {cfg.JOINT_STATES_TOPIC}")
    if _mobile_base_enabled():
        print(f"[DDS] Publishing odometry: {cfg.ODOM_TOPIC} ({cfg.ODOM_FRAME} -> {bridge.base_frame})")
        print(f"[DDS] Applying swerve cmd_vel: {cfg.CMD_VEL_TOPIC}")
        print("[INFO] SG2 base integration: kinematic cmd_vel")
    if bridge.publish_lift_target_state:
        print("[DDS] Publishing lift_joint target in joint_states to keep incremental lift commands stable.")
    if not args_cli.disable_tf:
        print(f"[DDS] Publishing TF: {cfg.TF_TOPIC} ({bridge.base_frame} -> robot links)")
    if camera_enabled:
        print(f"[DDS] Publishing head camera: {SG2_HEAD_CAMERA_TOPIC}")

    try:
        run_simulator(sim, scene, bridge)
    finally:
        bridge.shutdown()


def _close_simulation_app():
    try:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=args_cli.max_runtime > 0.0)
    except TypeError:
        simulation_app.close()


if __name__ == "__main__":
    main()
    _close_simulation_app()
