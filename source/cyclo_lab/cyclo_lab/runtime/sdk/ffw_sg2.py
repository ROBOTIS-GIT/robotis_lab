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

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import torch
from pynput.keyboard import Listener

from cyclo_lab.runtime.publishers.articulation_state_publisher import ArticulationStatePublisher
from cyclo_lab.runtime.publishers.camera_publishers import publish_compressed_camera
from cyclo_lab.robot_specs.ffw.sg2 import (
    BASE_BODY as FFW_SG2_BASE_BODY,
    BASE_FRAME as FFW_SG2_BASE_FRAME,
    FFW_SG2_ACTION_TOPICS,
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_CAMERA_TOPICS,
    FFW_SG2_JOYSTICK_TRIGGER_TOPIC,
    FFW_SG2_PUBLISHED_JOINT_NAMES,
    JOINT_STATES_TOPIC,
    ODOM_FRAME as FFW_SG2_ODOM_FRAME,
    ODOM_TOPIC as FFW_SG2_ODOM_TOPIC,
    TF_TOPIC as FFW_SG2_TF_TOPIC,
)
from cyclo_lab.runtime.config import FFW_SG2_CMD_VEL_TIMEOUT
from cyclo_lab.runtime.transport.ros2_zenoh import (
    COMPRESSED_IMAGE,
    JOINT_TRAJECTORY,
    STRING,
    TWIST,
    close_endpoints,
    create_publisher,
    create_subscriber,
    ros_domain_id,
)


class FFWSG2Sdk:
    """FFW SG2 teleoperation interface over Zenoh ROS2 topics."""

    def __init__(
        self,
        env,
        mode: str,
        camera_publish_hz: float | None = None,
        publish_odometry_tf: bool = True,
        enable_joystick_trigger: bool = True,
        enable_keyboard_listener: bool = True,
    ):
        self.env = env
        self.mode = mode  # 'record' or 'inference'
        self.domain_id = ros_domain_id()
        self.camera_publish_hz = camera_publish_hz
        self.publish_odometry_tf = publish_odometry_tf
        self.enable_joystick_trigger = enable_joystick_trigger
        self.enable_keyboard_listener = enable_keyboard_listener
        self._last_camera_publish_time = 0.0
        self.left_arm_trajectory_cmd = None
        self.right_arm_trajectory_cmd = None
        self.head_joint_trajectory_cmd = None
        self.lift_joint_trajectory_cmd = None
        self._target_joint_state = None
        self.latest_cmd_vel = (0.0, 0.0, 0.0)
        self.last_cmd_vel_time = 0.0
        self.cmd_vel_timeout = FFW_SG2_CMD_VEL_TIMEOUT
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}
        self._first_episode = True
        self._episode_phase = "idle"
        self.lock = threading.Lock()

        self.joint_names = list(FFW_SG2_ACTION_JOINT_NAMES)
        self.published_joint_names = list(FFW_SG2_PUBLISHED_JOINT_NAMES)
        self.total_action_dim = getattr(self.env.action_manager, "total_action_dim", len(self.joint_names))
        self.include_base_action = (
            "base_action" in getattr(self.env.action_manager, "active_terms", [])
            and self.total_action_dim == len(self.joint_names) + 3
        )

        self.subscribers = [
            create_subscriber(
                topic=FFW_SG2_ACTION_TOPICS["left_arm"],
                msg_type=JOINT_TRAJECTORY,
                callback=lambda msg: self._on_joint_trajectory("left_arm", msg),
            ),
            create_subscriber(
                topic=FFW_SG2_ACTION_TOPICS["right_arm"],
                msg_type=JOINT_TRAJECTORY,
                callback=lambda msg: self._on_joint_trajectory("right_arm", msg),
            ),
            create_subscriber(
                topic=FFW_SG2_ACTION_TOPICS["head"],
                msg_type=JOINT_TRAJECTORY,
                callback=lambda msg: self._on_joint_trajectory("head", msg),
            ),
            create_subscriber(
                topic=FFW_SG2_ACTION_TOPICS["lift"],
                msg_type=JOINT_TRAJECTORY,
                callback=lambda msg: self._on_joint_trajectory("lift", msg),
            ),
        ]
        if self.enable_joystick_trigger:
            self.subscribers.append(
                create_subscriber(
                    topic=FFW_SG2_JOYSTICK_TRIGGER_TOPIC,
                    msg_type=STRING,
                    callback=self._on_joystick_trigger,
                )
            )
        if self.include_base_action:
            self.subscribers.append(
                create_subscriber(
                    topic=FFW_SG2_ACTION_TOPICS["mobile"],
                    msg_type=TWIST,
                    callback=self._on_cmd_vel,
                )
            )

        self.state_publisher = ArticulationStatePublisher(
            self.env.scene["robot"],
            joint_names=self.published_joint_names,
            joint_states_topic=JOINT_STATES_TOPIC,
            base_frame=FFW_SG2_BASE_FRAME,
            base_body=FFW_SG2_BASE_BODY,
            odom_topic=FFW_SG2_ODOM_TOPIC if self.publish_odometry_tf else None,
            tf_topic=FFW_SG2_TF_TOPIC if self.publish_odometry_tf else None,
            odom_frame=FFW_SG2_ODOM_FRAME,
        )
        self.camera_writers = {}
        if self.camera_publish_hz is None or self.camera_publish_hz != 0.0:
            self.camera_writers = {
                cam_name: create_publisher(topic, COMPRESSED_IMAGE)
                for cam_name, topic in FFW_SG2_CAMERA_TOPICS.items()
            }
        self.publishers = [*self.state_publisher.publishers, *self.camera_writers.values()]

        self.listener = None
        if self.enable_keyboard_listener:
            self.listener = Listener(on_press=self._on_press)
            self.listener.start()

        print(f"[Zenoh ROS2] FFWSG2Sdk ready. ROS_DOMAIN_ID={self.domain_id}")
        print(
            "[Zenoh ROS2] FFW_SG2 action mode: "
            f"{self.total_action_dim}D ({'joint+base' if self.include_base_action else 'joint-only'})"
        )
        print(f"[Zenoh ROS2] Publishing joint states: {JOINT_STATES_TOPIC}")
        if self.publish_odometry_tf:
            print(
                f"[Zenoh ROS2] Publishing odometry: "
                f"{FFW_SG2_ODOM_TOPIC} ({FFW_SG2_ODOM_FRAME} -> {FFW_SG2_BASE_FRAME})"
            )
            print(f"[Zenoh ROS2] Publishing TF: {FFW_SG2_TF_TOPIC} ({FFW_SG2_BASE_FRAME} -> robot links)")
        else:
            print("[Zenoh ROS2] Odometry/TF publishing disabled for this session.")
        if self.camera_publish_hz == 0.0:
            print("[Zenoh ROS2] Camera topic publishing disabled for this session.")
        elif self.camera_publish_hz is not None and self.camera_publish_hz > 0.0:
            print(f"[Zenoh ROS2] Camera topic publishing throttled to {self.camera_publish_hz:g} Hz.")
        if not self.enable_joystick_trigger:
            print("[Zenoh ROS2] Joystick trigger subscription disabled for this session.")
        if not self.enable_keyboard_listener:
            print("[Zenoh ROS2] Keyboard listener disabled for this session.")
        self._keyboard_controls()

    # ----------------------
    # Keyboard controls
    # ----------------------
    def _keyboard_controls(self):
        print("\n[Control] Press keys to control the FFW_SG2 robot:")
        if self.mode == "record":
            print("[N / Right Joystick Button] Save successful episode and proceed to the next one")
            print("[R / Left Joystick Button] Skip failed episode (not saved) and proceed to the next one")
            print("[B / Right Joystick Button] Start recording the current episode")
            print("[Info] Robot control is always active in record mode.")
        elif self.mode == "inference":
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start robot control")

    def _on_press(self, key):
        key_char = getattr(key, "char", None)
        if key_char is None:
            return
        key_char = key_char.lower()

        if self.mode == "record":
            if key_char == "b":
                self._start_recording()
            elif key_char == "r":
                self._skip_episode()
            elif key_char == "n":
                self._save_episode()
        elif self.mode == "inference":
            if key_char == "b":
                self._started = True
                self._reset_state = False
            elif key_char == "r":
                self._started = False
                self._reset_state = True
                self._call_callback("R")

    def _call_callback(self, key):
        if key in self._additional_callbacks:
            self._additional_callbacks[key]()

    def _start_recording(self):
        if self._episode_phase == "recording":
            return
        print("[Control] Start recording requested.")
        self._started = True
        self._reset_state = False
        with self.lock:
            if self._target_joint_state is None:
                self._target_joint_state = self._read_current_joint_state()
        if self._first_episode:
            self._first_episode = False
        self._episode_phase = "recording"
        self._call_callback("B")

    def start_recording(self):
        self._start_recording()

    def _save_episode(self):
        if self.mode == "record" and self._episode_phase != "recording":
            print("[Control] Save ignored because recording has not started.")
            return
        print("[Control] Save episode requested.")
        self._started = False
        self._reset_state = True
        self._clear_command_cache()
        self._call_callback("N")
        self._episode_phase = "idle"

    def _skip_episode(self):
        print("[Control] Reset/skip episode requested.")
        self._started = False
        self._reset_state = True
        self._clear_command_cache()
        self._call_callback("R")
        if self._episode_phase == "recording" and not self._first_episode:
            self._first_episode = True
            self._episode_phase = "idle"

    # ----------------------
    # Subscribers
    # ----------------------
    def _on_joint_trajectory(self, label: str, msg):
        if msg is None or not msg.points:
            return
        joint_dict = {name: float(pos) for name, pos in zip(msg.joint_names, msg.points[-1].positions)}
        with self.lock:
            if label == "left_arm":
                self.left_arm_trajectory_cmd = joint_dict
            elif label == "right_arm":
                self.right_arm_trajectory_cmd = joint_dict
            elif label == "head":
                self.head_joint_trajectory_cmd = self.head_joint_trajectory_cmd or {}
                self.head_joint_trajectory_cmd.update(joint_dict)
            elif label == "lift":
                self.lift_joint_trajectory_cmd = self.lift_joint_trajectory_cmd or {}
                self.lift_joint_trajectory_cmd.update(joint_dict)

    def _on_joystick_trigger(self, msg):
        if self.mode != "record" or msg is None:
            return
        joystick_trigger = getattr(msg, "data", "")
        if joystick_trigger == "right":
            if self._first_episode:
                self._start_recording()
            elif self._episode_phase == "recording":
                self._save_episode()
            elif self._episode_phase == "idle":
                self._start_recording()
        elif joystick_trigger == "left":
            self._skip_episode()

    def _on_cmd_vel(self, msg):
        if msg is None:
            return
        with self.lock:
            self.latest_cmd_vel = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
            self.last_cmd_vel_time = time.monotonic()

    def _current_cmd_vel(self) -> tuple[float, float, float]:
        with self.lock:
            cmd_vel = self.latest_cmd_vel
            last_msg_time = self.last_cmd_vel_time
        if last_msg_time == 0.0:
            return 0.0, 0.0, 0.0
        if self.cmd_vel_timeout > 0.0 and time.monotonic() - last_msg_time > self.cmd_vel_timeout:
            return 0.0, 0.0, 0.0
        return cmd_vel

    def _clear_command_cache(self):
        with self.lock:
            self.left_arm_trajectory_cmd = None
            self.right_arm_trajectory_cmd = None
            self.head_joint_trajectory_cmd = None
            self.lift_joint_trajectory_cmd = None
            self._target_joint_state = None
            self.latest_cmd_vel = (0.0, 0.0, 0.0)
            self.last_cmd_vel_time = 0.0
        self.state_publisher.reset_odom_origin()

    # ----------------------
    # Publishers
    # ----------------------
    def _publish_camera(self, cam_name: str):
        writer = self.camera_writers.get(cam_name)
        if writer is None:
            return
        if cam_name not in self.env.scene.keys():
            return
        try:
            publish_compressed_camera(cam_name, self.env.scene[cam_name], writer, frame_id="camera_frame")
        except Exception as exc:
            print(f"[Zenoh ROS2] camera publish error for {cam_name}:", exc)

    def _compute_action_state(self):
        state = {"reset": self._reset_state, "started": self._started}
        if state["reset"]:
            self._reset_state = False
            return state
        state["joint_state"] = self._get_device_state()
        return state

    def _read_current_joint_state(self):
        obs_joint_name = self.env.scene["robot"].data.joint_names
        all_positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()

        if isinstance(all_positions[0], list):
            all_positions = [p for sub in all_positions for p in sub]

        joint_state = {}
        for name in self.joint_names:
            if name in obs_joint_name:
                joint_state[name] = all_positions[obs_joint_name.index(name)]
            else:
                joint_state[name] = 0.0
        return joint_state

    def _get_device_state(self):
        with self.lock:
            if self._target_joint_state is None:
                self._target_joint_state = self._read_current_joint_state()
            joint_state = dict(self._target_joint_state)

            if self.left_arm_trajectory_cmd:
                joint_state.update(self.left_arm_trajectory_cmd)
            if self.right_arm_trajectory_cmd:
                joint_state.update(self.right_arm_trajectory_cmd)
            if self.head_joint_trajectory_cmd:
                joint_state.update(self.head_joint_trajectory_cmd)
            if self.lift_joint_trajectory_cmd:
                joint_state.update(self.lift_joint_trajectory_cmd)

            self._target_joint_state.update({name: joint_state[name] for name in self.joint_names})
            return joint_state

    def get_action(self):
        action = self._compute_action_state()
        if action["reset"]:
            return {"reset": True}
        if self.mode != "record" and not action["started"]:
            return None

        joint_state = action["joint_state"]
        positions = [joint_state.get(name, 0.0) for name in self.joint_names]
        if self.include_base_action:
            positions.extend(self._current_cmd_vel())
        return torch.tensor(positions, device=self.env.device, dtype=torch.float32).unsqueeze(0)

    def publish_observations(self):
        self.state_publisher.publish_all()
        if not self.camera_writers:
            return
        if self.camera_publish_hz is not None and self.camera_publish_hz > 0.0:
            now = time.monotonic()
            if now - self._last_camera_publish_time < 1.0 / self.camera_publish_hz:
                return
            self._last_camera_publish_time = now
        for cam_name in self.camera_writers:
            self._publish_camera(cam_name)

    # ----------------------
    # Utility
    # ----------------------
    def shutdown(self):
        close_endpoints(self.subscribers)
        close_endpoints(self.publishers)
        if self.listener is not None:
            try:
                self.listener.stop()
            except Exception:
                pass
        print("FFWSG2Sdk shutdown complete")

    def reset(self):
        self._reset_state = False
        self._clear_command_cache()

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func
