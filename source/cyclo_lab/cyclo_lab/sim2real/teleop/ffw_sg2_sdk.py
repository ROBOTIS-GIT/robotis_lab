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
from collections.abc import Callable

import cv2
import torch
from pynput.keyboard import Listener

from cyclo_lab.sim2real.specs.ffw_sg2 import (
    FFW_SG2_ACTION_TOPICS,
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_CAMERA_TOPICS,
    FFW_SG2_JOINT_STATES_TOPIC,
    FFW_SG2_JOYSTICK_TRIGGER_TOPIC,
    FFW_SG2_PUBLISHED_JOINT_NAMES,
)
from cyclo_lab.sim2real.transport.ros2_zenoh import (
    COMPRESSED_IMAGE,
    JOINT_STATE,
    JOINT_TRAJECTORY,
    STRING,
    close_endpoints,
    create_publisher,
    create_subscriber,
    make_compressed_image_kwargs,
    make_joint_state_kwargs,
    ros_domain_id,
)


class FFWSG2Sdk:
    """FFW SG2 teleoperation interface over Zenoh ROS2 topics."""

    def __init__(self, env, mode: str):
        self.env = env
        self.mode = mode  # 'record' or 'inference'
        self.domain_id = ros_domain_id()
        self.left_arm_trajectory_cmd = None
        self.right_arm_trajectory_cmd = None
        self.head_joint_trajectory_cmd = None
        self.lift_joint_trajectory_cmd = None
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}
        self._first_episode = True
        self._episode_phase = "idle"
        self.lock = threading.Lock()

        self.joint_names = list(FFW_SG2_ACTION_JOINT_NAMES)
        self.published_joint_names = list(FFW_SG2_PUBLISHED_JOINT_NAMES)

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
            create_subscriber(
                topic=FFW_SG2_JOYSTICK_TRIGGER_TOPIC,
                msg_type=STRING,
                callback=self._on_joystick_trigger,
            ),
        ]

        self.joint_state_writer = create_publisher(FFW_SG2_JOINT_STATES_TOPIC, JOINT_STATE)
        self.camera_writers = {
            cam_name: create_publisher(topic, COMPRESSED_IMAGE)
            for cam_name, topic in FFW_SG2_CAMERA_TOPICS.items()
        }
        self.publishers = [self.joint_state_writer, *self.camera_writers.values()]

        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

        print(f"[Zenoh ROS2] FFWSG2Sdk ready. ROS_DOMAIN_ID={self.domain_id}")
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
        elif self.mode == "inference":
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start robot control")

    def _on_press(self, key):
        try:
            if self.mode == "record":
                if key.char == "b":
                    self._start_recording()
                elif key.char == "r":
                    self._skip_episode()
                elif key.char == "n":
                    self._save_episode()
            elif self.mode == "inference":
                if key.char == "b":
                    self._started = True
                    self._reset_state = False
                elif key.char == "r":
                    self._started = False
                    self._reset_state = True
                    self._call_callback("R")
        except AttributeError:
            pass

    def _call_callback(self, key):
        if key in self._additional_callbacks:
            self._additional_callbacks[key]()

    def _start_recording(self):
        self._started = True
        self._reset_state = False
        if self._first_episode:
            self._first_episode = False
        self._episode_phase = "recording"

    def _save_episode(self):
        self._started = False
        self._reset_state = True
        self._call_callback("N")
        self._episode_phase = "idle"

    def _skip_episode(self):
        self._started = False
        self._reset_state = True
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

    # ----------------------
    # Publishers
    # ----------------------
    def _publish_joint_states(self):
        obs_joint_name = self.env.scene["robot"].data.joint_names
        all_positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()
        all_velocities = self.env.scene["robot"].data.joint_vel.squeeze(0).tolist()
        all_efforts = [0.0] * len(all_positions)

        if isinstance(all_positions[0], list):
            all_positions = [p for sub in all_positions for p in sub]
        if isinstance(all_velocities[0], list):
            all_velocities = [v for sub in all_velocities for v in sub]

        published_names = [name for name in self.published_joint_names if name in obs_joint_name]
        indices = [obs_joint_name.index(name) for name in published_names]
        positions = [all_positions[i] for i in indices]
        velocities = [all_velocities[i] for i in indices]
        efforts = [all_efforts[i] for i in indices]

        try:
            self.joint_state_writer.publish(
                **make_joint_state_kwargs(
                    names=published_names,
                    positions=positions,
                    velocities=velocities,
                    efforts=efforts,
                    frame_id="base_link",
                )
            )
        except Exception as exc:
            print("[Zenoh ROS2] joint_states publish error:", exc)

    def _publish_camera(self, cam_name: str):
        writer = self.camera_writers.get(cam_name)
        if writer is None:
            return
        try:
            img = self.env.scene[cam_name].data.output["rgb"][0].cpu().numpy()
            if img.dtype != "uint8":
                max_value = float(img.max()) if img.size else 0.0
                if max_value <= 1.0:
                    img = img * 255.0
                img = img.clip(0, 255).astype("uint8")
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            success, buffer = cv2.imencode(".jpg", img)
            if not success:
                raise RuntimeError("cv2.imencode('.jpg', image) failed")

            writer.publish(
                **make_compressed_image_kwargs(
                    data=buffer.tobytes(),
                    frame_id="camera_frame",
                    fmt="jpeg",
                )
            )
        except Exception as exc:
            print(f"[Zenoh ROS2] camera publish error for {cam_name}:", exc)

    def _compute_action_state(self):
        state = {"reset": self._reset_state, "started": self._started}
        if state["reset"]:
            self._reset_state = False
            return state
        state["joint_state"] = self._get_device_state()
        return state

    def _get_device_state(self):
        with self.lock:
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

            if self.left_arm_trajectory_cmd:
                joint_state.update(self.left_arm_trajectory_cmd)
            if self.right_arm_trajectory_cmd:
                joint_state.update(self.right_arm_trajectory_cmd)
            if self.head_joint_trajectory_cmd:
                joint_state.update(self.head_joint_trajectory_cmd)
            if self.lift_joint_trajectory_cmd:
                joint_state.update(self.lift_joint_trajectory_cmd)

            return joint_state

    def get_action(self):
        action = self._compute_action_state()
        if action["reset"]:
            return {"reset": True}
        if not action["started"]:
            return None

        joint_state = action["joint_state"]
        positions = [joint_state.get(name, 0.0) for name in self.joint_names]
        return torch.tensor(positions, device=self.env.device, dtype=torch.float32).unsqueeze(0)

    def publish_observations(self):
        self._publish_joint_states()
        self._publish_camera("cam_head")

    # ----------------------
    # Utility
    # ----------------------
    def shutdown(self):
        close_endpoints(self.subscribers)
        close_endpoints(self.publishers)
        try:
            self.listener.stop()
        except Exception:
            pass
        print("FFWSG2Sdk shutdown complete")

    def reset(self):
        self._reset_state = False

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func
