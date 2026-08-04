# Copyright 2026 ROBOTIS CO., LTD.
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
# Author: Seongwoo Kim

"""FFW-SG2 observation and action mapping for GR00T checkpoints."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np
from cyclo_arena.policies.adapters.base import Gr00tRobotAdapter

FFW_SG2_MANIP_ACTION_DIM = 16
FFW_SG2_FULL_ACTION_DIM = 19
FFW_SG2_MOBILE_ACTION_DIM = 22
FFW_SG2_SIDE_ACTION_DIM = 8
FFW_SG2_MODALITY_KEYS = {
    "video": ("cam_left_head",),
    "state": ("arm_left", "arm_right"),
    "action": ("arm_left", "arm_right"),
    "language": ("annotation.human.task_description",),
}
FFW_SG2_SHOWROOM_MODALITY_KEYS = {
    "video": ("cam_left_head", "cam_left_wrist", "cam_right_wrist"),
    "state": ("arm_left", "arm_right", "odometry"),
    "action": ("arm_left", "arm_right", "odometry"),
    "language": ("annotation.human.task_description",),
}
FFW_SG2_SHOWROOM_CAMERA_KEYS = {
    "cam_left_head": "camera_obs.cam_head_rgb",
    "cam_left_wrist": "camera_obs.cam_wrist_left_rgb",
    "cam_right_wrist": "camera_obs.cam_wrist_right_rgb",
}


def _config_value(config: Any, name: str) -> Any:
    """Read one field from a GR00T ModalityConfig or decoded mapping."""
    return config[name] if isinstance(config, Mapping) else getattr(config, name)


class FFWSG2Gr00tAdapter(Gr00tRobotAdapter):
    """Map FFW-SG2 joint/camera observations to its fine-tuned schema."""

    def __init__(
        self,
        modality_configs: Mapping[str, Any],
        action_repeat: int = 2,
    ) -> None:
        assert action_repeat > 0, "action_repeat must be positive"
        self._action_repeat = action_repeat
        self._delta_indices = {
            modality: self._validate_modality(modality_configs, modality) for modality in FFW_SG2_MODALITY_KEYS
        }
        self._video_history: deque[np.ndarray] = deque(maxlen=self._history_size(self._delta_indices["video"]))
        self._state_history: deque[np.ndarray] = deque(maxlen=self._history_size(self._delta_indices["state"]))
        self._current_joint_pos: np.ndarray | None = None

    @staticmethod
    def _validate_modality(modality_configs: Mapping[str, Any], modality: str) -> tuple[int, ...]:
        assert modality in modality_configs, f"GR00T schema is missing {modality!r}"
        config = modality_configs[modality]
        actual_keys = tuple(_config_value(config, "modality_keys"))
        expected_keys = FFW_SG2_MODALITY_KEYS[modality]
        assert actual_keys == expected_keys, f"GR00T {modality} keys={actual_keys!r}, expected {expected_keys!r}"
        delta_indices = tuple(int(value) for value in _config_value(config, "delta_indices"))
        assert delta_indices, f"GR00T {modality} delta_indices must not be empty"
        if modality in {"video", "state"}:
            assert all(
                value <= 0 for value in delta_indices
            ), f"GR00T {modality} observation offsets must be non-positive: {delta_indices!r}"
        return delta_indices

    @staticmethod
    def _history_size(delta_indices: tuple[int, ...]) -> int:
        return 1 - min(delta_indices)

    @staticmethod
    def _select_history(history: deque[np.ndarray], delta_indices: tuple[int, ...]) -> np.ndarray:
        assert history, "GR00T observation history is empty"
        samples = tuple(history)
        latest_index = len(samples) - 1
        selected = [samples[max(0, latest_index + delta_index)] for delta_index in delta_indices]
        return np.stack(selected, axis=1)

    @staticmethod
    def _rgb_to_uint8(rgb: Any) -> np.ndarray:
        rgb_array = np.asarray(rgb)
        if np.issubdtype(rgb_array.dtype, np.floating):
            if rgb_array.size and float(rgb_array.max()) <= 1.0:
                rgb_array = rgb_array * 255.0
        return np.clip(rgb_array, 0.0, 255.0).astype(np.uint8)

    @property
    def observation_keys(self) -> list[str]:
        return ["policy.joint_pos", "camera_obs.cam_head_rgb"]

    @property
    def action_dim(self) -> int:
        return FFW_SG2_FULL_ACTION_DIM

    @property
    def model_action_horizon(self) -> int:
        return len(self._delta_indices["action"])

    def build_policy_observation(
        self,
        observation: Mapping[str, Any],
        task_description: str,
    ) -> dict[str, Any]:
        joint_pos = np.asarray(observation["policy.joint_pos"], dtype=np.float32)
        head_rgb = self._rgb_to_uint8(observation["camera_obs.cam_head_rgb"])
        assert (
            joint_pos.ndim == 2 and joint_pos.shape[-1] == self.action_dim
        ), f"Expected batched FFW-SG2 joint state (*, {self.action_dim}), got {joint_pos.shape}"
        assert (
            head_rgb.ndim == 4 and head_rgb.shape[0] == joint_pos.shape[0]
        ), f"Expected batched RGB images for {joint_pos.shape[0]} environments, got {head_rgb.shape}"
        assert head_rgb.shape[-1] == 3, f"Expected RGB channel dimension, got {head_rgb.shape}"

        self._current_joint_pos = joint_pos
        self._state_history.append(joint_pos)
        self._video_history.append(head_rgb)
        state_history = self._select_history(self._state_history, self._delta_indices["state"])
        video_history = self._select_history(self._video_history, self._delta_indices["video"])
        language_horizon = len(self._delta_indices["language"])
        return {
            "language": {
                "annotation.human.task_description": [
                    [task_description] * language_horizon for _ in range(joint_pos.shape[0])
                ]
            },
            "video": {"cam_left_head": video_history},
            "state": {
                "arm_left": state_history[..., :FFW_SG2_SIDE_ACTION_DIM],
                "arm_right": state_history[
                    ...,
                    FFW_SG2_SIDE_ACTION_DIM:FFW_SG2_MANIP_ACTION_DIM,
                ],
            },
        }

    def build_action_chunk(
        self,
        policy_action: Mapping[str, Any],
    ) -> np.ndarray:
        assert self._current_joint_pos is not None, "An observation must be processed before its action"
        left_action = np.asarray(policy_action["arm_left"], dtype=np.float32)
        right_action = np.asarray(policy_action["arm_right"], dtype=np.float32)
        expected_shape = (
            self._current_joint_pos.shape[0],
            self.model_action_horizon,
            FFW_SG2_SIDE_ACTION_DIM,
        )
        assert left_action.shape == expected_shape, f"Unexpected left action shape: {left_action.shape}"
        assert right_action.shape == expected_shape, f"Unexpected right action shape: {right_action.shape}"

        manipulation_action = np.concatenate((left_action, right_action), axis=-1)
        lift_head_hold = np.repeat(
            self._current_joint_pos[:, None, FFW_SG2_MANIP_ACTION_DIM:],
            self.model_action_horizon,
            axis=1,
        )
        full_action = np.concatenate((manipulation_action, lift_head_hold), axis=-1)
        return np.repeat(full_action, self._action_repeat, axis=1)

    def reset(self) -> None:
        self._video_history.clear()
        self._state_history.clear()
        self._current_joint_pos = None


class FFWSG2ShowroomGr00tAdapter(FFWSG2Gr00tAdapter):
    """Map the three-camera mobile showroom checkpoint to FFW-SG2."""

    def __init__(
        self,
        modality_configs: Mapping[str, Any],
        action_repeat: int = 3,
    ) -> None:
        assert action_repeat > 0, "action_repeat must be positive"
        self._action_repeat = action_repeat
        self._delta_indices = {
            modality: self._validate_showroom_modality(
                modality_configs,
                modality,
            )
            for modality in FFW_SG2_SHOWROOM_MODALITY_KEYS
        }
        video_history_size = self._history_size(self._delta_indices["video"])
        self._video_histories: dict[str, deque[np.ndarray]] = {
            camera_key: deque(maxlen=video_history_size) for camera_key in FFW_SG2_SHOWROOM_CAMERA_KEYS
        }
        state_history_size = self._history_size(self._delta_indices["state"])
        self._state_history = deque(maxlen=state_history_size)
        self._base_twist_history: deque[np.ndarray] = deque(maxlen=state_history_size)
        self._current_joint_pos: np.ndarray | None = None

    @staticmethod
    def _validate_showroom_modality(
        modality_configs: Mapping[str, Any],
        modality: str,
    ) -> tuple[int, ...]:
        assert modality in modality_configs, f"GR00T schema is missing {modality!r}"
        config = modality_configs[modality]
        actual_keys = tuple(_config_value(config, "modality_keys"))
        expected_keys = FFW_SG2_SHOWROOM_MODALITY_KEYS[modality]
        assert actual_keys == expected_keys, f"GR00T {modality} keys={actual_keys!r}, expected {expected_keys!r}"
        delta_indices = tuple(int(value) for value in _config_value(config, "delta_indices"))
        assert delta_indices, f"GR00T {modality} delta_indices must not be empty"
        if modality in {"video", "state"}:
            assert all(
                value <= 0 for value in delta_indices
            ), f"GR00T {modality} observation offsets must be non-positive: {delta_indices!r}"
        return delta_indices

    @property
    def observation_keys(self) -> list[str]:
        return [
            "policy.joint_pos",
            "policy.base_twist",
            *FFW_SG2_SHOWROOM_CAMERA_KEYS.values(),
        ]

    @property
    def action_dim(self) -> int:
        return FFW_SG2_MOBILE_ACTION_DIM

    def build_policy_observation(
        self,
        observation: Mapping[str, Any],
        task_description: str,
    ) -> dict[str, Any]:
        joint_pos = np.asarray(observation["policy.joint_pos"], dtype=np.float32)
        base_twist = np.asarray(
            observation["policy.base_twist"],
            dtype=np.float32,
        )
        assert (
            joint_pos.ndim == 2 and joint_pos.shape[-1] == FFW_SG2_FULL_ACTION_DIM
        ), f"Expected batched FFW-SG2 joint state (*, {FFW_SG2_FULL_ACTION_DIM}), got {joint_pos.shape}"
        assert base_twist.shape == (
            joint_pos.shape[0],
            3,
        ), f"Expected batched FFW-SG2 base twist ({joint_pos.shape[0]}, 3), got {base_twist.shape}"

        camera_samples = {
            camera_key: self._rgb_to_uint8(observation[observation_key])
            for camera_key, observation_key in FFW_SG2_SHOWROOM_CAMERA_KEYS.items()
        }
        for camera_key, camera_sample in camera_samples.items():
            assert camera_sample.ndim == 4, f"Expected batched RGB images for {camera_key!r}, got {camera_sample.shape}"
            assert (
                camera_sample.shape[0] == joint_pos.shape[0]
            ), f"Expected {joint_pos.shape[0]} images for {camera_key!r}, got {camera_sample.shape[0]}"
            assert (
                camera_sample.shape[-1] == 3
            ), f"Expected RGB channel dimension for {camera_key!r}, got {camera_sample.shape}"

        self._current_joint_pos = joint_pos
        self._state_history.append(joint_pos)
        self._base_twist_history.append(base_twist)
        for camera_key, camera_sample in camera_samples.items():
            self._video_histories[camera_key].append(camera_sample)

        state_history = self._select_history(
            self._state_history,
            self._delta_indices["state"],
        )
        base_twist_history = self._select_history(
            self._base_twist_history,
            self._delta_indices["state"],
        )
        video = {
            camera_key: self._select_history(
                self._video_histories[camera_key],
                self._delta_indices["video"],
            )
            for camera_key in FFW_SG2_SHOWROOM_CAMERA_KEYS
        }
        language_horizon = len(self._delta_indices["language"])
        return {
            "language": {
                "annotation.human.task_description": [
                    [task_description] * language_horizon for _ in range(joint_pos.shape[0])
                ]
            },
            "video": video,
            "state": {
                "arm_left": state_history[..., :FFW_SG2_SIDE_ACTION_DIM],
                "arm_right": state_history[
                    ...,
                    FFW_SG2_SIDE_ACTION_DIM:FFW_SG2_MANIP_ACTION_DIM,
                ],
                "odometry": base_twist_history,
            },
        }

    def build_action_chunk(
        self,
        policy_action: Mapping[str, Any],
    ) -> np.ndarray:
        assert self._current_joint_pos is not None, "An observation must be processed before its action"
        left_action = np.asarray(policy_action["arm_left"], dtype=np.float32)
        right_action = np.asarray(policy_action["arm_right"], dtype=np.float32)
        base_action = np.asarray(policy_action["odometry"], dtype=np.float32)
        side_shape = (
            self._current_joint_pos.shape[0],
            self.model_action_horizon,
            FFW_SG2_SIDE_ACTION_DIM,
        )
        base_shape = (*side_shape[:2], 3)
        assert left_action.shape == side_shape, f"Unexpected left action shape: {left_action.shape}"
        assert right_action.shape == side_shape, f"Unexpected right action shape: {right_action.shape}"
        assert base_action.shape == base_shape, f"Unexpected odometry action shape: {base_action.shape}"

        manipulation_action = np.concatenate(
            (left_action, right_action),
            axis=-1,
        )
        lift_head_hold = np.repeat(
            self._current_joint_pos[:, None, FFW_SG2_MANIP_ACTION_DIM:],
            self.model_action_horizon,
            axis=1,
        )
        full_action = np.concatenate(
            (manipulation_action, lift_head_hold, base_action),
            axis=-1,
        )
        return np.repeat(full_action, self._action_repeat, axis=1)

    def reset(self) -> None:
        for history in self._video_histories.values():
            history.clear()
        self._state_history.clear()
        self._base_twist_history.clear()
        self._current_joint_pos = None
