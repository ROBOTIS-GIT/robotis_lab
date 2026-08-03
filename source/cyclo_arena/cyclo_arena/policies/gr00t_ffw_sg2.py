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

"""Checkpoint-driven GR00T N1.6/N1.7 policy for the FFW-SG2 embodiment."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.policy.action_chunking import ActionChunkingState
from isaaclab_arena.policy.policy_base import PolicyBase

from cyclo_arena.policies.gr00t_rpc import Gr00tPolicyClient

FFW_SG2_MANIP_ACTION_DIM = 16
FFW_SG2_FULL_ACTION_DIM = 19
FFW_SG2_SIDE_ACTION_DIM = 8
FFW_SG2_MODALITY_KEYS = {
    "video": ("cam_left_head",),
    "state": ("arm_left", "arm_right"),
    "action": ("arm_left", "arm_right"),
    "language": ("annotation.human.task_description",),
}


@dataclass
class Gr00tFFWSG2RemotePolicyCfg:
    """Configure the version-independent FFW-SG2 GR00T remote policy."""

    num_envs: int = 1
    device: str = "cuda:0"
    remote_host: str = "127.0.0.1"
    remote_port: int = 5555
    remote_timeout_ms: int = 120000
    remote_kill_on_exit: bool = False
    action_repeat: int = 2
    action_chunk_length: int = 32

    def __post_init__(self) -> None:
        assert self.num_envs > 0, "num_envs must be positive"
        assert self.action_repeat > 0, "action_repeat must be positive"
        assert self.action_chunk_length > 0, "action_chunk_length must be positive"


class Gr00tFFWSG2RemotePolicy(PolicyBase):
    """Run compatible FFW-SG2 GR00T N1.6 and N1.7 checkpoints."""

    config_class = Gr00tFFWSG2RemotePolicyCfg

    def __init__(self, config: Gr00tFFWSG2RemotePolicyCfg):
        super().__init__(config)
        self.device = torch.device(config.device)
        self.task_description: str | None = None
        self._client: Gr00tPolicyClient | None = Gr00tPolicyClient(
            host=config.remote_host,
            port=config.remote_port,
            timeout_ms=config.remote_timeout_ms,
        )
        if not self._client.ping():
            raise ConnectionError(
                f"Cannot reach GR00T server at {config.remote_host}:"
                f"{config.remote_port}"
            )

        schema = self._client.get_modality_config()
        self._video_delta_indices = self._delta_indices(schema, "video")
        self._state_delta_indices = self._delta_indices(schema, "state")
        self._language_horizon = len(self._delta_indices(schema, "language"))
        self._model_action_horizon = len(self._delta_indices(schema, "action"))
        repeated_action_horizon = (
            self._model_action_horizon * config.action_repeat
        )
        assert config.action_chunk_length <= repeated_action_horizon, (
            f"action_chunk_length={config.action_chunk_length} exceeds the model's "
            f"repeated action horizon={repeated_action_horizon}"
        )

        self._video_history: deque[np.ndarray] = deque(
            maxlen=self._history_size(self._video_delta_indices)
        )
        self._state_history: deque[np.ndarray] = deque(
            maxlen=self._history_size(self._state_delta_indices)
        )
        self._chunking_state: ActionChunkingState | None = ActionChunkingState(
            num_envs=config.num_envs,
            action_chunk_length=config.action_chunk_length,
            action_horizon=repeated_action_horizon,
            action_dim=FFW_SG2_FULL_ACTION_DIM,
            device=self.device,
        )

    @staticmethod
    def _delta_indices(
        schema: Mapping[str, Mapping[str, Any]], modality: str
    ) -> tuple[int, ...]:
        assert modality in schema, f"GR00T schema is missing {modality!r}"
        values = schema[modality]
        expected_keys = FFW_SG2_MODALITY_KEYS[modality]
        actual_keys = tuple(values.get("modality_keys", ()))
        assert actual_keys == expected_keys, (
            f"GR00T {modality} keys={actual_keys!r}, expected {expected_keys!r}"
        )
        delta_indices = tuple(int(value) for value in values.get("delta_indices", ()))
        assert delta_indices, f"GR00T {modality} delta_indices must not be empty"
        if modality in {"video", "state"}:
            assert all(value <= 0 for value in delta_indices), (
                f"GR00T {modality} observation offsets must be non-positive: "
                f"{delta_indices!r}"
            )
        return delta_indices

    @staticmethod
    def _history_size(delta_indices: tuple[int, ...]) -> int:
        return 1 - min(delta_indices)

    @staticmethod
    def _select_history(
        history: deque[np.ndarray], delta_indices: tuple[int, ...]
    ) -> np.ndarray:
        assert history, "GR00T observation history is empty"
        samples = tuple(history)
        latest_index = len(samples) - 1
        selected = [
            samples[max(0, latest_index + delta_index)]
            for delta_index in delta_indices
        ]
        return np.stack(selected, axis=1)

    @staticmethod
    def _rgb_to_numpy(head_rgb: torch.Tensor) -> np.ndarray:
        head_rgb_np = head_rgb.detach().cpu().numpy()
        if np.issubdtype(head_rgb_np.dtype, np.floating):
            if head_rgb_np.size and float(head_rgb_np.max()) <= 1.0:
                head_rgb_np = head_rgb_np * 255.0
        return np.clip(head_rgb_np, 0.0, 255.0).astype(np.uint8)

    def set_task_description(self, task_description: str | None) -> str:
        assert task_description, "A GR00T language instruction is required"
        self.task_description = task_description
        return task_description

    def get_action(
        self, env: gym.Env, observation: dict[str, Any]
    ) -> torch.Tensor:
        del env
        assert self._chunking_state is not None, "GR00T policy is closed"
        joint_pos = observation["policy"]["joint_pos"]
        head_rgb = observation["camera_obs"]["cam_head_rgb"]
        assert joint_pos.shape == (self.config.num_envs, FFW_SG2_FULL_ACTION_DIM), (
            f"Expected FFW-SG2 joint state shape "
            f"({self.config.num_envs}, {FFW_SG2_FULL_ACTION_DIM}), "
            f"got {joint_pos.shape}"
        )
        assert head_rgb.shape[0] == self.config.num_envs and head_rgb.shape[-1] == 3, (
            f"Expected batched RGB images, got {head_rgb.shape}"
        )
        self._state_history.append(
            joint_pos.detach().cpu().numpy().astype(np.float32)
        )
        self._video_history.append(self._rgb_to_numpy(head_rgb))
        return self._chunking_state.get_action(self._fetch_action_chunk)

    def _fetch_action_chunk(self) -> torch.Tensor:
        assert self._client is not None, "GR00T client is closed"
        assert self.task_description is not None, "Task description is not set"

        state_history = self._select_history(
            self._state_history, self._state_delta_indices
        )
        video_history = self._select_history(
            self._video_history, self._video_delta_indices
        )
        current_joint_pos = state_history[:, -1]
        policy_observation = {
            "language": {
                "annotation.human.task_description": [
                    [self.task_description] * self._language_horizon
                    for _ in range(self.config.num_envs)
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
        policy_action, _ = self._client.get_action(policy_observation)
        left_action = np.asarray(policy_action["arm_left"], dtype=np.float32)
        right_action = np.asarray(policy_action["arm_right"], dtype=np.float32)
        expected_side_shape = (
            self.config.num_envs,
            self._model_action_horizon,
            FFW_SG2_SIDE_ACTION_DIM,
        )
        assert left_action.shape == expected_side_shape, (
            f"Unexpected left action shape: {left_action.shape}"
        )
        assert right_action.shape == expected_side_shape, (
            f"Unexpected right action shape: {right_action.shape}"
        )

        manipulation_action = np.concatenate(
            (left_action, right_action), axis=-1
        )
        lift_head_hold = np.repeat(
            current_joint_pos[:, None, FFW_SG2_MANIP_ACTION_DIM:],
            self._model_action_horizon,
            axis=1,
        )
        full_action = np.concatenate(
            (manipulation_action, lift_head_hold), axis=-1
        )
        full_action = np.repeat(full_action, self.config.action_repeat, axis=1)
        return torch.as_tensor(
            full_action, dtype=torch.float32, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        assert self._client is not None, "GR00T client is closed"
        assert self._chunking_state is not None, "GR00T policy is closed"
        self._client.reset()
        self._video_history.clear()
        self._state_history.clear()
        self._chunking_state.reset(env_ids)

    @property
    def is_remote(self) -> bool:
        return True

    def shutdown_remote(self, kill_server: bool = False) -> None:
        client = self._client
        self._client = None
        self._chunking_state = None
        self._video_history.clear()
        self._state_history.clear()
        if client is None:
            return
        try:
            if kill_server:
                client.kill_server()
        finally:
            client.close()

    @staticmethod
    def add_args_to_parser(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        group = parser.add_argument_group("FFW-SG2 GR00T remote policy")
        group.add_argument("--remote_host", default="127.0.0.1")
        group.add_argument("--remote_port", type=int, default=5555)
        group.add_argument("--remote_timeout_ms", type=int, default=120000)
        group.add_argument("--remote_kill_on_exit", action="store_true")
        group.add_argument("--action_repeat", type=int, default=2)
        group.add_argument("--action_chunk_length", type=int, default=32)
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "Gr00tFFWSG2RemotePolicy":
        return Gr00tFFWSG2RemotePolicy(
            Gr00tFFWSG2RemotePolicyCfg(
                num_envs=args.num_envs,
                device=args.device,
                remote_host=args.remote_host,
                remote_port=args.remote_port,
                remote_timeout_ms=args.remote_timeout_ms,
                action_repeat=args.action_repeat,
                action_chunk_length=args.action_chunk_length,
            )
        )
