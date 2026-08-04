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

"""GR00T model wrapper for Arena's native remote-policy server."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from cyclo_arena.policies.adapters.base import Gr00tRobotAdapter
from isaaclab_arena.remote_policy.action_protocol import ChunkingActionProtocol
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy


@dataclass
class CycloGr00tServerPolicyCfg:
    """Configure one GR00T runtime and its robot data adapter."""

    model_path: str
    robot_adapter: str
    embodiment_tag: str = "NEW_EMBODIMENT"
    device: str = "cuda"
    action_repeat: int = 2
    action_chunk_length: int = 32


def _load_class(target: str) -> type:
    """Load one class from a ``module:Class`` or dotted path."""
    if ":" in target:
        module_name, class_name = target.rsplit(":", 1)
    else:
        module_name, class_name = target.rsplit(".", 1)
    return getattr(import_module(module_name), class_name)


def _resolve_embodiment_tag(tag: str):
    """Resolve an N1.7 embodiment enum value."""
    from gr00t.data.embodiment_tags import EmbodimentTag

    return EmbodimentTag.resolve(tag)


class CycloGr00tServerSidePolicy(ServerSidePolicy):
    """Expose NVIDIA GR00T through Arena's generic server-side contract."""

    config_class = CycloGr00tServerPolicyCfg

    def __init__(self, config: CycloGr00tServerPolicyCfg) -> None:
        super().__init__(config)
        assert config.action_repeat > 0, "action_repeat must be positive"
        assert config.action_chunk_length > 0, "action_chunk_length must be positive"
        assert (
            config.action_chunk_length % config.action_repeat == 0
        ), "action_chunk_length must preserve complete repeated model actions"

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self._policy = Gr00tPolicy(
            embodiment_tag=_resolve_embodiment_tag(config.embodiment_tag),
            model_path=config.model_path,
            device=config.device,
            strict=True,
        )
        adapter_class = _load_class(config.robot_adapter)
        assert issubclass(adapter_class, Gr00tRobotAdapter), f"{config.robot_adapter!r} is not a Gr00tRobotAdapter"
        self._robot_adapter: Gr00tRobotAdapter = adapter_class(
            self._policy.get_modality_config(),
            action_repeat=config.action_repeat,
        )
        repeated_horizon = self._robot_adapter.model_action_horizon * config.action_repeat
        assert (
            config.action_chunk_length <= repeated_horizon
        ), f"action_chunk_length={config.action_chunk_length} exceeds the repeated model horizon={repeated_horizon}"

    def _build_protocol(self) -> ChunkingActionProtocol:
        return ChunkingActionProtocol(
            action_dim=self._robot_adapter.action_dim,
            observation_keys=self._robot_adapter.observation_keys,
            action_chunk_length=self.config.action_chunk_length,
            action_horizon=(self._robot_adapter.model_action_horizon * self.config.action_repeat),
        )

    def set_task_description(self, task_description: str | None) -> dict[str, Any]:
        assert task_description, "A GR00T language instruction is required"
        self._task_description = task_description
        return {"status": "ok"}

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert self._task_description is not None, "Task description is not set"
        policy_observation = self._robot_adapter.build_policy_observation(
            observation,
            self._task_description,
        )
        policy_action, info = self._policy.get_action(
            policy_observation,
            options=options,
        )
        action_chunk = self._robot_adapter.build_action_chunk(policy_action)
        return {"action": action_chunk}, info

    def reset(
        self,
        env_ids: list[int] | None = None,
        reset_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del env_ids
        self._robot_adapter.reset()
        return self._policy.reset(options=reset_options)

    @staticmethod
    def add_args_to_parser(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Cyclo GR00T server policy")
        group.add_argument("--model_path", required=True)
        group.add_argument("--robot_adapter", required=True)
        group.add_argument("--embodiment_tag", default="NEW_EMBODIMENT")
        group.add_argument("--device", default="cuda")
        group.add_argument("--action_repeat", type=int, default=2)
        group.add_argument("--action_chunk_length", type=int, default=32)
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "CycloGr00tServerSidePolicy":
        return CycloGr00tServerSidePolicy(
            CycloGr00tServerPolicyCfg(
                model_path=args.model_path,
                robot_adapter=args.robot_adapter,
                embodiment_tag=args.embodiment_tag,
                device=args.device,
                action_repeat=args.action_repeat,
                action_chunk_length=args.action_chunk_length,
            )
        )
