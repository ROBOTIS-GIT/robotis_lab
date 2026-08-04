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

"""Typed registry for simulator-independent Cyclo Arena metadata."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, TypeVar

from cyclo_arena.core.contracts import (
    CompositionPlan,
    ModelAdapterSpec,
    PolicySpec,
    RobotSpec,
    SceneSpec,
    TaskSpec,
)

if TYPE_CHECKING:
    from cyclo_arena.core.workflows import WorkflowSpec

SpecT = TypeVar("SpecT")


class CycloArenaRegistry:
    """Register and validate Cyclo Arena composition metadata."""

    def __init__(self) -> None:
        self._robots: dict[str, RobotSpec] = {}
        self._scenes: dict[str, SceneSpec] = {}
        self._tasks: dict[str, TaskSpec] = {}
        self._policies: dict[str, PolicySpec] = {}
        self._model_adapters: dict[str, ModelAdapterSpec] = {}

    @staticmethod
    def _register(collection: dict[str, SpecT], name: str, spec: SpecT) -> None:
        assert name not in collection, f"Duplicate Cyclo Arena registration: {name!r}"
        collection[name] = spec

    def register_robot(self, spec: RobotSpec) -> None:
        """Register one robot."""
        self._register(self._robots, spec.name, spec)

    def register_scene(self, spec: SceneSpec) -> None:
        """Register one scene."""
        self._register(self._scenes, spec.name, spec)

    def register_task(self, spec: TaskSpec) -> None:
        """Register one task."""
        self._register(self._tasks, spec.name, spec)

    def register_policy(self, spec: PolicySpec) -> None:
        """Register one policy."""
        self._register(self._policies, spec.name, spec)

    def register_model_adapter(self, spec: ModelAdapterSpec) -> None:
        """Register and validate one robot-to-model adapter."""
        assert spec.robot in self._robots, f"Unknown robot: {spec.robot!r}"
        assert spec.policy in self._policies, f"Unknown policy: {spec.policy!r}"
        robot = self._robots[spec.robot]
        assert (
            spec.embodiment in robot.embodiments
        ), f"Adapter {spec.name!r} embodiment {spec.embodiment!r} is not registered for robot {robot.name!r}"
        policy = self._policies[spec.policy]
        missing_capabilities = policy.required_capabilities - robot.capabilities
        assert not missing_capabilities, (
            f"Robot {robot.name!r} cannot use adapter {spec.name!r}; missing "
            f"{sorted(capability.value for capability in missing_capabilities)}"
        )
        self._register(self._model_adapters, spec.name, spec)

    @property
    def robots(self) -> Mapping[str, RobotSpec]:
        """Return registered robots as a read-only mapping."""
        return MappingProxyType(self._robots)

    @property
    def scenes(self) -> Mapping[str, SceneSpec]:
        """Return registered scenes as a read-only mapping."""
        return MappingProxyType(self._scenes)

    @property
    def tasks(self) -> Mapping[str, TaskSpec]:
        """Return registered tasks as a read-only mapping."""
        return MappingProxyType(self._tasks)

    @property
    def policies(self) -> Mapping[str, PolicySpec]:
        """Return registered policies as a read-only mapping."""
        return MappingProxyType(self._policies)

    @property
    def model_adapters(self) -> Mapping[str, ModelAdapterSpec]:
        """Return registered robot-to-model adapters as a read-only mapping."""
        return MappingProxyType(self._model_adapters)

    @property
    def workflows(self) -> Mapping[str, WorkflowSpec]:
        """Return the shared workflow registry for backward compatibility."""
        from cyclo_arena.core.workflows import WORKFLOWS

        return WORKFLOWS

    def compose(self, robot: str, scene: str, task: str) -> CompositionPlan:
        """Validate and compose one robot, scene, and task selection."""
        assert robot in self._robots, f"Unknown robot: {robot!r}"
        assert scene in self._scenes, f"Unknown scene: {scene!r}"
        assert task in self._tasks, f"Unknown task: {task!r}"
        robot_spec = self._robots[robot]
        scene_spec = self._scenes[scene]
        task_spec = self._tasks[task]
        scene_spec.placement_for(robot)
        missing_capabilities = task_spec.required_capabilities - robot_spec.capabilities
        assert not missing_capabilities, (
            f"Robot {robot!r} cannot run task {task!r}; missing "
            f"{sorted(capability.value for capability in missing_capabilities)}"
        )
        return CompositionPlan(
            robot=robot_spec,
            scene=scene_spec,
            task=task_spec,
        )
