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

"""Static contracts for Cyclo robots, scenes, tasks, policies, and workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from cyclo_arena.core.capabilities import Capability


@dataclass(frozen=True)
class RobotPlacementSpec:
    """Define the default pose of one robot in a scene."""

    position_xyz: tuple[float, float, float]
    yaw: float


@dataclass(frozen=True)
class RobotSpec:
    """Describe one Cyclo robot and its Arena runtime adapter."""

    name: str
    description: str
    embodiments: tuple[str, ...]
    default_embodiment: str
    runtime_adapter: str
    capabilities: frozenset[Capability] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        assert self.name, "Robot name must not be empty"
        assert self.embodiments, f"Robot {self.name!r} has no embodiments"
        assert (
            self.default_embodiment in self.embodiments
        ), f"Default embodiment {self.default_embodiment!r} is not registered for robot {self.name!r}"


@dataclass(frozen=True)
class SceneSpec:
    """Describe a robot-independent Arena background and its placements."""

    name: str
    description: str
    background_name: str
    placements: Mapping[str, RobotPlacementSpec]
    registration_modules: tuple[str, ...] = ()
    additional_assets_factory: str | None = None
    add_ground_plane: bool = False
    background_position_xyz: tuple[float, float, float] | None = None
    background_rotation_wxyz: tuple[float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    constructor_arg_names: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.name, "Scene name must not be empty"
        assert self.background_name, f"Scene {self.name!r} has no background asset"
        assert self.placements, f"Scene {self.name!r} has no robot placements"
        object.__setattr__(self, "placements", MappingProxyType(dict(self.placements)))
        object.__setattr__(
            self,
            "constructor_arg_names",
            MappingProxyType(dict(self.constructor_arg_names)),
        )

    def placement_for(self, robot_name: str) -> RobotPlacementSpec:
        """Return the configured placement for one robot."""
        assert robot_name in self.placements, f"Robot {robot_name!r} has no placement in scene {self.name!r}"
        return self.placements[robot_name]


@dataclass(frozen=True)
class TaskSpec:
    """Describe a task factory and the robot capabilities it requires."""

    name: str
    description: str
    factory: str
    required_capabilities: frozenset[Capability] = field(default_factory=frozenset)
    constructor_arg_names: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constructor_arg_names",
            MappingProxyType(dict(self.constructor_arg_names)),
        )


@dataclass(frozen=True)
class PolicySpec:
    """Describe one policy exposed by Cyclo Arena."""

    name: str
    description: str
    runtime_target: str | None = None
    required_capabilities: frozenset[Capability] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ModelAdapterSpec:
    """Describe one robot-to-model interface independent of checkpoints."""

    name: str
    description: str
    robot: str
    policy: str
    embodiment: str
    model_types: tuple[str, ...]
    processor_embodiment: str
    modality_keys: Mapping[str, tuple[str, ...]]
    action_horizon: int
    server_robot_adapter: str
    action_representation: str = "ABSOLUTE"
    action_repeat: int = 2
    action_chunk_length: int = 32
    server_image: str = "cyclo-gr00t:n1.7"
    server_source_revision: str = ""
    server_workdir: str = "/workspace"
    server_embodiment_tag: str = "NEW_EMBODIMENT"
    server_device: str = "cuda"
    remote_timeout_ms: int = 120000
    startup_timeout_seconds: int = 300
    enable_cameras: bool = True

    def __post_init__(self) -> None:
        assert self.name, "Model adapter name must not be empty"
        assert self.model_types, f"Model adapter {self.name!r} has no model types"
        assert self.processor_embodiment, f"Model adapter {self.name!r} has no processor embodiment"
        assert self.action_horizon > 0, "Model action horizon must be positive"
        assert self.server_robot_adapter, f"Model adapter {self.name!r} has no server robot adapter"
        assert self.action_repeat > 0, "Action repeat must be positive"
        assert (
            0 < self.action_chunk_length <= (self.action_horizon * self.action_repeat)
        ), "Action chunk length must fit in the repeated model horizon"
        assert (
            self.action_chunk_length % self.action_repeat == 0
        ), "Action chunk length must preserve complete repeated model actions"
        assert self.server_image, f"Model adapter {self.name!r} has no server image"
        assert self.server_source_revision, f"Model adapter {self.name!r} has no GR00T source revision"
        assert self.server_workdir.startswith("/"), "GR00T server workdir must be an absolute container path"
        object.__setattr__(
            self,
            "modality_keys",
            MappingProxyType(dict(self.modality_keys)),
        )


@dataclass(frozen=True)
class WorkflowSpec:
    """Describe one user-facing Cyclo Arena workflow."""

    name: str
    description: str


@dataclass(frozen=True)
class CompositionPlan:
    """Hold validated static components for one environment build."""

    robot: RobotSpec
    scene: SceneSpec
    task: TaskSpec

    @property
    def name(self) -> str:
        """Return a deterministic name for the composed environment."""
        return f"cyclo_{self.robot.name}_{self.scene.name}_{self.task.name}"

    @property
    def placement(self) -> RobotPlacementSpec:
        """Return this composition's default robot placement."""
        return self.scene.placement_for(self.robot.name)


def import_target(target: str) -> Any:
    """Import and return a ``module:attribute`` target."""
    import importlib

    module_name, separator, attribute_name = target.partition(":")
    assert separator and module_name and attribute_name, f"Import target must use module:attribute syntax: {target!r}"
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)
