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

"""Load and validate a Cyclo Arena run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from cyclo_arena.core.capabilities import Capability
from cyclo_arena.core.registry import CycloArenaRegistry

CONFIG_SCHEMA_VERSION = 1


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    assert isinstance(value, Mapping), f"{label} must be a mapping"
    return value


def _named_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Normalize a short component name or its expanded mapping."""
    if isinstance(value, str):
        return {"name": value}
    return _mapping(value, label)


def _reject_unknown_keys(values: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(values) - allowed
    assert not unknown, f"Unknown {label} keys: {sorted(unknown)}"


def _optional_float_tuple(value: Any, length: int, label: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    assert isinstance(value, (list, tuple)) and len(value) == length, f"{label} must contain {length} numbers"
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class RobotRunConfig:
    """Select a robot embodiment and optional initial joint pose."""

    name: str
    embodiment: str | None = None
    initial_pose: str | None = None
    head_position: tuple[float, float] | None = None
    lift_position: float | None = None


@dataclass(frozen=True)
class SceneRunConfig:
    """Select a scene and optional robot placement overrides."""

    name: str
    robot_position_xyz: tuple[float, float, float] | None = None
    robot_yaw: float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRunConfig:
    """Select a task and optional language description."""

    name: str = "scene_only"
    description: str | None = None


@dataclass(frozen=True)
class PolicyRunConfig:
    """Select a policy and its remote connection settings."""

    type: str = "zero_action"
    remote_host: str | None = None
    remote_port: int | None = None
    remote_timeout_ms: int | None = None


@dataclass(frozen=True)
class ModelRunConfig:
    """Select a local checkpoint and its model adapter."""

    checkpoint: str
    adapter: str = "auto"


@dataclass(frozen=True)
class RuntimeRunConfig:
    """Configure simulation length, device, rendering, and reproducibility."""

    num_steps: int | None = None
    num_episodes: int | None = None
    num_envs: int = 1
    device: str | None = None
    seed: int | None = None
    enable_cameras: bool | None = None
    headless: bool = False


@dataclass(frozen=True)
class RunConfig:
    """Represent one portable Cyclo Arena run configuration."""

    robot: RobotRunConfig
    scene: SceneRunConfig
    task: TaskRunConfig = field(default_factory=TaskRunConfig)
    policy: PolicyRunConfig = field(default_factory=PolicyRunConfig)
    model: ModelRunConfig | None = None
    runtime: RuntimeRunConfig = field(default_factory=RuntimeRunConfig)
    schema_version: int = CONFIG_SCHEMA_VERSION
    source_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        assert (
            self.schema_version == CONFIG_SCHEMA_VERSION
        ), f"Unsupported Cyclo Arena config schema {self.schema_version}; expected {CONFIG_SCHEMA_VERSION}"
        assert not (
            self.runtime.num_steps is not None and self.runtime.num_episodes is not None
        ), "runtime.num_steps and runtime.num_episodes are mutually exclusive"
        assert self.runtime.num_envs > 0, "runtime.num_envs must be positive"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RunConfig":
        """Construct a strict run configuration from parsed data."""
        _reject_unknown_keys(
            values,
            {
                "schema_version",
                "robot",
                "scene",
                "task",
                "policy",
                "model",
                "runtime",
            },
            "top-level",
        )
        assert not (
            "model" in values and "policy" in values
        ), "model and policy are mutually exclusive; a checkpoint model selects its policy"
        robot = _named_mapping(values.get("robot"), "robot")
        scene = _named_mapping(values.get("scene"), "scene")
        task = _mapping(values.get("task", {}), "task")
        policy = _mapping(values.get("policy", {}), "policy")
        model = None
        if "model" in values:
            model = (
                {"checkpoint": values["model"]}
                if isinstance(values["model"], str)
                else _mapping(values["model"], "model")
            )
        runtime = _mapping(values.get("runtime", {}), "runtime")
        _reject_unknown_keys(
            robot,
            {
                "name",
                "embodiment",
                "initial_pose",
                "head_position",
                "lift_position",
            },
            "robot",
        )
        _reject_unknown_keys(
            scene,
            {"name", "robot_position_xyz", "robot_yaw", "options"},
            "scene",
        )
        _reject_unknown_keys(task, {"name", "description"}, "task")
        _reject_unknown_keys(
            policy,
            {"type", "remote_host", "remote_port", "remote_timeout_ms"},
            "policy",
        )
        if model is not None:
            _reject_unknown_keys(model, {"checkpoint", "adapter"}, "model")
            assert model.get("checkpoint"), "model.checkpoint is required"
        _reject_unknown_keys(
            runtime,
            {
                "num_steps",
                "num_episodes",
                "num_envs",
                "device",
                "seed",
                "enable_cameras",
                "headless",
            },
            "runtime",
        )
        assert robot.get("name"), "robot.name is required"
        assert scene.get("name"), "scene.name is required"
        scene_options = _mapping(scene.get("options", {}), "scene.options")
        return cls(
            schema_version=int(values.get("schema_version", CONFIG_SCHEMA_VERSION)),
            robot=RobotRunConfig(
                name=str(robot["name"]),
                embodiment=robot.get("embodiment"),
                initial_pose=robot.get("initial_pose"),
                head_position=_optional_float_tuple(robot.get("head_position"), 2, "robot.head_position"),
                lift_position=(float(robot["lift_position"]) if robot.get("lift_position") is not None else None),
            ),
            scene=SceneRunConfig(
                name=str(scene["name"]),
                robot_position_xyz=_optional_float_tuple(
                    scene.get("robot_position_xyz"),
                    3,
                    "scene.robot_position_xyz",
                ),
                robot_yaw=(float(scene["robot_yaw"]) if scene.get("robot_yaw") is not None else None),
                options=dict(scene_options),
            ),
            task=TaskRunConfig(
                name=str(task.get("name", "scene_only")),
                description=(str(task["description"]).strip() if task.get("description") is not None else None),
            ),
            policy=PolicyRunConfig(
                type=str(policy.get("type", "zero_action")),
                remote_host=(str(policy["remote_host"]).strip() if policy.get("remote_host") is not None else None),
                remote_port=(int(policy["remote_port"]) if policy.get("remote_port") is not None else None),
                remote_timeout_ms=(
                    int(policy["remote_timeout_ms"]) if policy.get("remote_timeout_ms") is not None else None
                ),
            ),
            model=(
                ModelRunConfig(
                    checkpoint=str(model["checkpoint"]),
                    adapter=str(model.get("adapter", "auto")),
                )
                if model is not None
                else None
            ),
            runtime=RuntimeRunConfig(
                num_steps=(int(runtime["num_steps"]) if runtime.get("num_steps") is not None else None),
                num_episodes=(int(runtime["num_episodes"]) if runtime.get("num_episodes") is not None else None),
                num_envs=int(runtime.get("num_envs", 1)),
                device=runtime.get("device"),
                seed=(int(runtime["seed"]) if runtime.get("seed") is not None else None),
                enable_cameras=(bool(runtime["enable_cameras"]) if runtime.get("enable_cameras") is not None else None),
                headless=bool(runtime.get("headless", False)),
            ),
        )

    def resolve_model(self, registry: CycloArenaRegistry):
        """Resolve this run's local checkpoint, if configured."""
        if self.model is None:
            return None
        from cyclo_arena.core.model_resolver import resolve_model

        return resolve_model(
            checkpoint=self.model.checkpoint,
            robot=self.robot.name,
            adapter_name=self.model.adapter,
            registry=registry,
            base_directory=(self.source_path.parent if self.source_path is not None else None),
        )

    def to_run_values(
        self,
        registry: CycloArenaRegistry,
        model_adapter_override: str | None = None,
    ) -> dict[str, Any]:
        """Validate registry compatibility and return CLI-shaped run values."""
        assert self.robot.name in registry.robots, f"Unknown Cyclo Arena robot: {self.robot.name!r}"
        assert self.scene.name in registry.scenes, f"Unknown Cyclo Arena scene: {self.scene.name!r}"
        assert self.task.name in registry.tasks, f"Unknown Cyclo Arena task: {self.task.name!r}"
        model = None
        model_adapter = None
        if self.model is not None:
            if model_adapter_override is None:
                model = self.resolve_model(registry)
                model_adapter = model.adapter
            else:
                assert (
                    model_adapter_override in registry.model_adapters
                ), f"Unknown Cyclo Arena model adapter: {model_adapter_override!r}"
                model_adapter = registry.model_adapters[model_adapter_override]
            assert (
                model_adapter.robot == self.robot.name
            ), f"Adapter {model_adapter.name!r} requires robot {model_adapter.robot!r}, not {self.robot.name!r}"
        prepared_remote_port = None
        if model is not None:
            from cyclo_arena.core.server_state import load_server_port

            prepared_remote_port = load_server_port(model)
        policy_name = model_adapter.policy if model_adapter is not None else self.policy.type
        assert policy_name in registry.policies, f"Unknown Cyclo Arena policy: {policy_name!r}"
        registry.compose(
            robot=self.robot.name,
            scene=self.scene.name,
            task=self.task.name,
        )
        robot_spec = registry.robots[self.robot.name]
        embodiment = (
            self.robot.embodiment
            or (model_adapter.embodiment if model_adapter is not None else None)
            or robot_spec.default_embodiment
        )
        assert (
            embodiment in robot_spec.embodiments
        ), f"Embodiment {embodiment!r} is not valid for robot {robot_spec.name!r}"
        policy = registry.policies[policy_name]
        missing_capabilities = policy.required_capabilities - robot_spec.capabilities
        assert not missing_capabilities, (
            f"Robot {robot_spec.name!r} cannot use policy {policy.name!r}; missing "
            f"{sorted(capability.value for capability in missing_capabilities)}"
        )
        enable_cameras = (
            self.runtime.enable_cameras
            if self.runtime.enable_cameras is not None
            else (model_adapter.enable_cameras if model_adapter is not None else False)
        )
        if Capability.HEAD_CAMERA in policy.required_capabilities:
            assert enable_cameras, f"Policy {policy.name!r} requires runtime.enable_cameras=true"
        values = {
            "robot": self.robot.name,
            "scene": self.scene.name,
            "task": self.task.name,
            "embodiment": embodiment,
            "policy_type": policy.runtime_target or policy.name,
            "task_description": self.task.description,
            "remote_host": "127.0.0.1" if model_adapter is not None else self.policy.remote_host,
            "remote_port": self.policy.remote_port or prepared_remote_port,
            "remote_timeout_ms": (
                model_adapter.remote_timeout_ms if model_adapter is not None else self.policy.remote_timeout_ms
            ),
            "robot_pose": self.robot.initial_pose,
            "robot_position_xyz": self.scene.robot_position_xyz,
            "robot_yaw": self.scene.robot_yaw,
            "head_position": self.robot.head_position,
            "lift_position": self.robot.lift_position,
            "num_steps": self.runtime.num_steps,
            "num_episodes": self.runtime.num_episodes,
            "num_envs": self.runtime.num_envs,
            "device": self.runtime.device,
            "seed": self.runtime.seed,
            "enable_cameras": enable_cameras,
            "headless": self.runtime.headless,
        }
        values.update(self.scene.options)
        return values


def load_run_config(path: str | Path) -> RunConfig:
    """Load one YAML run configuration."""
    import yaml

    config_path = Path(path).expanduser().resolve()
    assert config_path.is_file(), f"Cyclo Arena config does not exist: {config_path}"
    with config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    config = RunConfig.from_mapping(_mapping(values, "config"))
    return RunConfig(
        robot=config.robot,
        scene=config.scene,
        task=config.task,
        policy=config.policy,
        model=config.model,
        runtime=config.runtime,
        schema_version=config.schema_version,
        source_path=config_path,
    )
