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

"""Discover checkpoints and match their metadata to model adapters."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cyclo_arena.core.contracts import ModelAdapterSpec
from cyclo_arena.core.registry import CycloArenaRegistry

MODEL_ROOT_ENVIRONMENT = "CYCLO_ARENA_MODEL_ROOT"
DEFAULT_MODEL_ROOT = str(
    Path(__file__).resolve().parents[4] / "docker" / "workspace" / "model"
)


@dataclass(frozen=True)
class ResolvedModel:
    """Hold one validated local checkpoint and its runtime adapter."""

    checkpoint: Path
    adapter: ModelAdapterSpec
    model_type: str

    @property
    def name(self) -> str:
        """Return the checkpoint directory name for display."""
        return self.checkpoint.name


@dataclass(frozen=True)
class DiscoveredModel:
    """Describe one checkpoint found below the configured model root."""

    checkpoint: Path
    model_type: str
    compatible_adapters: tuple[str, ...]


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    assert path.is_file(), f"GR00T checkpoint is missing {label}: {path}"
    with path.open(encoding="utf-8") as json_file:
        values = json.load(json_file)
    assert isinstance(values, Mapping), f"{label} must contain a JSON object"
    return values


def resolve_checkpoint_path(
    checkpoint: str | Path,
    base_directory: Path | None = None,
) -> Path:
    """Resolve an expanded checkpoint path relative to its run config."""
    expanded = Path(os.path.expandvars(str(checkpoint))).expanduser()
    if not expanded.is_absolute():
        expanded = (base_directory or Path.cwd()) / expanded
    return expanded.resolve()


def _checkpoint_metadata(
    checkpoint: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    assert checkpoint.is_dir(), f"GR00T checkpoint directory does not exist: {checkpoint}"
    assert any(checkpoint.glob("*.safetensors")), (
        f"GR00T checkpoint contains no safetensors weights: {checkpoint}"
    )
    processor_directory = (
        checkpoint
        if (checkpoint / "processor_config.json").is_file()
        else checkpoint / "processor"
    )
    assert (processor_directory / "processor_config.json").is_file(), (
        f"GR00T checkpoint is missing processor_config.json: {checkpoint}"
    )
    assert (processor_directory / "statistics.json").is_file(), (
        f"GR00T checkpoint is missing statistics.json: {processor_directory}"
    )
    model_config = _read_json(checkpoint / "config.json", "config.json")
    processor_config = _read_json(
        processor_directory / "processor_config.json",
        "processor_config.json",
    )
    return model_config, processor_config


def compatibility_errors(
    model_config: Mapping[str, Any],
    processor_config: Mapping[str, Any],
    adapter: ModelAdapterSpec,
) -> tuple[str, ...]:
    """Return metadata differences that make a checkpoint incompatible."""
    errors: list[str] = []
    model_type = str(model_config.get("model_type", ""))
    if model_type not in adapter.model_types:
        errors.append(
            f"model_type={model_type!r}, expected one of {adapter.model_types!r}"
        )
    processor_kwargs = processor_config.get("processor_kwargs", {})
    modality_configs = (
        processor_kwargs.get("modality_configs", {})
        if isinstance(processor_kwargs, Mapping)
        else {}
    )
    embodiment = modality_configs.get(adapter.processor_embodiment, {})
    if not isinstance(embodiment, Mapping):
        errors.append(
            f"processor embodiment {adapter.processor_embodiment!r} is missing"
        )
        return tuple(errors)
    for modality_name, expected_keys in adapter.modality_keys.items():
        modality = embodiment.get(modality_name, {})
        actual_keys = (
            tuple(modality.get("modality_keys", ()))
            if isinstance(modality, Mapping)
            else ()
        )
        if actual_keys != expected_keys:
            errors.append(
                f"{modality_name} keys={actual_keys!r}, expected {expected_keys!r}"
            )
    action = embodiment.get("action", {})
    if isinstance(action, Mapping):
        action_horizon = len(action.get("delta_indices", ()))
        if action_horizon != adapter.action_horizon:
            errors.append(
                f"action horizon={action_horizon}, expected {adapter.action_horizon}"
            )
        action_configs = action.get("action_configs", ()) or ()
        representations = tuple(
            config.get("rep")
            for config in action_configs
            if isinstance(config, Mapping)
        )
        expected_representations = (adapter.action_representation,) * len(
            adapter.modality_keys["action"]
        )
        if representations != expected_representations:
            errors.append(
                f"action representations={representations!r}, "
                f"expected {expected_representations!r}"
            )
    return tuple(errors)


def resolve_model(
    checkpoint: str | Path,
    robot: str,
    adapter_name: str,
    registry: CycloArenaRegistry,
    base_directory: Path | None = None,
) -> ResolvedModel:
    """Resolve and validate a checkpoint against an explicit or automatic adapter."""
    checkpoint_path = resolve_checkpoint_path(checkpoint, base_directory)
    model_config, processor_config = _checkpoint_metadata(checkpoint_path)
    if adapter_name == "auto":
        candidates = tuple(
            adapter
            for adapter in registry.model_adapters.values()
            if adapter.robot == robot
        )
    else:
        assert adapter_name in registry.model_adapters, (
            f"Unknown Cyclo Arena model adapter: {adapter_name!r}"
        )
        candidates = (registry.model_adapters[adapter_name],)
    assert candidates, f"No model adapters are registered for robot {robot!r}"

    incompatibilities: dict[str, tuple[str, ...]] = {}
    matches: list[ModelAdapterSpec] = []
    for adapter in candidates:
        assert adapter.robot == robot, (
            f"Adapter {adapter.name!r} requires robot {adapter.robot!r}, not {robot!r}"
        )
        errors = compatibility_errors(model_config, processor_config, adapter)
        if errors:
            incompatibilities[adapter.name] = errors
        else:
            matches.append(adapter)
    assert len(matches) == 1, (
        f"Expected one compatible model adapter for {checkpoint_path}; "
        f"found {[adapter.name for adapter in matches]}. "
        f"Compatibility errors: {incompatibilities}"
    )
    return ResolvedModel(
        checkpoint=checkpoint_path,
        adapter=matches[0],
        model_type=str(model_config["model_type"]),
    )


def model_search_root() -> Path:
    """Return the configured Cyclo Arena model workspace."""
    root = Path(
        os.path.expandvars(
            os.environ.get(MODEL_ROOT_ENVIRONMENT, DEFAULT_MODEL_ROOT)
        )
    ).expanduser()
    checkpoint_root = root / "checkpoints"
    return checkpoint_root.resolve() if checkpoint_root.is_dir() else root.resolve()


def discover_models(
    registry: CycloArenaRegistry,
    root: Path | None = None,
) -> tuple[DiscoveredModel, ...]:
    """Scan local checkpoint directories and report compatible adapters."""
    search_root = (root or model_search_root()).expanduser().resolve()
    if not search_root.is_dir():
        return ()
    discovered: list[DiscoveredModel] = []
    for config_path in sorted(search_root.rglob("config.json")):
        checkpoint = config_path.parent
        if not (
            (checkpoint / "processor_config.json").is_file()
            or (checkpoint / "processor" / "processor_config.json").is_file()
        ):
            continue
        if not any(checkpoint.glob("*.safetensors")):
            continue
        try:
            model_config, processor_config = _checkpoint_metadata(checkpoint)
        except (AssertionError, json.JSONDecodeError):
            continue
        compatible = tuple(
            adapter.name
            for adapter in registry.model_adapters.values()
            if not compatibility_errors(model_config, processor_config, adapter)
        )
        discovered.append(
            DiscoveredModel(
                checkpoint=checkpoint,
                model_type=str(model_config.get("model_type", "unknown")),
                compatible_adapters=compatible,
            )
        )
    return tuple(discovered)
