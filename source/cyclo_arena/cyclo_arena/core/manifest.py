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

"""Resolve portable run configurations into immutable execution manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cyclo_arena.core.config import RunConfig
from cyclo_arena.core.model_resolver import ResolvedModel, resolve_checkpoint_path
from cyclo_arena.core.registry import CycloArenaRegistry
from cyclo_arena.core.workflows import resolve_workflow

MANIFEST_SCHEMA_VERSION = 1


class _FrozenList(tuple):
    """Retain a list's source type while storing immutable items."""


def _freeze(value: Any) -> Any:
    """Return a deeply immutable copy of a manifest value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _detach(value: Any) -> Any:
    """Return a detached copy with the original run-value container types."""
    if isinstance(value, Mapping):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_detach(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_detach(item) for item in value)
    return value


def _serialize(value: Any) -> Any:
    """Return a serialization-friendly copy of a manifest value."""
    if isinstance(value, Mapping):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_serialize(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class ManifestModel:
    """Identify the checkpoint and adapter selected for one manifest."""

    checkpoint: Path
    adapter: str
    model_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", self.checkpoint.expanduser().resolve())
        assert self.adapter, "Manifest model adapter must not be empty"

    def to_resolved_model(self, registry: CycloArenaRegistry) -> ResolvedModel:
        """Recreate the validated runtime model without reading checkpoint metadata."""
        assert self.adapter in registry.model_adapters, f"Unknown Cyclo Arena model adapter: {self.adapter!r}"
        assert self.model_type is not None, "Resolved manifest model type is unavailable"
        return ResolvedModel(
            checkpoint=self.checkpoint,
            adapter=registry.model_adapters[self.adapter],
            model_type=self.model_type,
        )


@dataclass(frozen=True)
class ResolvedManifest:
    """Hold one validated, immutable Cyclo Arena execution plan."""

    workflow: str
    run_values: Mapping[str, Any]
    model: ManifestModel | None = None
    profile: str | None = None
    source_path: Path | None = None
    schema_version: int = MANIFEST_SCHEMA_VERSION
    config_schema_version: int = 1

    def __post_init__(self) -> None:
        assert (
            self.schema_version == MANIFEST_SCHEMA_VERSION
        ), f"Unsupported Cyclo Arena manifest schema {self.schema_version}; expected {MANIFEST_SCHEMA_VERSION}"
        workflow = resolve_workflow(self.workflow)
        assert workflow.is_supported, workflow.readiness_detail
        object.__setattr__(self, "workflow", workflow.name)
        object.__setattr__(self, "run_values", _freeze(self.run_values))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", self.source_path.expanduser().resolve())

    @classmethod
    def from_run_config(
        cls,
        config: RunConfig,
        registry: CycloArenaRegistry,
        *,
        workflow: str = "infer",
        profile: str | None = None,
        model_adapter_override: str | None = None,
    ) -> "ResolvedManifest":
        """Resolve an existing version-one run configuration without changing its behavior."""
        workflow_spec = resolve_workflow(workflow)
        assert workflow_spec.is_supported, workflow_spec.readiness_detail
        resolved_model = None
        if config.model is not None and model_adapter_override is None:
            resolved_model = config.resolve_model(registry)
        run_values = config.to_run_values(
            registry,
            model_adapter_override=model_adapter_override,
            resolved_model=resolved_model,
        )
        model = None
        if config.model is not None:
            base_directory = config.source_path.parent if config.source_path is not None else None
            if model_adapter_override is None:
                assert resolved_model is not None
                model = ManifestModel(
                    checkpoint=resolved_model.checkpoint,
                    adapter=resolved_model.adapter.name,
                    model_type=resolved_model.model_type,
                )
            else:
                assert (
                    model_adapter_override in registry.model_adapters
                ), f"Unknown Cyclo Arena model adapter: {model_adapter_override!r}"
                model = ManifestModel(
                    checkpoint=resolve_checkpoint_path(
                        config.model.checkpoint,
                        base_directory=base_directory,
                    ),
                    adapter=model_adapter_override,
                )
        return cls(
            workflow=workflow_spec.name,
            run_values=run_values,
            model=model,
            profile=profile,
            source_path=config.source_path,
            config_schema_version=config.schema_version,
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ResolvedManifest":
        """Load a resolved mapping without re-reading its source configuration."""
        assert isinstance(values, Mapping), "Cyclo Arena manifest must contain a mapping"
        run_values = values.get("run")
        assert isinstance(run_values, Mapping), "Cyclo Arena manifest is missing its resolved run mapping"
        model_values = values.get("model")
        model = None
        if model_values is not None:
            assert isinstance(model_values, Mapping), "Cyclo Arena manifest model must contain a mapping"
            model = ManifestModel(
                checkpoint=Path(str(model_values["checkpoint"])),
                adapter=str(model_values["adapter"]),
                model_type=(str(model_values["model_type"]) if model_values.get("model_type") is not None else None),
            )
        return cls(
            schema_version=int(values.get("schema_version", MANIFEST_SCHEMA_VERSION)),
            config_schema_version=int(values.get("config_schema_version", 1)),
            workflow=str(values["workflow"]),
            profile=(str(values["profile"]) if values.get("profile") is not None else None),
            source_path=(Path(str(values["source_path"])) if values.get("source_path") is not None else None),
            model=model,
            run_values=run_values,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ResolvedManifest":
        """Load a resolved manifest from JSON."""
        manifest_path = Path(path).expanduser().resolve()
        assert manifest_path.is_file(), f"Cyclo Arena manifest does not exist: {manifest_path}"
        with manifest_path.open(encoding="utf-8") as manifest_file:
            values = json.load(manifest_file)
        return cls.from_mapping(values)

    def to_run_values(self) -> dict[str, Any]:
        """Return detached CLI-shaped values for the existing Arena runner."""
        return _detach(self.run_values)

    def with_run_overrides(self, **overrides: Any) -> "ResolvedManifest":
        """Return a new manifest with explicit runtime or CLI value overrides."""
        values = self.to_run_values()
        values.update(overrides)
        return replace(self, run_values=values)

    def to_mapping(self) -> dict[str, Any]:
        """Return a serialization-friendly representation for process boundaries."""
        model = None
        if self.model is not None:
            model = {
                "checkpoint": str(self.model.checkpoint),
                "adapter": self.model.adapter,
                "model_type": self.model.model_type,
            }
        return {
            "schema_version": self.schema_version,
            "config_schema_version": self.config_schema_version,
            "workflow": self.workflow,
            "profile": self.profile,
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "model": model,
            "run": _serialize(self.run_values),
        }

    def to_json(self) -> str:
        """Return a stable JSON representation for process boundaries."""
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"

    @property
    def fingerprint(self) -> str:
        """Return a stable short identity for this resolved execution plan."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def write(self, path: str | Path) -> Path:
        """Write this manifest atomically and return its absolute path."""
        manifest_path = Path(path).expanduser().resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
        temporary_path.write_text(self.to_json(), encoding="utf-8")
        temporary_path.replace(manifest_path)
        return manifest_path
