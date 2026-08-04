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

"""Discover and load named Cyclo Arena profiles without exposing file paths."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from cyclo_arena.core.config import RunConfig, load_run_config
from cyclo_arena.core.manifest import ResolvedManifest
from cyclo_arena.core.registry import CycloArenaRegistry

DEFAULT_PROFILE_ID = "ffw_sg2_showroom_gr00t"
PROFILE_ROOT_ENVIRONMENT = "CYCLO_ARENA_PROFILE_ROOT"


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def default_profile_roots() -> tuple[Path, ...]:
    """Return profile roots for environment, source-checkout, and installed layouts."""
    module_path = Path(__file__).resolve()
    candidates: list[Path] = []
    configured_roots = os.environ.get(PROFILE_ROOT_ENVIRONMENT)
    if configured_roots:
        candidates.extend(Path(value) for value in configured_roots.split(os.pathsep) if value)
    repository_root = os.environ.get("CYCLOLAB_PATH")
    if repository_root:
        candidates.append(Path(repository_root) / "source" / "cyclo_arena" / "configs" / "profiles")
    candidates.extend((
        module_path.parents[2] / "configs" / "profiles",
        module_path.parents[1] / "profiles",
        Path(sys.prefix) / "share" / "cyclo_arena" / "profiles",
    ))
    return _unique_paths(candidates)


def _profile_parts(profile_id: str) -> tuple[str, ...]:
    normalized = profile_id.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    assert normalized and not path.is_absolute(), "Profile ID must be a non-empty relative name"
    assert path.suffix not in {".yaml", ".yml"}, "Profile ID must not include a YAML extension"
    assert all(part not in {"", ".", ".."} for part in path.parts), f"Invalid profile ID: {profile_id!r}"
    return path.parts


@dataclass(frozen=True)
class NamedProfile:
    """Associate a friendly profile ID with its portable run configuration."""

    name: str
    path: Path

    def load(self) -> RunConfig:
        """Load the profile as a version-one run configuration."""
        return load_run_config(self.path)

    def resolve(
        self,
        registry: CycloArenaRegistry,
        *,
        workflow: str = "infer",
        model_adapter_override: str | None = None,
    ) -> ResolvedManifest:
        """Resolve this profile into an immutable execution manifest."""
        return ResolvedManifest.from_run_config(
            self.load(),
            registry,
            workflow=workflow,
            profile=self.name,
            model_adapter_override=model_adapter_override,
        )


class ProfileStore:
    """Resolve stable profile IDs from one or more ordered search roots."""

    def __init__(self, roots: Iterable[str | Path] | None = None) -> None:
        selected_roots = default_profile_roots() if roots is None else (Path(root) for root in roots)
        self._roots = _unique_paths(selected_roots)
        assert self._roots, "At least one Cyclo Arena profile root is required"

    @property
    def roots(self) -> tuple[Path, ...]:
        """Return profile search roots in precedence order."""
        return self._roots

    def names(self) -> tuple[str, ...]:
        """Return all discoverable profile IDs in deterministic order."""
        names: set[str] = set()
        for root in self._roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*.yaml"):
                names.add(path.relative_to(root).with_suffix("").as_posix())
        return tuple(sorted(names))

    def get(self, profile_id: str) -> NamedProfile:
        """Return a named profile using search-root precedence."""
        parts = _profile_parts(profile_id)
        for root in self._roots:
            candidate = root.joinpath(*parts).with_suffix(".yaml")
            if candidate.is_file():
                return NamedProfile(name=PurePosixPath(*parts).as_posix(), path=candidate.resolve())
        searched = ", ".join(str(root) for root in self._roots)
        raise AssertionError(f"Unknown Cyclo Arena profile {profile_id!r}; searched: {searched}")

    def load(self, profile_id: str) -> RunConfig:
        """Load a profile by friendly ID."""
        return self.get(profile_id).load()

    def resolve(
        self,
        profile_id: str,
        registry: CycloArenaRegistry,
        *,
        workflow: str = "infer",
        model_adapter_override: str | None = None,
    ) -> ResolvedManifest:
        """Resolve a friendly profile ID into an immutable execution manifest."""
        return self.get(profile_id).resolve(
            registry,
            workflow=workflow,
            model_adapter_override=model_adapter_override,
        )


def load_profile(profile_id: str = DEFAULT_PROFILE_ID) -> RunConfig:
    """Load one profile from the default store."""
    return ProfileStore().load(profile_id)
