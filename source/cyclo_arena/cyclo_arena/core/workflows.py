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

"""Simulator-independent workflow definitions for Cyclo Arena."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class WorkflowKind(str, Enum):
    """Describe how a workflow target is launched."""

    MODULE = "module"
    SCRIPT = "script"
    SHELL = "shell"


class WorkflowReadiness(str, Enum):
    """Describe the integration readiness of a workflow."""

    READY = "ready"
    REQUIRES_SETUP = "requires_setup"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class WorkflowRequirement:
    """Describe one machine-readable workflow prerequisite."""

    name: str
    description: str

    def __post_init__(self) -> None:
        assert self.name and not self.name.isspace(), "Workflow requirement name must not be empty"
        assert self.description and not self.description.isspace(), "Workflow requirement description must not be empty"


@dataclass(frozen=True)
class WorkflowSpec:
    """Describe one user-facing workflow and its upstream launch target."""

    name: str
    description: str
    kind: WorkflowKind
    upstream_target: str | None
    launcher_target: str | None = None
    """Optional Cyclo-owned adapter target used to launch the upstream workflow."""
    aliases: tuple[str, ...] = ()
    default_args: tuple[str, ...] = ()
    requirements: tuple[WorkflowRequirement, ...] = ()
    readiness: WorkflowReadiness = WorkflowReadiness.READY
    readiness_detail: str = ""

    def __post_init__(self) -> None:
        assert self.name and not self.name.isspace(), "Workflow name must not be empty"
        assert self.description and not self.description.isspace(), "Workflow description must not be empty"
        assert isinstance(self.kind, WorkflowKind), "Workflow kind must be a WorkflowKind"
        assert isinstance(self.readiness, WorkflowReadiness), "Workflow readiness must be a WorkflowReadiness"
        assert len(set(self.aliases)) == len(self.aliases), f"Workflow {self.name!r} contains duplicate aliases"
        assert self.name not in self.aliases, f"Workflow {self.name!r} cannot alias itself"
        assert all(alias and not alias.isspace() for alias in self.aliases), "Workflow aliases must not be empty"
        requirement_names = tuple(requirement.name for requirement in self.requirements)
        assert len(set(requirement_names)) == len(
            requirement_names
        ), f"Workflow {self.name!r} contains duplicate requirements"
        if self.readiness is WorkflowReadiness.UNSUPPORTED:
            assert self.upstream_target is None, "Unsupported workflows must not expose a launch target"
            assert self.launcher_target is None, "Unsupported workflows must not expose a launcher target"
            assert self.readiness_detail, "Unsupported workflows must explain why they are unavailable"
        else:
            assert self.upstream_target, f"Workflow {self.name!r} has no upstream target"
        if self.launcher_target is not None:
            assert (
                self.launcher_target and not self.launcher_target.isspace()
            ), "Workflow launcher target must not be empty"
        if self.readiness is WorkflowReadiness.REQUIRES_SETUP:
            assert self.readiness_detail, "Workflows requiring setup must describe the missing setup"

    @property
    def is_supported(self) -> bool:
        """Return whether the workflow has an implemented launch path."""
        return self.readiness is not WorkflowReadiness.UNSUPPORTED

    @property
    def is_ready(self) -> bool:
        """Return whether the workflow needs no integration-specific setup."""
        return self.readiness is WorkflowReadiness.READY

    @property
    def executable_target(self) -> str:
        """Return the Cyclo adapter target or the original upstream target."""
        target = self.launcher_target or self.upstream_target
        assert target is not None, f"Workflow {self.name!r} has no executable target"
        return target


class WorkflowRegistry(Mapping[str, WorkflowSpec]):
    """Provide immutable canonical and alias lookup for workflow specs."""

    def __init__(self, specs: Sequence[WorkflowSpec]):
        canonical: dict[str, WorkflowSpec] = {}
        aliases: dict[str, str] = {}
        for spec in specs:
            assert spec.name not in canonical, f"Workflow {spec.name!r} is already registered"
            assert spec.name not in aliases, f"Workflow name {spec.name!r} collides with an alias"
            canonical[spec.name] = spec
            for alias in spec.aliases:
                assert alias not in canonical, f"Workflow alias {alias!r} collides with a workflow name"
                assert alias not in aliases, f"Workflow alias {alias!r} is already registered"
                aliases[alias] = spec.name
        self._canonical = MappingProxyType(canonical)
        self._aliases = MappingProxyType(aliases)

    def __getitem__(self, name: str) -> WorkflowSpec:
        return self._canonical[self.canonical_name(name)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._canonical)

    def __len__(self) -> int:
        return len(self._canonical)

    @property
    def aliases(self) -> Mapping[str, str]:
        """Return alias-to-canonical-name mappings."""
        return self._aliases

    @property
    def command_names(self) -> tuple[str, ...]:
        """Return every canonical name and accepted command alias."""
        return (*self._canonical, *self._aliases)

    def canonical_name(self, name: str) -> str:
        """Resolve a canonical workflow name from a name or alias."""
        if name in self._canonical:
            return name
        if name in self._aliases:
            return self._aliases[name]
        raise KeyError(f"Unknown workflow: {name!r}")

    def resolve(self, name: str) -> WorkflowSpec:
        """Resolve a workflow specification from a name or alias."""
        return self[name]


_COMPOSED_ENVIRONMENT = WorkflowRequirement(
    "composed_environment",
    "A registered robot, scene, and task composition.",
)
_POLICY_RUNTIME = WorkflowRequirement(
    "policy_runtime",
    "A policy implementation and any model runtime it requires.",
)
_DATASET = WorkflowRequirement(
    "arena_dataset",
    "An Arena-compatible HDF5 demonstration dataset.",
)
_TELEOP_RETARGETER = WorkflowRequirement(
    "teleop_retargeter",
    "A teleoperation device and robot-specific retargeter.",
)
_SUCCESS_PREDICATE = WorkflowRequirement(
    "success_predicate",
    "Task success predicates suitable for demonstration recording.",
)
_MIMIC_TASK = WorkflowRequirement(
    "mimic_task",
    "Robot-specific Mimic subtasks and environment configuration.",
)
_ISAACLAB_ARENA_SUBMODULE = WorkflowRequirement(
    "isaaclab_arena_submodule",
    "The initialized upstream IsaacLab-Arena submodule.",
)


WORKFLOWS = WorkflowRegistry((
    WorkflowSpec(
        name="infer",
        description="Run one policy in one Arena environment.",
        kind=WorkflowKind.MODULE,
        upstream_target="isaaclab_arena.evaluation.policy_runner",
        launcher_target="cyclo_arena.compat.policy_runner",
        aliases=("run", "inference", "policy"),
        requirements=(_COMPOSED_ENVIRONMENT, _POLICY_RUNTIME),
    ),
    WorkflowSpec(
        name="evaluate",
        description="Run a sequential JSON evaluation job set.",
        kind=WorkflowKind.MODULE,
        upstream_target="isaaclab_arena.evaluation.eval_runner",
        aliases=("eval",),
        requirements=(_COMPOSED_ENVIRONMENT, _POLICY_RUNTIME),
        readiness=WorkflowReadiness.REQUIRES_SETUP,
        readiness_detail="Each external Cyclo environment must provide an Arena-compatible evaluation job.",
    ),
    WorkflowSpec(
        name="teleop",
        description="Teleoperate an Arena environment.",
        kind=WorkflowKind.MODULE,
        upstream_target="isaaclab_arena.scripts.imitation_learning.teleop",
        requirements=(_COMPOSED_ENVIRONMENT, _TELEOP_RETARGETER),
        readiness=WorkflowReadiness.REQUIRES_SETUP,
        readiness_detail="The selected robot must provide a compatible teleoperation retargeter.",
    ),
    WorkflowSpec(
        name="record",
        description="Record teleoperated demonstrations to HDF5.",
        kind=WorkflowKind.MODULE,
        upstream_target="isaaclab_arena.scripts.imitation_learning.record_demos",
        requirements=(_COMPOSED_ENVIRONMENT, _TELEOP_RETARGETER, _SUCCESS_PREDICATE),
        readiness=WorkflowReadiness.REQUIRES_SETUP,
        readiness_detail="The selected robot and task must provide teleoperation and success semantics.",
    ),
    WorkflowSpec(
        name="replay",
        description="Replay an Arena demonstration dataset.",
        kind=WorkflowKind.MODULE,
        upstream_target="isaaclab_arena.scripts.imitation_learning.replay_demos",
        requirements=(_COMPOSED_ENVIRONMENT, _DATASET),
    ),
    WorkflowSpec(
        name="mimic-annotate",
        description="Annotate demonstrations for Isaac Lab Mimic.",
        kind=WorkflowKind.MODULE,
        upstream_target="isaaclab_arena.scripts.imitation_learning.annotate_demos",
        aliases=("annotate", "mimic.annotate"),
        requirements=(_DATASET, _MIMIC_TASK),
        readiness=WorkflowReadiness.REQUIRES_SETUP,
        readiness_detail="The selected robot and task must provide Mimic subtask definitions.",
    ),
    WorkflowSpec(
        name="mimic-generate",
        description="Generate Mimic demonstrations from annotated source data.",
        kind=WorkflowKind.MODULE,
        upstream_target="isaaclab_arena.scripts.imitation_learning.generate_dataset",
        aliases=("generate", "mimic.generate"),
        requirements=(_DATASET, _MIMIC_TASK),
        readiness=WorkflowReadiness.REQUIRES_SETUP,
        readiness_detail="The selected robot and task must provide a Mimic-compatible environment.",
    ),
    WorkflowSpec(
        name="serve",
        description="Run an Arena remote policy server.",
        kind=WorkflowKind.MODULE,
        upstream_target="isaaclab_arena.remote_policy.remote_policy_server_runner",
        requirements=(_POLICY_RUNTIME,),
    ),
    WorkflowSpec(
        name="rl-train",
        description="Train an Arena RL environment with Isaac Lab RSL-RL.",
        kind=WorkflowKind.SCRIPT,
        upstream_target="third_party/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py",
        requirements=(_COMPOSED_ENVIRONMENT,),
        readiness=WorkflowReadiness.REQUIRES_SETUP,
        readiness_detail="The selected task must provide an RSL-RL environment and training configuration.",
    ),
    WorkflowSpec(
        name="gr00t-server",
        description="Start Arena's isolated GR00T server container.",
        kind=WorkflowKind.SHELL,
        upstream_target="third_party/IsaacLab-Arena/docker/run_gr00t_server.sh",
        requirements=(_ISAACLAB_ARENA_SUBMODULE, _POLICY_RUNTIME),
    ),
    WorkflowSpec(
        name="test",
        description="Run the Arena pytest suite or a selected subset.",
        kind=WorkflowKind.MODULE,
        upstream_target="pytest",
        default_args=("-q", "third_party/IsaacLab-Arena/isaaclab_arena/tests"),
        requirements=(_ISAACLAB_ARENA_SUBMODULE,),
    ),
    WorkflowSpec(
        name="convert",
        description="Convert Cyclo demonstrations to a LeRobot-compatible dataset.",
        kind=WorkflowKind.MODULE,
        upstream_target=None,
        requirements=(_DATASET,),
        readiness=WorkflowReadiness.UNSUPPORTED,
        readiness_detail="A Cyclo-to-LeRobot schema converter has not been implemented yet.",
    ),
    WorkflowSpec(
        name="train",
        description="Fine-tune a GR00T model from a Cyclo dataset.",
        kind=WorkflowKind.MODULE,
        upstream_target=None,
        requirements=(_DATASET,),
        readiness=WorkflowReadiness.UNSUPPORTED,
        readiness_detail="A Cyclo GR00T training profile has not been implemented yet.",
    ),
))

# Explicit name for callers that prefer the registry role over the mapping role.
WORKFLOW_REGISTRY = WORKFLOWS


def resolve_workflow(name: str) -> WorkflowSpec:
    """Resolve a workflow by canonical name or backward-compatible alias."""
    return WORKFLOWS.resolve(name)
