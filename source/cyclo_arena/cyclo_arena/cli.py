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

"""Unified launcher for Cyclo-owned and upstream Isaac Lab Arena workflows."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cyclo_arena.catalog import REGISTRY
from cyclo_arena.core.config import load_run_config
from cyclo_arena.core.model_resolver import discover_models, model_search_root
from cyclo_arena.core.robot_pose import list_robot_poses

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ISAAC_SIM_PYTHON = Path("/isaac-sim/python.sh")
ISAAC_SIM_REEXEC_ENV = "CYCLO_ARENA_ISAAC_SIM_PYTHON"
COMPOSED_ENVIRONMENT_CLASS = (
    "cyclo_arena.environments.composed:CycloArenaEnvironment"
)
COMPOSED_ENVIRONMENT_NAME = "cyclo_composed"


@dataclass(frozen=True)
class WorkflowTarget:
    """Describe an upstream Python module or repository script."""

    kind: str
    target: str
    default_args: tuple[str, ...] = ()


PASSTHROUGH_WORKFLOWS = {
    "policy": WorkflowTarget("module", "isaaclab_arena.evaluation.policy_runner"),
    "evaluate": WorkflowTarget("module", "isaaclab_arena.evaluation.eval_runner"),
    "teleop": WorkflowTarget(
        "module", "isaaclab_arena.scripts.imitation_learning.teleop"
    ),
    "record": WorkflowTarget(
        "module", "isaaclab_arena.scripts.imitation_learning.record_demos"
    ),
    "replay": WorkflowTarget(
        "module", "isaaclab_arena.scripts.imitation_learning.replay_demos"
    ),
    "annotate": WorkflowTarget(
        "module", "isaaclab_arena.scripts.imitation_learning.annotate_demos"
    ),
    "generate": WorkflowTarget(
        "module", "isaaclab_arena.scripts.imitation_learning.generate_dataset"
    ),
    "serve": WorkflowTarget(
        "module", "isaaclab_arena.remote_policy.remote_policy_server_runner"
    ),
    "rl-train": WorkflowTarget(
        "script",
        "third_party/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py",
    ),
    "gr00t-server": WorkflowTarget(
        "shell", "third_party/IsaacLab-Arena/docker/run_gr00t_server.sh"
    ),
    "test": WorkflowTarget(
        "module",
        "pytest",
        ("-q", "third_party/IsaacLab-Arena/isaaclab_arena/tests"),
    ),
}


def _python_launcher() -> str:
    """Return the Isaac Sim Python launcher when it is available."""
    if ISAAC_SIM_PYTHON.is_file():
        return str(ISAAC_SIM_PYTHON)
    return sys.executable


def _exec_workflow(target: WorkflowTarget, forwarded_args: Sequence[str]) -> None:
    """Replace this process with an upstream Arena workflow."""
    workflow_args = list(forwarded_args) or list(target.default_args)
    if target.kind == "module":
        command = [_python_launcher(), "-m", target.target, *workflow_args]
    else:
        target_path = REPOSITORY_ROOT / target.target
        if not target_path.is_file():
            raise FileNotFoundError(
                f"Workflow target is unavailable: {target_path}. "
                "Initialize the corresponding optional submodule or dependency first."
            )
        if target.kind == "script":
            command = [_python_launcher(), str(target_path), *workflow_args]
        elif target.kind == "shell":
            command = ["bash", str(target_path), *workflow_args]
        else:
            raise ValueError(f"Unsupported workflow target kind: {target.kind}")
    os.execv(command[0], command)


def _ensure_isaac_sim_python(command_args: Sequence[str]) -> None:
    """Relaunch the CLI with Isaac Sim's initialized Python environment."""
    if not ISAAC_SIM_PYTHON.is_file() or os.environ.get(ISAAC_SIM_REEXEC_ENV):
        return
    command = [
        str(ISAAC_SIM_PYTHON),
        "-m",
        "cyclo_arena.cli",
        *command_args,
    ]
    environment = os.environ.copy()
    environment[ISAAC_SIM_REEXEC_ENV] = "1"
    os.execve(command[0], command, environment)


def _add_run_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "run", help="Launch a Cyclo-owned Arena environment with a policy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Cyclo Arena YAML run configuration.",
    )
    parser.add_argument("--robot", choices=tuple(REGISTRY.robots))
    parser.add_argument("--scene", choices=tuple(REGISTRY.scenes))
    parser.add_argument("--task", choices=tuple(REGISTRY.tasks))
    parser.add_argument("--policy-type")
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--num-steps", type=int)
    length.add_argument("--num-episodes", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    cameras = parser.add_mutually_exclusive_group()
    cameras.add_argument(
        "--enable-cameras",
        dest="enable_cameras",
        action="store_true",
        default=None,
    )
    cameras.add_argument(
        "--disable-cameras",
        dest="enable_cameras",
        action="store_false",
    )
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument(
        "--headless", dest="headless", action="store_true", default=None
    )
    visualization.add_argument(
        "--windowed", dest="headless", action="store_false"
    )
    parser.add_argument("--embodiment")
    parser.add_argument("--robot-pose")
    parser.add_argument(
        "--resolved-model-adapter",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--task-description")
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-port", type=int)
    parser.add_argument("--remote-timeout-ms", type=int)
    parser.add_argument(
        "--robot-position-xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--robot-yaw", type=float)
    parser.add_argument(
        "--head-position",
        type=float,
        nargs=2,
        metavar=("HEAD_1", "HEAD_2"),
    )
    parser.add_argument("--lift-position", type=float)
    parser.add_argument(
        "--kitchen-layout",
        type=int,
        help="Lightwheel RoboCasa kitchen layout ID.",
    )
    parser.add_argument(
        "--kitchen-style",
        type=int,
        help="Lightwheel RoboCasa kitchen style ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the upstream command without running it.",
    )


def _build_parser() -> argparse.ArgumentParser:
    workflow_names = ", ".join(PASSTHROUGH_WORKFLOWS)
    parser = argparse.ArgumentParser(
        prog="cyclo-arena",
        description="ROBOTIS launcher for Isaac Lab Arena environments and workflows.",
        epilog=(
            "Exact upstream passthrough commands: "
            f"{workflow_names}. Example: cyclo-arena policy --help"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    doctor = subparsers.add_parser(
        "doctor", help="Verify the complete Cyclo Arena installation."
    )
    doctor.add_argument("--strict", action="store_true")
    listing = subparsers.add_parser("list", help="List Cyclo Arena integrations.")
    listing.add_argument(
        "category",
        choices=(
            "robots",
            "scenes",
            "tasks",
            "embodiments",
            "policies",
            "models",
            "model-adapters",
            "poses",
            "workflows",
        ),
        nargs="?",
        default="robots",
    )
    _add_run_parser(subparsers)
    return parser


def _print_catalog(category: str) -> None:
    """Print one static catalog without starting Isaac Sim."""
    if category == "models":
        search_root = model_search_root()
        models = discover_models(REGISTRY, search_root)
        print(f"Model search root: {search_root}")
        if not models:
            print("No GR00T checkpoints were found.")
            return
        for model in models:
            compatibility = (
                ", ".join(model.compatible_adapters)
                if model.compatible_adapters
                else "incompatible"
            )
            print(model.checkpoint)
            print(f"  type: {model.model_type}")
            print(f"  adapter: {compatibility}")
        return
    if category == "robots":
        entries = {name: spec.description for name, spec in REGISTRY.robots.items()}
    elif category == "scenes":
        entries = {name: spec.description for name, spec in REGISTRY.scenes.items()}
    elif category == "tasks":
        entries = {name: spec.description for name, spec in REGISTRY.tasks.items()}
    elif category == "embodiments":
        entries = {
            embodiment: f"Arena embodiment for {robot.name}."
            for robot in REGISTRY.robots.values()
            for embodiment in robot.embodiments
        }
    elif category == "policies":
        entries = {
            name: spec.description for name, spec in REGISTRY.policies.items()
        }
    elif category == "model-adapters":
        entries = {
            name: spec.description
            for name, spec in REGISTRY.model_adapters.items()
        }
    elif category == "poses":
        entries = {
            f"{robot}/{pose}": "Named initial joint pose."
            for robot in REGISTRY.robots
            for pose in list_robot_poses(robot)
        }
    else:
        entries = {
            name: spec.description for name, spec in REGISTRY.workflows.items()
        }
    for name, description in entries.items():
        print(f"{name:24} {description}")


def _resolve_run_args(args: argparse.Namespace) -> argparse.Namespace:
    """Merge an optional run file with explicit CLI overrides."""
    values = {}
    if args.config is not None:
        values.update(
            load_run_config(args.config).to_run_values(
                REGISTRY,
                model_adapter_override=args.resolved_model_adapter,
            )
        )

    override_names = (
        "robot",
        "scene",
        "task",
        "policy_type",
        "num_steps",
        "num_episodes",
        "num_envs",
        "device",
        "seed",
        "enable_cameras",
        "headless",
        "embodiment",
        "robot_pose",
        "task_description",
        "remote_host",
        "remote_port",
        "remote_timeout_ms",
        "robot_position_xyz",
        "robot_yaw",
        "head_position",
        "lift_position",
        "kitchen_layout",
        "kitchen_style",
    )
    for name in override_names:
        value = getattr(args, name)
        if value is not None:
            values[name] = value

    if args.num_steps is not None:
        values["num_episodes"] = None
    elif args.num_episodes is not None:
        values["num_steps"] = None

    defaults = {
        "robot": None,
        "scene": None,
        "task": "scene_only",
        "policy_type": "zero_action",
        "num_steps": None,
        "num_episodes": None,
        "num_envs": 1,
        "device": None,
        "seed": None,
        "enable_cameras": False,
        "headless": False,
        "embodiment": None,
        "robot_pose": None,
        "task_description": None,
        "remote_host": None,
        "remote_port": None,
        "remote_timeout_ms": None,
        "robot_position_xyz": None,
        "robot_yaw": None,
        "head_position": None,
        "lift_position": None,
        "kitchen_layout": 1,
        "kitchen_style": 1,
    }
    resolved = argparse.Namespace(
        **{name: values.get(name, default) for name, default in defaults.items()},
        dry_run=args.dry_run,
    )
    if resolved.robot is None or resolved.scene is None:
        raise ValueError(
            "Select robot and scene in --config or with --robot and --scene"
        )
    return resolved


def _run_environment(args: argparse.Namespace) -> int:
    """Construct the correctly ordered upstream policy-runner arguments."""
    args = _resolve_run_args(args)
    plan = REGISTRY.compose(args.robot, args.scene, args.task)
    embodiment = args.embodiment or plan.robot.default_embodiment
    if embodiment not in plan.robot.embodiments:
        allowed = ", ".join(plan.robot.embodiments)
        raise ValueError(
            f"Embodiment {embodiment!r} is not valid for robot "
            f"{plan.robot.name!r}. Choose one of: {allowed}"
        )

    forwarded = [
        "--environment",
        COMPOSED_ENVIRONMENT_CLASS,
        "--policy_type",
        args.policy_type,
    ]
    if args.num_episodes is not None:
        forwarded += ["--num_episodes", str(args.num_episodes)]
    else:
        forwarded += ["--num_steps", str(args.num_steps or 1800)]
    forwarded += ["--num_envs", str(args.num_envs)]
    if args.device:
        forwarded += ["--device", args.device]
    if args.seed is not None:
        forwarded += ["--seed", str(args.seed)]
    if args.enable_cameras:
        forwarded.append("--enable_cameras")
    if args.headless:
        forwarded.append("--headless")
    if args.remote_host:
        forwarded += ["--remote_host", args.remote_host]
    if args.remote_port is not None:
        forwarded += ["--remote_port", str(args.remote_port)]
    if args.remote_timeout_ms is not None:
        forwarded += ["--remote_timeout_ms", str(args.remote_timeout_ms)]
    forwarded += [
        COMPOSED_ENVIRONMENT_NAME,
        "--robot",
        args.robot,
        "--scene",
        args.scene,
        "--cyclo_task",
        args.task,
        "--embodiment",
        embodiment,
    ]
    if args.robot_pose:
        forwarded += ["--robot_pose", args.robot_pose]
    if args.task_description:
        forwarded += ["--task_description", args.task_description]
    if args.robot_position_xyz is not None:
        forwarded += [
            "--robot_position_xyz",
            *(str(value) for value in args.robot_position_xyz),
        ]
    if args.robot_yaw is not None:
        forwarded += ["--robot_yaw", str(args.robot_yaw)]
    if args.head_position is not None:
        forwarded += [
            "--head_position",
            *(str(value) for value in args.head_position),
        ]
    if args.lift_position is not None:
        forwarded += ["--lift_position", str(args.lift_position)]
    forwarded += ["--kitchen_layout", str(args.kitchen_layout)]
    forwarded += ["--kitchen_style", str(args.kitchen_style)]
    target = PASSTHROUGH_WORKFLOWS["policy"]
    if args.dry_run:
        command = [_python_launcher(), "-m", target.target, *forwarded]
        print(shlex.join(command))
        return 0
    _exec_workflow(target, forwarded)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Cyclo Arena command-line interface."""
    command_args = list(sys.argv[1:] if argv is None else argv)
    if command_args and command_args[0] in PASSTHROUGH_WORKFLOWS:
        _exec_workflow(PASSTHROUGH_WORKFLOWS[command_args[0]], command_args[1:])
        return 0

    parser = _build_parser()
    args = parser.parse_args(command_args)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "doctor":
        _ensure_isaac_sim_python(command_args)
        from cyclo_arena.doctor import run_checks

        return run_checks(strict=args.strict)
    if args.command == "list":
        _print_catalog(args.category)
        return 0
    if args.command == "run":
        _ensure_isaac_sim_python(command_args)
        return _run_environment(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
