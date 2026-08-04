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

"""Validate Cyclo Arena's runtime, upstream APIs, and shared-asset contract."""

from __future__ import annotations

import argparse
import configparser
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

CYCLO_LAB_ROOT = Path(os.environ.get("CYCLOLAB_PATH", Path(__file__).resolve().parents[3])).resolve()
ISAACLAB_SUBMODULE_PATH = Path("third_party/IsaacLab")
ARENA_SUBMODULE_PATH = Path("third_party/IsaacLab-Arena")
ISAACLAB_ROOT = CYCLO_LAB_ROOT / ISAACLAB_SUBMODULE_PATH
ARENA_ROOT = CYCLO_LAB_ROOT / ARENA_SUBMODULE_PATH
ZENOH_SDK_ROOT = CYCLO_LAB_ROOT / "third_party" / "zenoh_ros2_sdk"
ARENA_SUBMODULE_NAME = "third_party/IsaacLab-Arena"
EXPECTED_ARENA_URL = "https://github.com/isaac-sim/IsaacLab-Arena.git"
EXPECTED_ARENA_BRANCH = "feature/arena_v0.2_on_lab_2.3"
EXPECTED_ZENOH_SDK_COMMIT = "be2c4d4595305a9c282fca09820a8b3bfb8076a3"
EXPECTED_ZENOH_SDK_VERSION = "0.1.8"
EXPECTED_CYCLO_ARENA_VERSION = "0.1.0"
OPTIONAL_MODULES = {
    "onnxruntime": "ONNX policy execution",
    "vuer": "Vuer teleoperation",
    "lightwheel_sdk": "Lightwheel assets and benchmarks",
    "tenacity": "Plotly retry support",
}
DEFERRED_SIMULATION_MODULES = ("isaaclab_arena.policy.action_chunking_client",)


def _run_git(root: Path, *args: str) -> str | None:
    """Run a read-only Git query and return stripped stdout when it succeeds."""
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _submodule_commit(submodule_root: Path) -> str | None:
    if not (submodule_root / ".git").exists():
        return None
    return _run_git(submodule_root, "rev-parse", "HEAD")


def _gitlink_entry(superproject_root: Path, submodule_root: Path) -> tuple[str, str, str] | None:
    """Return the committed mode, object type, and revision for a submodule path."""
    try:
        relative_path = submodule_root.relative_to(superproject_root)
    except ValueError:
        return None
    output = _run_git(superproject_root, "ls-tree", "HEAD", "--", relative_path.as_posix())
    if not output:
        return None
    metadata, separator, _ = output.partition("\t")
    fields = metadata.split()
    if not separator or len(fields) != 3:
        return None
    mode, object_type, commit = fields
    return mode, object_type, commit


def _gitmodule_values(gitmodules_path: Path, submodule_name: str) -> dict[str, str] | None:
    """Load one submodule section from a .gitmodules file."""
    try:
        contents = gitmodules_path.read_text(encoding="utf-8")
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(contents)
    except (OSError, configparser.Error):
        return None
    section = f'submodule "{submodule_name}"'
    if not parser.has_section(section):
        return None
    return dict(parser.items(section))


def _import_module(module_name: str):
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # noqa: BLE001 - report every import failure together
        return None, f"{type(exc).__name__}: {exc}"


def _check_submodule(
    label: str,
    root: Path,
    expected_commit: str | None,
    failures: list[str],
) -> None:
    commit = _submodule_commit(root)
    if commit is None:
        failures.append(f"{label} submodule is missing or uninitialized: {root}")
        return
    print(f"[OK] {label} submodule: {commit}")
    if expected_commit is not None and commit != expected_commit:
        failures.append(f"Expected {label} {expected_commit}, found {commit}")


def _check_pinned_submodule(
    label: str,
    superproject_root: Path,
    submodule_root: Path,
    failures: list[str],
) -> None:
    """Validate an initialized submodule against its committed gitlink."""
    entry = _gitlink_entry(superproject_root, submodule_root)
    if entry is None:
        failures.append(f"{label} has no committed gitlink: {submodule_root}")
        return
    mode, object_type, expected_commit = entry
    if mode != "160000" or object_type != "commit":
        failures.append(f"{label} must be a 160000 gitlink, found mode={mode} type={object_type}: {submodule_root}")
        return

    commit = _submodule_commit(submodule_root)
    if commit is None:
        failures.append(f"{label} submodule is missing or uninitialized: {submodule_root}")
        return
    if commit != expected_commit:
        failures.append(f"Expected {label} gitlink {expected_commit}, found {commit}")
        return
    print(f"[OK] {label} submodule matches gitlink: {commit}")


def _check_arena_gitmodule(superproject_root: Path, failures: list[str]) -> None:
    """Validate Arena's official upstream and Lab 2.3 compatibility branch."""
    values = _gitmodule_values(superproject_root / ".gitmodules", ARENA_SUBMODULE_NAME)
    if values is None:
        failures.append(f"Arena submodule is not configured in {superproject_root / '.gitmodules'}")
        return

    expected_values = {
        "path": ARENA_SUBMODULE_PATH.as_posix(),
        "url": EXPECTED_ARENA_URL,
        "branch": EXPECTED_ARENA_BRANCH,
    }
    for key, expected_value in expected_values.items():
        actual_value = values.get(key)
        if actual_value != expected_value:
            failures.append(f"Expected Arena .gitmodules {key}={expected_value}, found {actual_value}")
    if all(values.get(key) == expected_value for key, expected_value in expected_values.items()):
        print(f"[OK] Arena upstream: {EXPECTED_ARENA_URL} ({EXPECTED_ARENA_BRANCH})")


def _check_dependency_contracts(failures: list[str]) -> None:
    """Validate pinned simulation dependencies and the fixed Zenoh release."""
    _check_pinned_submodule("Isaac Lab", CYCLO_LAB_ROOT, ISAACLAB_ROOT, failures)
    _check_arena_gitmodule(CYCLO_LAB_ROOT, failures)
    _check_pinned_submodule("Arena", CYCLO_LAB_ROOT, ARENA_ROOT, failures)
    _check_submodule("Zenoh ROS2 SDK", ZENOH_SDK_ROOT, EXPECTED_ZENOH_SDK_COMMIT, failures)


def run_checks(strict: bool = False) -> int:
    """Run installation checks and return a process status."""
    failures: list[str] = []
    _check_dependency_contracts(failures)

    if str(ARENA_ROOT) not in sys.path:
        sys.path.insert(0, str(ARENA_ROOT))

    for module_name in (
        "isaaclab",
        "isaaclab_arena",
        "isaaclab_arena.assets.asset_registry",
        "isaaclab_arena.remote_policy.policy_client",
        "isaaclab_arena.remote_policy.policy_server",
        "isaaclab_arena.remote_policy.remote_policy_server_runner",
        "cyclo_lab",
        "cyclo_arena",
        "cyclo_arena.environments.composed",
        "zenoh_ros2_sdk",
        "msgpack",
        "zmq",
    ):
        module, error = _import_module(module_name)
        if error is not None:
            failures.append(f"Cannot import {module_name}: {error}")
        else:
            print(f"[OK] Import {module_name}: {getattr(module, '__file__', None)}")

    for module_name in DEFERRED_SIMULATION_MODULES:
        module_path = ARENA_ROOT / (module_name.replace(".", "/") + ".py")
        if not module_path.is_file():
            failures.append(f"Cannot find deferred simulation module: {module_name}")
        else:
            print(f"[OK] Deferred simulation module: {module_path}")

    for distribution, expected_version in (
        ("cyclo-arena", EXPECTED_CYCLO_ARENA_VERSION),
        ("zenoh-ros2-sdk", EXPECTED_ZENOH_SDK_VERSION),
    ):
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Package metadata is unavailable: {distribution}")
        else:
            print(f"[OK] Package {distribution}: {installed_version}")
            if installed_version != expected_version:
                failures.append(f"Expected {distribution} {expected_version}, found {installed_version}")

    for executable in ("cyclo-arena", "zenoh-ros2"):
        executable_path = shutil.which(executable)
        if executable_path is None:
            failures.append(f"Required CLI is not on PATH: {executable}")
        else:
            print(f"[OK] CLI {executable}: {executable_path}")

    action_spec, error = _import_module("cyclo_lab.robot_specs.ffw.sg2")
    if error is not None:
        failures.append(f"Cannot import FFW-SG2 action contract: {error}")
    else:
        action_joint_names = action_spec.FFW_SG2_ACTION_JOINT_NAMES
        if len(action_joint_names) != 19 or len(set(action_joint_names)) != 19:
            failures.append("FFW-SG2 action contract must contain 19 unique joints")
        else:
            print("[OK] Shared FFW-SG2 action contract: 19 ordered joint values")

    for module_name, capability in OPTIONAL_MODULES.items():
        _, error = _import_module(module_name)
        if error is None:
            print(f"[OK] Optional dependency {module_name}: {capability}")
        else:
            print(f"[MISSING] Optional dependency {module_name}: {capability} ({error})")
            if strict:
                failures.append(f"Optional dependency {module_name} is unavailable")

    print(f"[INFO] Python: {sys.version.split()[0]}")
    if failures:
        for failure in failures:
            print(f"[ERROR] {failure}", file=sys.stderr)
        return 1
    print("[OK] Cyclo Arena is available with the shared Cyclo Lab assets.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and run the installation checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    return run_checks(strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
