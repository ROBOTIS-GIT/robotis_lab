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
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

CYCLO_LAB_ROOT = Path(os.environ.get("CYCLOLAB_PATH", Path(__file__).resolve().parents[3])).resolve()
ISAACLAB_ROOT = CYCLO_LAB_ROOT / "third_party" / "IsaacLab"
ARENA_ROOT = CYCLO_LAB_ROOT / "third_party" / "IsaacLab-Arena"
ZENOH_SDK_ROOT = CYCLO_LAB_ROOT / "third_party" / "zenoh_ros2_sdk"
EXPECTED_ISAACLAB_COMMIT = "5528d986d8909825a29f3c97656108abf054a261"
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


def _submodule_commit(submodule_root: Path) -> str | None:
    if not (submodule_root / ".git").exists():
        return None
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={submodule_root}",
            "-C",
            str(submodule_root),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


def run_checks(strict: bool = False) -> int:
    """Run installation checks and return a process status."""
    failures: list[str] = []
    _check_submodule("Isaac Lab", ISAACLAB_ROOT, EXPECTED_ISAACLAB_COMMIT, failures)
    # Arena follows its configured compatibility branch. Validate its public
    # integration contracts below instead of rejecting every upstream update.
    _check_submodule("Arena", ARENA_ROOT, None, failures)
    _check_submodule("Zenoh ROS2 SDK", ZENOH_SDK_ROOT, EXPECTED_ZENOH_SDK_COMMIT, failures)

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
