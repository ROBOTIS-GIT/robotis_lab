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

"""Load model-independent robot initial poses from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

POSE_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "robots"


@dataclass(frozen=True)
class RobotPoseConfig:
    """Describe one named robot joint pose."""

    name: str
    robot: str
    joint_positions: Mapping[str, float]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    assert isinstance(value, Mapping), f"{label} must be a mapping"
    return value


def load_robot_pose(robot: str, pose: str) -> RobotPoseConfig:
    """Load one robot pose by its stable name."""
    import yaml

    assert pose and all(character.isalnum() or character in "_-" for character in pose), (
        f"Invalid robot pose name: {pose!r}"
    )
    pose_path = POSE_CONFIG_ROOT / robot / "poses" / f"{pose}.yaml"
    assert pose_path.is_file(), f"Unknown pose {pose!r} for robot {robot!r}: {pose_path}"
    with pose_path.open(encoding="utf-8") as pose_file:
        values = _mapping(yaml.safe_load(pose_file), "robot pose")
    unknown = set(values) - {"name", "robot", "joint_positions"}
    assert not unknown, f"Unknown robot pose keys: {sorted(unknown)}"
    assert values.get("name") == pose, (
        f"Robot pose name {values.get('name')!r} does not match {pose!r}"
    )
    assert values.get("robot") == robot, (
        f"Robot pose {pose!r} belongs to {values.get('robot')!r}, not {robot!r}"
    )
    joint_positions = _mapping(values.get("joint_positions"), "joint_positions")
    return RobotPoseConfig(
        name=pose,
        robot=robot,
        joint_positions={name: float(value) for name, value in joint_positions.items()},
    )


def list_robot_poses(robot: str) -> tuple[str, ...]:
    """Return all named pose files for one robot."""
    pose_root = POSE_CONFIG_ROOT / robot / "poses"
    if not pose_root.is_dir():
        return ()
    return tuple(sorted(path.stem for path in pose_root.glob("*.yaml")))
