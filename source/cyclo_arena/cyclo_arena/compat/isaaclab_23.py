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

"""Bridge Arena's optional G1 imports on Cyclo's pinned Isaac Lab 2.3 runtime."""

from __future__ import annotations

import importlib
from enum import Enum
from typing import Any


class ArenaRuntimeCompatibilityError(RuntimeError):
    """Report an Arena feature that the pinned Isaac Lab runtime cannot provide."""


class _RetargeterRequirement(Enum):
    """Import-time stand-in for the newer Isaac Lab retargeter requirement enum."""

    HAND_TRACKING = "hand_tracking"
    HEAD_TRACKING = "head_tracking"
    MOTION_CONTROLLER = "motion_controller"


class _XrAnchorRotationMode(Enum):
    """Import-time stand-in for the newer Isaac Lab XR anchor mode enum."""

    FIXED = "fixed"
    FOLLOW_PRIM = "follow_prim"
    FOLLOW_PRIM_SMOOTHED = "follow_prim_smoothed"
    CUSTOM = "custom_rotation"


_G1_MOTION_CONTROLLER_CONFIGS = (
    "G1LowerBodyStandingMotionControllerRetargeterCfg",
    "G1TriHandUpperBodyMotionControllerGripperRetargeterCfg",
)


def _unsupported_retargeter_config(name: str, base: type) -> type:
    """Create an import-compatible config that fails if the unsupported feature is used."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ArenaRuntimeCompatibilityError(
            f"{name} requires Isaac Lab OpenXR motion-controller APIs that are not available "
            "in Cyclo Lab's pinned Isaac Lab 2.3 runtime. FFW-SG2 policy inference is "
            "supported, but G1 motion-controller teleoperation is not."
        )

    return type(name, (base,), {"__init__": __init__, "__module__": __name__})


def install_isaaclab_23_compat() -> tuple[str, ...]:
    """Install only missing import-time APIs required by Arena's eager registry loading."""
    retargeters = importlib.import_module("isaaclab.devices.openxr.retargeters")
    xr_cfg = importlib.import_module("isaaclab.devices.openxr.xr_cfg")
    retargeter_base = importlib.import_module("isaaclab.devices.retargeter_base")
    installed: list[str] = []

    if not hasattr(retargeter_base.RetargeterBase, "Requirement"):
        retargeter_base.RetargeterBase.Requirement = _RetargeterRequirement
        installed.append("RetargeterBase.Requirement")

    for name in _G1_MOTION_CONTROLLER_CONFIGS:
        if not hasattr(retargeters, name):
            setattr(
                retargeters,
                name,
                _unsupported_retargeter_config(name, retargeter_base.RetargeterCfg),
            )
            installed.append(name)

    if not hasattr(xr_cfg, "XrAnchorRotationMode"):
        xr_cfg.XrAnchorRotationMode = _XrAnchorRotationMode
        installed.append("XrAnchorRotationMode")

    return tuple(installed)
