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

"""Tests for the deferred Isaac Lab 2.3 compatibility boundary."""

import types
import unittest
from unittest import mock

from cyclo_arena.compat import isaaclab_23


class IsaacLab23CompatibilityTest(unittest.TestCase):
    """Verify feature-detected shims remain narrow, idempotent, and fail-fast."""

    @staticmethod
    def _modules():
        class RetargeterCfg:
            pass

        class RetargeterBase:
            pass

        return {
            "isaaclab.devices.openxr.retargeters": types.SimpleNamespace(),
            "isaaclab.devices.openxr.xr_cfg": types.SimpleNamespace(),
            "isaaclab.devices.retargeter_base": types.SimpleNamespace(
                RetargeterBase=RetargeterBase,
                RetargeterCfg=RetargeterCfg,
            ),
        }

    def test_installs_missing_import_apis_and_is_idempotent(self):
        modules = self._modules()

        with mock.patch.object(isaaclab_23.importlib, "import_module", side_effect=modules.__getitem__):
            installed = isaaclab_23.install_isaaclab_23_compat()
            retargeter_class = (
                modules["isaaclab.devices.openxr.retargeters"].G1LowerBodyStandingMotionControllerRetargeterCfg
            )
            installed_again = isaaclab_23.install_isaaclab_23_compat()

        self.assertEqual(
            set(installed),
            {
                "RetargeterBase.Requirement",
                "G1LowerBodyStandingMotionControllerRetargeterCfg",
                "G1TriHandUpperBodyMotionControllerGripperRetargeterCfg",
                "XrAnchorRotationMode",
            },
        )
        self.assertEqual(installed_again, ())
        self.assertIs(
            retargeter_class,
            modules["isaaclab.devices.openxr.retargeters"].G1LowerBodyStandingMotionControllerRetargeterCfg,
        )
        self.assertEqual(
            modules["isaaclab.devices.openxr.xr_cfg"].XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED.value,
            "follow_prim_smoothed",
        )

    def test_missing_motion_controller_config_fails_if_instantiated(self):
        modules = self._modules()

        with mock.patch.object(isaaclab_23.importlib, "import_module", side_effect=modules.__getitem__):
            isaaclab_23.install_isaaclab_23_compat()

        config_class = modules["isaaclab.devices.openxr.retargeters"].G1LowerBodyStandingMotionControllerRetargeterCfg
        with self.assertRaisesRegex(isaaclab_23.ArenaRuntimeCompatibilityError, "G1 motion-controller teleoperation"):
            config_class(sim_device="cuda:0")

    def test_native_apis_are_never_overwritten(self):
        modules = self._modules()
        native_requirement = object()
        native_lower = object()
        native_upper = object()
        native_anchor = object()
        modules["isaaclab.devices.retargeter_base"].RetargeterBase.Requirement = native_requirement
        modules["isaaclab.devices.openxr.retargeters"].G1LowerBodyStandingMotionControllerRetargeterCfg = native_lower
        (
            modules["isaaclab.devices.openxr.retargeters"].G1TriHandUpperBodyMotionControllerGripperRetargeterCfg
        ) = native_upper
        modules["isaaclab.devices.openxr.xr_cfg"].XrAnchorRotationMode = native_anchor

        with mock.patch.object(isaaclab_23.importlib, "import_module", side_effect=modules.__getitem__):
            installed = isaaclab_23.install_isaaclab_23_compat()

        self.assertEqual(installed, ())
        self.assertIs(modules["isaaclab.devices.retargeter_base"].RetargeterBase.Requirement, native_requirement)
        self.assertIs(
            modules["isaaclab.devices.openxr.retargeters"].G1LowerBodyStandingMotionControllerRetargeterCfg,
            native_lower,
        )
        self.assertIs(
            modules["isaaclab.devices.openxr.retargeters"].G1TriHandUpperBodyMotionControllerGripperRetargeterCfg,
            native_upper,
        )
        self.assertIs(modules["isaaclab.devices.openxr.xr_cfg"].XrAnchorRotationMode, native_anchor)


if __name__ == "__main__":
    unittest.main()
