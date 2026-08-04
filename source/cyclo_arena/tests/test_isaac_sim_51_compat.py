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

"""Tests for the Isaac Sim 5.1 resolved-path compatibility boundary."""

import types
import unittest
from unittest import mock

from cyclo_arena.compat import isaac_sim_51


class IsaacSim51CompatibilityTest(unittest.TestCase):
    """Verify only the broken ``Ar.ResolvedPath`` input path is adapted."""

    @staticmethod
    def _modules(*, accepts_resolved_path: bool):
        class ResolvedPath:
            def __init__(self, path):
                self._path = path

            def GetPathString(self):
                return self._path

        calls = []
        crate_open_calls = []

        def is_usd_crate_file(filepath):
            calls.append(filepath)
            return str(filepath).endswith(".usd")

        class FileFormat:
            @staticmethod
            def GetFileExtension(filepath):
                if isinstance(filepath, ResolvedPath) and not accepts_resolved_path:
                    raise TypeError(
                        "FileFormat.GetFileExtension(ResolvedPath) did not match C++ signature"
                    )
                return "txt"

        def native_version_check(filepath, stage=None, usd_context_name=""):
            return False

        class Resolver:
            @staticmethod
            def CreateIdentifier(filepath):
                return filepath

            @staticmethod
            def Resolve(filepath):
                return ResolvedPath(filepath)

        class CrateInfo:
            @staticmethod
            def Open(filepath):
                crate_open_calls.append(filepath)

        omni_usd = types.SimpleNamespace(
            get_context=lambda _name: None,
            is_usd_crate_file_version_supported=native_version_check,
        )

        return (
            {
                "isaacsim.core.version": types.SimpleNamespace(
                    get_version=lambda: ("5.1.0", "release", "5", "1")
                ),
                "omni.usd": omni_usd,
                "omni.usd._impl.utils": types.SimpleNamespace(
                    is_usd_crate_file=is_usd_crate_file,
                    is_usd_crate_file_version_supported=native_version_check,
                ),
                "pxr.Ar": types.SimpleNamespace(
                    GetResolver=Resolver,
                    ResolvedPath=ResolvedPath,
                ),
                "pxr.Sdf": types.SimpleNamespace(FileFormat=FileFormat),
                "pxr.Usd": types.SimpleNamespace(CrateInfo=CrateInfo),
                "pxr.Tf": types.SimpleNamespace(ErrorException=RuntimeError),
                "carb": types.SimpleNamespace(log_error=lambda _message: None, log_warn=lambda _message: None),
            },
            calls,
            crate_open_calls,
        )

    def test_wraps_broken_helper_and_is_idempotent(self):
        modules, calls, crate_open_calls = self._modules(accepts_resolved_path=False)

        with mock.patch.object(isaac_sim_51.importlib, "import_module", side_effect=modules.__getitem__):
            installed = isaac_sim_51.install_isaac_sim_51_compat()
            version_result = modules["omni.usd"].is_usd_crate_file_version_supported("asset.usd")
            installed_again = isaac_sim_51.install_isaac_sim_51_compat()

        self.assertEqual(installed, ("omni.usd.is_usd_crate_file_version_supported(Ar.ResolvedPath)",))
        self.assertTrue(version_result)
        self.assertEqual(calls[-1], "asset.usd")
        self.assertEqual(crate_open_calls, ["asset.usd"])
        self.assertEqual(installed_again, ())

    def test_preserves_a_native_helper_that_accepts_resolved_paths(self):
        modules, _, _ = self._modules(accepts_resolved_path=True)
        native = modules["omni.usd._impl.utils"].is_usd_crate_file

        with mock.patch.object(isaac_sim_51.importlib, "import_module", side_effect=modules.__getitem__):
            installed = isaac_sim_51.install_isaac_sim_51_compat()

        self.assertEqual(installed, ())
        self.assertIs(modules["omni.usd._impl.utils"].is_usd_crate_file, native)

    def test_other_isaac_sim_versions_are_not_modified(self):
        modules, _, _ = self._modules(accepts_resolved_path=False)
        native = modules["omni.usd._impl.utils"].is_usd_crate_file
        modules["isaacsim.core.version"].get_version = lambda: ("6.0.0", "release", "6", "0")

        with mock.patch.object(isaac_sim_51.importlib, "import_module", side_effect=modules.__getitem__):
            installed = isaac_sim_51.install_isaac_sim_51_compat()

        self.assertEqual(installed, ())
        self.assertIs(modules["omni.usd._impl.utils"].is_usd_crate_file, native)


if __name__ == "__main__":
    unittest.main()
