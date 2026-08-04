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

"""Tests for the thin script's host/container routing."""

import unittest
from pathlib import Path
from unittest import mock

from cyclo_arena import entrypoint


class LaunchRequestTest(unittest.TestCase):
    """Verify friendly commands resolve to one unambiguous execution source."""

    def test_no_arguments_preserve_the_editable_default_config(self):
        request = entrypoint.parse_launch_request([])

        self.assertEqual(request.source_kind, "config")
        self.assertEqual(request.source, str(entrypoint.DEFAULT_CONFIG))
        self.assertEqual(request.forwarded_args, ())

    def test_named_profile_hides_its_config_path(self):
        request = entrypoint.parse_launch_request([
            "infer",
            "ffw_sg2_showroom_gr00t",
            "--num-steps",
            "2",
        ])

        self.assertEqual(request.source_kind, "profile")
        self.assertEqual(request.source, "ffw_sg2_showroom_gr00t")
        self.assertEqual(request.forwarded_args, ("--num-steps", "2"))

    def test_legacy_catalog_flags_use_the_static_cli(self):
        request = entrypoint.parse_launch_request(["--list-scenes"])

        self.assertEqual(request.static_args, ("list", "scenes"))

    def test_conflicting_sources_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "Select only one"):
            entrypoint.parse_launch_request([
                "--config",
                "run.yaml",
                "infer",
                "profile",
            ])


class EntrypointTest(unittest.TestCase):
    """Verify routing stays outside simulation-specific implementation code."""

    @mock.patch.object(entrypoint, "ISAAC_SIM_PYTHON", Path("/definitely/missing/isaac-sim/python.sh"))
    @mock.patch.object(entrypoint.host_launcher, "main", return_value=7)
    def test_host_profile_is_forwarded_to_the_orchestrator(self, host_main):
        result = entrypoint.main(["infer", "demo", "--headless"])

        self.assertEqual(result, 7)
        host_main.assert_called_once_with([
            "--profile",
            "demo",
            "--",
            "--headless",
        ])

    @mock.patch.object(entrypoint.cli, "main", return_value=0)
    def test_inside_run_is_forwarded_to_the_composed_cli(self, cli_main):
        result = entrypoint.main([
            "--inside",
            "infer",
            "demo",
            "--headless",
            "--num-steps",
            "1",
        ])

        self.assertEqual(result, 0)
        cli_main.assert_called_once_with([
            "infer",
            "demo",
            "--headless",
            "--num-steps",
            "1",
        ])

    @mock.patch.object(entrypoint.cli, "main", return_value=0)
    def test_static_commands_do_not_launch_docker(self, cli_main):
        result = entrypoint.main(["list", "profiles"])

        self.assertEqual(result, 0)
        cli_main.assert_called_once_with(("list", "profiles"))


if __name__ == "__main__":
    unittest.main()
