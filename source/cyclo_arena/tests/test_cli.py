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

"""Tests for the simulator-independent Cyclo Arena CLI behavior."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

from cyclo_arena import cli
from cyclo_arena.catalog import REGISTRY


class CliTest(unittest.TestCase):
    """Verify command construction without launching Isaac Sim."""

    @staticmethod
    def _direct_run_args(scene: str) -> list[str]:
        return [
            "run",
            "--robot",
            "ffw_sg2",
            "--scene",
            scene,
            "--policy-type",
            "zero_action",
        ]

    def test_run_dry_run_builds_composed_environment_command(self):
        output = io.StringIO()
        args = [
            *self._direct_run_args("galileo"),
            "--num-steps",
            "2",
            "--headless",
            "--remote-port",
            "5556",
            "--robot-position-xyz",
            "0.0",
            "0.18",
            "0.0",
            "--head-position",
            "0.5",
            "0.0",
            "--lift-position",
            "-0.2",
            "--dry-run",
        ]
        with contextlib.redirect_stdout(output):
            result = cli.main(args)

        command = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("isaaclab_arena.evaluation.policy_runner", command)
        self.assertIn("CycloArenaEnvironment", command)
        self.assertIn("cyclo_composed --robot ffw_sg2 --scene galileo", command)
        self.assertIn("--num_steps 2", command)
        self.assertIn("--headless", command)
        self.assertIn("--remote_port 5556", command)
        self.assertIn("--embodiment ffw_sg2_abs_joint_pos", command)
        self.assertIn("--robot_position_xyz 0.0 0.18 0.0", command)
        self.assertIn("--head_position 0.5 0.0", command)
        self.assertIn("--lift_position -0.2", command)

    def test_every_scene_builds_one_composed_environment(self):
        for scene in REGISTRY.scenes:
            with self.subTest(scene=scene):
                output = io.StringIO()
                args = [*self._direct_run_args(scene), "--dry-run"]
                with contextlib.redirect_stdout(output):
                    result = cli.main(args)

                command = output.getvalue()
                self.assertEqual(result, 0)
                self.assertIn("CycloArenaEnvironment", command)
                self.assertIn(f"--scene {scene}", command)

    def test_lightwheel_kitchen_options_are_forwarded(self):
        output = io.StringIO()
        args = [
            *self._direct_run_args("lightwheel_robocasa_kitchen"),
            "--kitchen-layout",
            "5",
            "--kitchen-style",
            "7",
            "--dry-run",
        ]
        with contextlib.redirect_stdout(output):
            result = cli.main(args)

        command = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("--kitchen_layout 5", command)
        self.assertIn("--kitchen_style 7", command)

    def test_checkpoint_config_uses_resolved_adapter_and_robot_pose(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "run.yaml"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = cli.main([
                "run",
                "--config",
                str(config_path),
                "--resolved-model-adapter",
                "ffw_sg2_gr00t_n17_showroom",
                "--remote-port",
                "61234",
                "--dry-run",
            ])

        command = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("--remote_port 61234", command)
        self.assertIn("--embodiment ffw_sg2_mobile_abs_joint_pos", command)
        self.assertIn("--robot_pose showroom", command)

    def test_passthrough_preserves_native_arguments(self):
        with mock.patch.object(cli, "_exec_workflow") as execute:
            result = cli.main(["policy", "--help"])

        self.assertEqual(result, 0)
        execute.assert_called_once_with(cli.PASSTHROUGH_WORKFLOWS["policy"], ["--help"])


if __name__ == "__main__":
    unittest.main()
