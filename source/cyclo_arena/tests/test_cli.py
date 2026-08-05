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
from cyclo_arena.core.manifest import ManifestModel, ResolvedManifest
from cyclo_arena.core.profile_store import DEFAULT_PROFILE_ID, ProfileStore


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

    @staticmethod
    def _model_manifest(remote_port: int | None = None) -> ResolvedManifest:
        return ResolvedManifest(
            workflow="infer",
            profile="demo",
            model=ManifestModel(
                checkpoint=Path("/models/showroom_groot"),
                adapter="ffw_sg2_gr00t_n17_showroom",
                model_type="Gr00tN1d7",
            ),
            run_values={
                "robot": "ffw_sg2",
                "scene": "robotis_showroom_training",
                "task": "scene_only",
                "embodiment": "ffw_sg2_mobile_abs_joint_pos",
                "policy_type": "isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy",
                "remote_host": "127.0.0.1",
                "remote_port": remote_port,
                "enable_cameras": True,
            },
        )

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
        self.assertIn("cyclo_arena.compat.policy_runner", command)
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
        config_path = ProfileStore().get(DEFAULT_PROFILE_ID).path
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

    def test_inference_keeps_the_official_upstream_target_behind_the_adapter(self):
        infer = cli.PASSTHROUGH_WORKFLOWS["infer"]

        self.assertEqual(infer.upstream_target, "isaaclab_arena.evaluation.policy_runner")
        self.assertEqual(infer.launcher_target, "cyclo_arena.compat.policy_runner")

    def test_infer_accepts_a_named_profile_without_a_config_path(self):
        manifest = ResolvedManifest(
            workflow="infer",
            profile="demo",
            run_values={
                "robot": "ffw_sg2",
                "scene": "galileo",
                "task": "scene_only",
                "policy_type": "zero_action",
            },
        )
        output = io.StringIO()
        with mock.patch.object(cli.ProfileStore, "resolve", return_value=manifest) as resolve:
            with contextlib.redirect_stdout(output):
                result = cli.main(["infer", "demo", "--dry-run"])

        self.assertEqual(result, 0)
        resolve.assert_called_once_with(
            "demo",
            REGISTRY,
            model_adapter_override=None,
        )
        self.assertIn("--scene galileo", output.getvalue())

    def test_dry_run_does_not_require_prepared_server_state(self):
        manifest = self._model_manifest()
        output = io.StringIO()
        with mock.patch.object(cli, "_resolve_manifest_source", return_value=manifest), mock.patch.object(
            cli,
            "load_server_port",
        ) as load_server_port, contextlib.redirect_stdout(output):
            result = cli.main(["infer", "demo", "--dry-run"])

        self.assertEqual(result, 0)
        load_server_port.assert_not_called()
        self.assertNotIn("--remote_port", output.getvalue())

    def test_inference_loads_server_state_only_when_port_is_missing(self):
        manifest = self._model_manifest()
        with mock.patch.object(cli, "_resolve_manifest_source", return_value=manifest), mock.patch.object(
            cli,
            "load_server_port",
            return_value=61234,
        ) as load_server_port, mock.patch.object(cli, "_exec_workflow") as execute:
            result = cli.main(["infer", "demo"])

        self.assertEqual(result, 0)
        load_server_port.assert_called_once_with(manifest.model.to_resolved_model(REGISTRY))
        forwarded = execute.call_args.args[1]
        self.assertIn("--remote_port", forwarded)
        self.assertEqual(forwarded[forwarded.index("--remote_port") + 1], "61234")

    def test_manifest_or_cli_port_bypasses_server_state(self):
        cases = (
            (self._model_manifest(remote_port=5555), ["infer", "demo"]),
            (self._model_manifest(), ["infer", "demo", "--remote-port", "62000"]),
        )
        for manifest, arguments in cases:
            with self.subTest(arguments=arguments), mock.patch.object(
                cli,
                "_resolve_manifest_source",
                return_value=manifest,
            ), mock.patch.object(cli, "load_server_port") as load_server_port, mock.patch.object(
                cli,
                "_exec_workflow",
            ) as execute:
                result = cli.main(arguments)

            self.assertEqual(result, 0)
            load_server_port.assert_not_called()
            forwarded = execute.call_args.args[1]
            expected_port = "62000" if "--remote-port" in arguments else "5555"
            self.assertEqual(forwarded[forwarded.index("--remote_port") + 1], expected_port)

    def test_profile_catalog_is_available_without_isaac_sim(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = cli.main(["list", "profiles"])

        self.assertEqual(result, 0)
        self.assertIn("ffw_sg2_gr00t", output.getvalue())


if __name__ == "__main__":
    unittest.main()
