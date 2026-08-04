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

"""Tests for host-side Cyclo Arena orchestration decisions."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

from cyclo_arena import host_launcher
from cyclo_arena.catalog import REGISTRY
from cyclo_arena.core.model_resolver import ResolvedModel


class HostLauncherTest(unittest.TestCase):
    """Verify model startup and container launch ordering without Docker."""

    def setUp(self):
        self.config_path = Path(__file__).resolve().parents[1] / "configs" / "run.yaml"
        self.model = ResolvedModel(
            checkpoint=Path("/models/checkpoint").resolve(),
            adapter=REGISTRY.model_adapters["ffw_sg2_gr00t_n17"],
            model_type="Gr00tN1d7",
        )

    @mock.patch.object(host_launcher, "_launch_in_container", return_value=0)
    @mock.patch.object(host_launcher, "_ensure_model_server")
    @mock.patch.object(host_launcher, "_ensure_cyclo_container")
    @mock.patch.object(host_launcher.shutil, "which", return_value="/usr/bin/docker")
    def test_default_run_starts_selected_model(
        self,
        _which,
        ensure_container,
        ensure_model,
        launch,
    ):
        with mock.patch(
            "cyclo_arena.core.config.RunConfig.resolve_model",
            return_value=self.model,
        ):
            ensure_model.return_value = 61234
            result = host_launcher.main(["--config", str(self.config_path)])

        self.assertEqual(result, 0)
        ensure_container.assert_called_once_with("cyclo_lab")
        ensure_model.assert_called_once_with("cyclo_lab", self.model)
        launch.assert_called_once_with(
            "cyclo_lab",
            self.config_path,
            [],
            model_adapter="ffw_sg2_gr00t_n17",
            remote_port=61234,
        )

    @mock.patch.object(host_launcher, "_launch_in_container", return_value=0)
    @mock.patch.object(host_launcher, "_ensure_model_server")
    @mock.patch.object(host_launcher, "_ensure_cyclo_container")
    @mock.patch.object(host_launcher.shutil, "which", return_value="/usr/bin/docker")
    def test_dry_run_does_not_start_model(
        self,
        _which,
        _ensure_container,
        ensure_model,
        launch,
    ):
        with mock.patch(
            "cyclo_arena.core.config.RunConfig.resolve_model",
            return_value=self.model,
        ):
            result = host_launcher.main(["--config", str(self.config_path), "--", "--dry-run"])

        self.assertEqual(result, 0)
        ensure_model.assert_not_called()
        launch.assert_called_once_with(
            "cyclo_lab",
            self.config_path,
            ["--dry-run"],
            model_adapter="ffw_sg2_gr00t_n17",
            remote_port=None,
        )

    def test_runtime_query_prints_checkpoint_selected_image(self):
        output = io.StringIO()
        with mock.patch(
            "cyclo_arena.core.config.RunConfig.resolve_model",
            return_value=self.model,
        ), contextlib.redirect_stdout(output):
            result = host_launcher.main(["--config", str(self.config_path), "--print-server-runtime"])

        self.assertEqual(result, 0)
        self.assertEqual(
            output.getvalue().strip(),
            "cyclo-gr00t:n1.7\t23ace64f17aa5015259b8609d371eb61a357c776",
        )

    @mock.patch.object(
        host_launcher,
        "_cyclo_arena_fingerprint",
        return_value="source123",
    )
    @mock.patch.object(host_launcher, "_arena_revision", return_value="abc123")
    @mock.patch.object(host_launcher, "_huggingface_root")
    @mock.patch.object(host_launcher, "_run")
    def test_server_uses_the_mounted_upstream_arena_protocol(
        self,
        run,
        huggingface_root,
        _arena_revision,
        _source_fingerprint,
    ):
        run.return_value = mock.Mock(returncode=0)
        huggingface_root.return_value = Path("/models")

        host_launcher._create_server_container(
            self.model,
            "cyclo-gr00t-test",
            61234,
        )

        docker_run = run.call_args_list[-1].args[0]
        shell_command = docker_run[-1]
        self.assertIn(
            "isaaclab_arena.remote_policy.remote_policy_server_runner",
            shell_command,
        )
        self.assertIn(
            "cyclo_arena.policies.gr00t_server.CycloGr00tServerSidePolicy",
            shell_command,
        )
        self.assertIn("cyclo_arena.arena_revision=abc123", docker_run)
        self.assertIn("cyclo_arena.source_fingerprint=source123", docker_run)
        self.assertIn(
            f"{host_launcher.REPOSITORY_ROOT / 'third_party' / 'IsaacLab-Arena'}:"
            f"{host_launcher.SERVER_ISAACLAB_ARENA_ROOT}:ro",
            docker_run,
        )


if __name__ == "__main__":
    unittest.main()
