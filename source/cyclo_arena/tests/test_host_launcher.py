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
        self.config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "run.yaml"
        )
        self.model = ResolvedModel(
            checkpoint=Path("/models/checkpoint").resolve(),
            adapter=REGISTRY.model_adapters["ffw_sg2_gr00t_n16"],
            model_type="Gr00tN1d6",
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
            model_adapter="ffw_sg2_gr00t_n16",
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
            result = host_launcher.main(
                ["--config", str(self.config_path), "--", "--dry-run"]
            )

        self.assertEqual(result, 0)
        ensure_model.assert_not_called()
        launch.assert_called_once_with(
            "cyclo_lab",
            self.config_path,
            ["--dry-run"],
            model_adapter="ffw_sg2_gr00t_n16",
            remote_port=None,
        )

    def test_runtime_query_prints_checkpoint_selected_image(self):
        output = io.StringIO()
        with mock.patch(
            "cyclo_arena.core.config.RunConfig.resolve_model",
            return_value=self.model,
        ), contextlib.redirect_stdout(output):
            result = host_launcher.main(
                ["--config", str(self.config_path), "--print-server-runtime"]
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            output.getvalue().strip(),
            "cyclo-gr00t:n1.6\t"
            "e29d8fc50b0e4745120ae3fb72447986fe638aa6",
        )


if __name__ == "__main__":
    unittest.main()
