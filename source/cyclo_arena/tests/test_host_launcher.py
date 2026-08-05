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
from cyclo_arena.core.manifest import ResolvedManifest
from cyclo_arena.core.model_resolver import ResolvedModel
from cyclo_arena.core.profile_store import DEFAULT_PROFILE_ID, ProfileStore


class HostLauncherTest(unittest.TestCase):
    """Verify model startup and container launch ordering without Docker."""

    def setUp(self):
        self.config_path = ProfileStore().get(DEFAULT_PROFILE_ID).path
        self.model = ResolvedModel(
            checkpoint=Path("/models/checkpoint").resolve(),
            adapter=REGISTRY.model_adapters["ffw_sg2_gr00t_n17"],
            model_type="Gr00tN1d7",
        )

    @mock.patch.object(host_launcher, "_run")
    def test_matching_server_image_revision_skips_build(self, run):
        run.return_value = mock.Mock(
            returncode=0,
            stdout=f"{self.model.adapter.server_source_revision}\n",
        )

        rebuilt = host_launcher._ensure_server_image(self.model)

        self.assertFalse(rebuilt)
        run.assert_called_once_with(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                '{{ index .Config.Labels "cyclo_arena.gr00t_revision" }}',
                self.model.adapter.server_image,
            ],
            check=False,
            capture_output=True,
        )

    @mock.patch.object(host_launcher, "_launch_in_container", return_value=0)
    @mock.patch.object(host_launcher, "_ensure_model_server")
    @mock.patch.object(host_launcher, "_ensure_cyclo_container")
    @mock.patch.object(host_launcher.shutil, "which", return_value="/usr/bin/docker")
    def test_no_arguments_start_the_default_profile_model(
        self,
        _which,
        ensure_container,
        ensure_model,
        launch,
    ):
        with mock.patch(
            "cyclo_arena.core.config.RunConfig.resolve_model",
            return_value=self.model,
        ) as resolve_model:
            ensure_model.return_value = 61234
            result = host_launcher.main([])

        self.assertEqual(result, 0)
        resolve_model.assert_called_once()
        ensure_container.assert_called_once_with("cyclo_lab")
        ensure_model.assert_called_once_with(
            "cyclo_lab",
            self.model,
            rebuild_image=False,
        )
        launch.assert_called_once()
        container_name, manifest, forwarded_args = launch.call_args.args
        self.assertEqual(container_name, "cyclo_lab")
        self.assertIsInstance(manifest, ResolvedManifest)
        self.assertEqual(manifest.profile, DEFAULT_PROFILE_ID)
        self.assertEqual(manifest.run_values["remote_port"], 61234)
        self.assertEqual(manifest.model.adapter, "ffw_sg2_gr00t_n17")
        self.assertEqual(forwarded_args, [])

    @mock.patch.object(host_launcher, "_launch_in_container", return_value=0)
    @mock.patch.object(host_launcher, "_ensure_model_server")
    @mock.patch.object(host_launcher, "_ensure_cyclo_container")
    @mock.patch.object(host_launcher.shutil, "which", return_value="/usr/bin/docker")
    def test_prepare_rebuild_flag_is_forwarded_to_the_model_server(
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
            result = host_launcher.main([
                "--prepare-only",
                "--rebuild-server-image",
            ])

        self.assertEqual(result, 0)
        ensure_container.assert_called_once_with("cyclo_lab")
        ensure_model.assert_called_once_with(
            "cyclo_lab",
            self.model,
            rebuild_image=True,
        )
        launch.assert_not_called()

    @mock.patch.object(host_launcher, "_ping_server", return_value=True)
    @mock.patch.object(host_launcher, "_create_server_container")
    @mock.patch.object(host_launcher, "_available_port", return_value=61234)
    @mock.patch.object(host_launcher, "_container_status", return_value="running")
    @mock.patch.object(host_launcher, "_stop_inactive_model_servers")
    @mock.patch.object(host_launcher, "_server_container_name", return_value="cyclo-gr00t-test")
    @mock.patch.object(host_launcher, "_ensure_server_image", return_value=True)
    @mock.patch.object(host_launcher, "_run")
    def test_rebuilt_image_recreates_the_server_container(
        self,
        run,
        ensure_image,
        _server_name,
        _stop_inactive,
        _container_status,
        _available_port,
        create_server,
        _ping_server,
    ):
        port = host_launcher._ensure_model_server(
            "cyclo_lab",
            self.model,
            rebuild_image=True,
        )

        self.assertEqual(port, 61234)
        ensure_image.assert_called_once_with(self.model, force_rebuild=True)
        run.assert_called_once_with([
            "docker",
            "rm",
            "--force",
            "cyclo-gr00t-test",
        ])
        create_server.assert_called_once_with(
            self.model,
            "cyclo-gr00t-test",
            61234,
        )

    def test_start_groot_shell_delegates_to_the_host_launcher(self):
        script_path = host_launcher.REPOSITORY_ROOT / "docker" / "container.sh"
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("local -a launcher_args=(--prepare-only)", script)
        self.assertIn("launcher_args+=(--rebuild-server-image)", script)
        self.assertIn("python3 -m cyclo_arena.host_launcher", script)
        self.assertNotIn("build_groot_image()", script)

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
        ) as resolve_model:
            result = host_launcher.main(["--config", str(self.config_path), "--", "--dry-run"])

        self.assertEqual(result, 0)
        resolve_model.assert_called_once()
        ensure_model.assert_not_called()
        launch.assert_called_once()
        container_name, manifest, forwarded_args = launch.call_args.args
        self.assertEqual(container_name, "cyclo_lab")
        self.assertIsInstance(manifest, ResolvedManifest)
        self.assertIsNone(manifest.profile)
        self.assertIsNone(manifest.run_values["remote_port"])
        self.assertEqual(forwarded_args, ["--dry-run"])

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

    @mock.patch.object(host_launcher, "_persist_manifest")
    @mock.patch.object(host_launcher, "_run")
    def test_container_receives_only_the_resolved_manifest(self, run, persist_manifest):
        manifest = ResolvedManifest(
            workflow="infer",
            run_values={"headless": True},
        )
        persist_manifest.return_value = (
            Path("/models/.cyclo_arena/manifests/test.json"),
            Path("/workspace/model/.cyclo_arena/manifests/test.json"),
        )
        run.return_value = mock.Mock(returncode=0)

        result = host_launcher._launch_in_container(
            "cyclo_lab",
            manifest,
            ["--num-steps", "2"],
        )

        self.assertEqual(result, 0)
        command = run.call_args.args[0]
        self.assertIn("--manifest", command)
        self.assertIn("/workspace/model/.cyclo_arena/manifests/test.json", command)
        self.assertIn("--headless", command)
        self.assertNotIn("--config", command)


if __name__ == "__main__":
    unittest.main()
