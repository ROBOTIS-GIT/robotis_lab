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

"""Tests for immutable manifests and path-independent named profiles."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cyclo_arena.catalog import REGISTRY
from cyclo_arena.core.config import RunConfig
from cyclo_arena.core.manifest import ResolvedManifest
from cyclo_arena.core.model_resolver import ResolvedModel
from cyclo_arena.core.profile_store import (
    DEFAULT_PROFILE_ID,
    ProfileStore,
    default_profile_roots,
)


class ResolvedManifestTest(unittest.TestCase):
    """Verify version-one run configs retain their existing execution values."""

    def test_v1_run_config_converts_without_changing_run_values(self):
        config = RunConfig.from_mapping({
            "robot": "ffw_sg2",
            "scene": {
                "name": "galileo",
                "robot_position_xyz": [1.0, 2.0, 3.0],
                "options": {"custom": {"indices": [1, 2]}},
            },
            "policy": {"type": "zero_action"},
            "runtime": {"num_steps": 20, "enable_cameras": True},
        })

        expected = config.to_run_values(REGISTRY)
        manifest = ResolvedManifest.from_run_config(config, REGISTRY)

        self.assertEqual(manifest.to_run_values(), expected)
        self.assertEqual(manifest.workflow, "infer")
        self.assertEqual(manifest.config_schema_version, 1)

    def test_manifest_is_deeply_immutable_and_returns_detached_values(self):
        config = RunConfig.from_mapping({
            "robot": "ffw_sg2",
            "scene": {
                "name": "galileo",
                "options": {"custom": {"indices": [1, 2]}},
            },
        })
        manifest = ResolvedManifest.from_run_config(config, REGISTRY)

        with self.assertRaises(TypeError):
            manifest.run_values["scene"] = "kitchen"
        with self.assertRaises(TypeError):
            manifest.run_values["custom"]["indices"] = (3,)
        detached = manifest.to_run_values()
        detached["custom"]["indices"].append(3)

        self.assertEqual(manifest.run_values["custom"]["indices"], (1, 2))

    def test_overrides_return_a_new_manifest(self):
        config = RunConfig.from_mapping({"robot": "ffw_sg2", "scene": "galileo"})
        manifest = ResolvedManifest.from_run_config(config, REGISTRY)

        overridden = manifest.with_run_overrides(num_steps=50, headless=True)

        self.assertIsNone(manifest.run_values["num_steps"])
        self.assertEqual(overridden.run_values["num_steps"], 50)
        self.assertTrue(overridden.run_values["headless"])

    def test_json_round_trip_preserves_the_resolved_plan(self):
        config = RunConfig.from_mapping({"robot": "ffw_sg2", "scene": "galileo"})
        manifest = ResolvedManifest.from_run_config(config, REGISTRY)

        with tempfile.TemporaryDirectory() as temp_directory:
            path = manifest.write(Path(temp_directory) / "manifest.json")
            restored = ResolvedManifest.load(path)

        self.assertEqual(restored.to_mapping(), manifest.to_mapping())
        self.assertEqual(restored.fingerprint, manifest.fingerprint)

    def test_manifest_resolution_does_not_read_runtime_server_state(self):
        config = RunConfig.from_mapping({
            "robot": "ffw_sg2",
            "scene": "robotis_showroom_training",
            "model": {"checkpoint": "/models/showroom_groot", "adapter": "auto"},
        })
        model = ResolvedModel(
            checkpoint=Path("/models/showroom_groot"),
            adapter=REGISTRY.model_adapters["ffw_sg2_gr00t_n17_showroom"],
            model_type="Gr00tN1d7",
        )

        with mock.patch.object(RunConfig, "resolve_model", return_value=model), mock.patch(
            "cyclo_arena.core.server_state.load_server_port",
        ) as load_server_port:
            manifest = ResolvedManifest.from_run_config(config, REGISTRY)

        load_server_port.assert_not_called()
        self.assertIsNone(manifest.run_values["remote_port"])
        self.assertEqual(manifest.model.adapter, "ffw_sg2_gr00t_n17_showroom")


class ProfileStoreTest(unittest.TestCase):
    """Verify profiles resolve by stable names in source and configured layouts."""

    def test_default_profile_captures_current_ffw_sg2_run(self):
        store = ProfileStore()

        self.assertIn(DEFAULT_PROFILE_ID, store.names())
        profile = store.get(DEFAULT_PROFILE_ID)
        config = profile.load()

        self.assertEqual(profile.name, DEFAULT_PROFILE_ID)
        self.assertEqual(config.robot.name, "ffw_sg2")
        self.assertEqual(config.robot.initial_pose, "showroom")
        self.assertEqual(config.scene.name, "robotis_showroom_training")
        self.assertEqual(config.model.checkpoint, "${CYCLO_ARENA_MODEL_ROOT}/showroom_groot")
        self.assertEqual(config.task.description, "Cyclo-Real-Showroom-FFW-SG2-v0")
        self.assertEqual(config.runtime.num_steps, 10000)
        self.assertTrue(config.runtime.enable_cameras)
        self.assertFalse(config.runtime.headless)

    def test_source_profile_lookup_does_not_depend_on_working_directory(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            with mock.patch("pathlib.Path.cwd", return_value=Path(temp_directory)):
                config = ProfileStore().load(DEFAULT_PROFILE_ID)

        self.assertEqual(config.scene.name, "robotis_showroom_training")

    def test_environment_profile_root_supports_installed_layouts(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            profile_path = root / "custom.yaml"
            profile_path.write_text("robot: ffw_sg2\nscene: galileo\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"CYCLO_ARENA_PROFILE_ROOT": str(root)}):
                roots = default_profile_roots()
                config = ProfileStore(roots=(roots[0],)).load("custom")

        self.assertEqual(roots[0], root.resolve())
        self.assertEqual(config.scene.name, "galileo")

    def test_unknown_and_path_traversal_profile_ids_are_rejected(self):
        store = ProfileStore()

        with self.assertRaisesRegex(AssertionError, "Unknown Cyclo Arena profile"):
            store.load("missing")
        with self.assertRaisesRegex(AssertionError, "Invalid profile ID"):
            store.load("../run")
        with self.assertRaisesRegex(AssertionError, "must not include a YAML extension"):
            store.load("ffw_sg2_gr00t.yaml")


if __name__ == "__main__":
    unittest.main()
