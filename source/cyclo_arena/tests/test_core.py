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

"""Tests for Cyclo Arena composition contracts and run configuration."""

import unittest
from dataclasses import replace
from pathlib import Path

from cyclo_arena.catalog import REGISTRY
from cyclo_arena.core.config import RunConfig
from cyclo_arena.core.profile_store import DEFAULT_PROFILE_ID, ProfileStore


class RegistryTest(unittest.TestCase):
    """Verify static compositions can be resolved without importing Isaac Sim."""

    def test_galileo_plan_resolves_independent_components(self):
        plan = REGISTRY.compose("ffw_sg2", "galileo", "scene_only")

        self.assertEqual(plan.robot.name, "ffw_sg2")
        self.assertEqual(plan.scene.name, "galileo")
        self.assertEqual(plan.task.name, "scene_only")
        self.assertEqual(plan.placement.position_xyz, (-0.0955, -1.107, 0.0))
        self.assertEqual(plan.placement.yaw, -1.78)

    def test_component_tuple_has_a_deterministic_name(self):
        plan = REGISTRY.compose(
            robot="ffw_sg2",
            scene="robotis_showroom",
            task="scene_only",
        )

        self.assertEqual(
            plan.name,
            "cyclo_ffw_sg2_robotis_showroom_scene_only",
        )


class RunConfigTest(unittest.TestCase):
    """Verify strict config parsing and registry validation."""

    def test_zero_action_config_maps_to_run_contract(self):
        config = RunConfig.from_mapping({
            "robot": "ffw_sg2",
            "scene": "galileo",
            "policy": {"type": "zero_action"},
            "runtime": {"num_steps": 1800, "enable_cameras": True},
        })
        values = config.to_run_values(REGISTRY)

        self.assertEqual(values["robot"], "ffw_sg2")
        self.assertEqual(values["scene"], "galileo")
        self.assertEqual(values["task"], "scene_only")
        self.assertEqual(values["embodiment"], "ffw_sg2_abs_joint_pos")
        self.assertEqual(values["policy_type"], "zero_action")
        self.assertEqual(values["num_steps"], 1800)
        self.assertTrue(values["enable_cameras"])

    def test_unknown_config_keys_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "Unknown robot keys"):
            RunConfig.from_mapping({
                "robot": {"name": "ffw_sg2", "unsupported": True},
                "scene": {"name": "galileo"},
            })

    def test_default_profile_selects_robot_scene_and_model(self):
        config = ProfileStore().load(DEFAULT_PROFILE_ID)
        values = config.to_run_values(
            REGISTRY,
            model_adapter_override="ffw_sg2_gr00t_n17_showroom",
        )

        self.assertEqual(values["robot"], "ffw_sg2")
        self.assertEqual(values["scene"], config.scene.name)
        self.assertEqual(
            values["embodiment"],
            "ffw_sg2_mobile_abs_joint_pos",
        )
        self.assertEqual(values["robot_pose"], "showroom")
        self.assertIsNone(values["remote_port"])
        self.assertTrue(values["enable_cameras"])

    def test_selected_model_composes_with_every_ffw_sg2_scene(self):
        config = ProfileStore().load(DEFAULT_PROFILE_ID)

        for scene_name in REGISTRY.scenes:
            with self.subTest(scene=scene_name):
                scene_config = replace(config.scene, name=scene_name)
                values = replace(config, scene=scene_config).to_run_values(
                    REGISTRY,
                    model_adapter_override="ffw_sg2_gr00t_n17_showroom",
                )
                self.assertEqual(values["robot"], "ffw_sg2")
                self.assertEqual(values["scene"], scene_name)

    def test_default_profile_documents_every_registered_selection(self):
        profile_path = ProfileStore().get(DEFAULT_PROFILE_ID).path
        documentation = profile_path.read_text(encoding="utf-8")

        for selection in (
            *REGISTRY.robots,
            *REGISTRY.scenes,
        ):
            with self.subTest(selection=selection):
                self.assertIn(selection, documentation)

    def test_default_profile_documents_every_supported_config_field(self):
        profile_path = ProfileStore().get(DEFAULT_PROFILE_ID).path
        documentation = profile_path.read_text(encoding="utf-8")
        supported_fields = (
            "schema_version",
            "robot",
            "name",
            "embodiment",
            "initial_pose",
            "head_position",
            "lift_position",
            "scene",
            "robot_position_xyz",
            "robot_yaw",
            "options",
            "kitchen_layout",
            "kitchen_style",
            "task",
            "description",
            "model",
            "checkpoint",
            "adapter",
            "policy",
            "type",
            "remote_host",
            "remote_port",
            "remote_timeout_ms",
            "runtime",
            "num_steps",
            "num_episodes",
            "num_envs",
            "device",
            "seed",
            "enable_cameras",
            "headless",
        )

        for field in supported_fields:
            with self.subTest(field=field):
                self.assertIn(f"{field}:", documentation)

    def test_legacy_run_yaml_is_not_a_second_default_source(self):
        legacy_config = Path(__file__).resolve().parents[1] / "configs" / "run.yaml"

        self.assertFalse(legacy_config.exists())

    def test_model_and_policy_cannot_be_selected_together(self):
        with self.assertRaisesRegex(AssertionError, "mutually exclusive"):
            RunConfig.from_mapping({
                "robot": "ffw_sg2",
                "scene": "robotis_showroom",
                "model": "/models/downloaded_checkpoint",
                "policy": {"type": "zero_action"},
            })


if __name__ == "__main__":
    unittest.main()
