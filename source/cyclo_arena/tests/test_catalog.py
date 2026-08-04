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

"""Tests for the static Cyclo Arena integration catalog."""

import unittest

from cyclo_arena.catalog import FFW_SG2_EMBODIMENTS, REGISTRY


class CatalogTest(unittest.TestCase):
    """Verify the FFW-SG2 environment and workflow surface."""

    def test_ffw_sg2_has_pose_independent_embodiments(self):
        robot = REGISTRY.robots["ffw_sg2"]

        self.assertEqual(
            FFW_SG2_EMBODIMENTS,
            (
                "ffw_sg2_abs_joint_pos",
                "ffw_sg2_mobile_abs_joint_pos",
            ),
        )
        self.assertEqual(robot.embodiments, FFW_SG2_EMBODIMENTS)
        self.assertEqual(robot.default_embodiment, "ffw_sg2_abs_joint_pos")

    def test_every_scene_composes_without_registered_environment_aliases(self):
        for scene in REGISTRY.scenes:
            with self.subTest(scene=scene):
                plan = REGISTRY.compose("ffw_sg2", scene, "scene_only")
                self.assertEqual(plan.robot.name, "ffw_sg2")
                self.assertEqual(plan.scene.name, scene)
                self.assertEqual(plan.task.name, "scene_only")

    def test_expected_workflows_are_exposed(self):
        self.assertEqual(
            set(REGISTRY.workflows),
            {
                "infer",
                "evaluate",
                "teleop",
                "record",
                "replay",
                "mimic-annotate",
                "mimic-generate",
                "serve",
                "rl-train",
                "gr00t-server",
                "test",
                "convert",
                "train",
            },
        )
        self.assertIs(REGISTRY.workflows["policy"], REGISTRY.workflows["infer"])

    def test_components_are_registered_separately(self):
        self.assertEqual(set(REGISTRY.robots), {"ffw_sg2"})
        self.assertEqual(len(REGISTRY.scenes), 12)
        self.assertEqual(set(REGISTRY.tasks), {"scene_only"})

    def test_ffw_sg2_gr00t_adapter_is_registered_without_models(self):
        self.assertEqual(
            set(REGISTRY.model_adapters),
            {
                "ffw_sg2_gr00t_n17",
                "ffw_sg2_gr00t_n17_showroom",
            },
        )
        n17_adapter = REGISTRY.model_adapters["ffw_sg2_gr00t_n17"]
        showroom_adapter = REGISTRY.model_adapters["ffw_sg2_gr00t_n17_showroom"]
        self.assertEqual(n17_adapter.robot, "ffw_sg2")
        self.assertEqual(n17_adapter.model_types, ("Gr00tN1d7",))
        self.assertEqual(n17_adapter.server_image, "cyclo-gr00t:n1.7")
        self.assertEqual(
            showroom_adapter.embodiment,
            "ffw_sg2_mobile_abs_joint_pos",
        )
        self.assertEqual(showroom_adapter.action_horizon, 16)
        self.assertEqual(showroom_adapter.action_repeat, 3)
        self.assertEqual(showroom_adapter.action_chunk_length, 48)

    def test_gr00t_client_is_the_upstream_arena_policy(self):
        policy = REGISTRY.policies["gr00t_closedloop"]

        self.assertEqual(
            policy.runtime_target,
            "isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy",
        )


if __name__ == "__main__":
    unittest.main()
