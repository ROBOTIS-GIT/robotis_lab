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

"""Tests for the transport-independent FFW-SG2 GR00T mapping."""

import unittest

import numpy as np
from cyclo_arena.policies.adapters.ffw_sg2 import (
    FFWSG2Gr00tAdapter,
    FFWSG2ShowroomGr00tAdapter,
)


class FFWSG2Gr00tAdapterTest(unittest.TestCase):
    """Verify model schema conversion without Isaac Sim or GR00T imports."""

    def setUp(self):
        self.modality_configs = {
            "video": {
                "modality_keys": ["cam_left_head"],
                "delta_indices": [-1, 0],
            },
            "state": {
                "modality_keys": ["arm_left", "arm_right"],
                "delta_indices": [-1, 0],
            },
            "action": {
                "modality_keys": ["arm_left", "arm_right"],
                "delta_indices": [0, 1],
            },
            "language": {
                "modality_keys": ["annotation.human.task_description"],
                "delta_indices": [0],
            },
        }

    def test_builds_checkpoint_observation_and_arena_action_chunk(self):
        adapter = FFWSG2Gr00tAdapter(
            self.modality_configs,
            action_repeat=2,
        )
        joint_pos = np.arange(19, dtype=np.float32)[None]
        observation = {
            "policy.joint_pos": joint_pos,
            "camera_obs.cam_head_rgb": np.ones((1, 4, 5, 3), dtype=np.float32),
        }

        policy_observation = adapter.build_policy_observation(
            observation,
            "sort the objects",
        )

        self.assertEqual(adapter.action_dim, 19)
        self.assertEqual(adapter.model_action_horizon, 2)
        self.assertEqual(
            policy_observation["video"]["cam_left_head"].shape,
            (1, 2, 4, 5, 3),
        )
        self.assertEqual(
            policy_observation["state"]["arm_left"].shape,
            (1, 2, 8),
        )
        self.assertEqual(
            policy_observation["language"]["annotation.human.task_description"],
            [["sort the objects"]],
        )
        self.assertEqual(
            policy_observation["video"]["cam_left_head"].dtype,
            np.uint8,
        )

        action_chunk = adapter.build_action_chunk({
            "arm_left": np.zeros((1, 2, 8), dtype=np.float32),
            "arm_right": np.ones((1, 2, 8), dtype=np.float32),
        })

        self.assertEqual(action_chunk.shape, (1, 4, 19))
        np.testing.assert_array_equal(action_chunk[:, 0], action_chunk[:, 1])
        np.testing.assert_array_equal(action_chunk[0, 0, 16:], joint_pos[0, 16:])

    def test_rejects_a_checkpoint_with_the_wrong_robot_schema(self):
        self.modality_configs["state"]["modality_keys"] = ["joint_pos"]

        with self.assertRaisesRegex(AssertionError, "state keys"):
            FFWSG2Gr00tAdapter(self.modality_configs)

    def test_builds_three_camera_mobile_showroom_contract(self):
        modality_configs = {
            "video": {
                "modality_keys": [
                    "cam_left_head",
                    "cam_left_wrist",
                    "cam_right_wrist",
                ],
                "delta_indices": [0],
            },
            "state": {
                "modality_keys": ["arm_left", "arm_right", "odometry"],
                "delta_indices": [0],
            },
            "action": {
                "modality_keys": ["arm_left", "arm_right", "odometry"],
                "delta_indices": [0, 1],
            },
            "language": {
                "modality_keys": ["annotation.human.task_description"],
                "delta_indices": [0],
            },
        }
        adapter = FFWSG2ShowroomGr00tAdapter(
            modality_configs,
            action_repeat=3,
        )
        joint_pos = np.arange(19, dtype=np.float32)[None]
        base_twist = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
        observation = {
            "policy.joint_pos": joint_pos,
            "policy.base_twist": base_twist,
            "camera_obs.cam_head_rgb": np.zeros(
                (1, 4, 5, 3),
                dtype=np.uint8,
            ),
            "camera_obs.cam_wrist_left_rgb": np.ones(
                (1, 4, 5, 3),
                dtype=np.uint8,
            ),
            "camera_obs.cam_wrist_right_rgb": np.full(
                (1, 4, 5, 3),
                2,
                dtype=np.uint8,
            ),
        }

        policy_observation = adapter.build_policy_observation(
            observation,
            "Cyclo-Real-Showroom-FFW-SG2-v0",
        )

        self.assertEqual(adapter.action_dim, 22)
        self.assertEqual(
            adapter.observation_keys,
            [
                "policy.joint_pos",
                "policy.base_twist",
                "camera_obs.cam_head_rgb",
                "camera_obs.cam_wrist_left_rgb",
                "camera_obs.cam_wrist_right_rgb",
            ],
        )
        self.assertEqual(
            set(policy_observation["video"]),
            {"cam_left_head", "cam_left_wrist", "cam_right_wrist"},
        )
        self.assertEqual(
            policy_observation["video"]["cam_left_wrist"].shape,
            (1, 1, 4, 5, 3),
        )
        np.testing.assert_array_equal(
            policy_observation["state"]["odometry"][:, 0],
            base_twist,
        )

        base_action = np.array(
            [[[0.2, 0.0, -0.1], [0.1, 0.0, 0.0]]],
            dtype=np.float32,
        )
        action_chunk = adapter.build_action_chunk({
            "arm_left": np.zeros((1, 2, 8), dtype=np.float32),
            "arm_right": np.ones((1, 2, 8), dtype=np.float32),
            "odometry": base_action,
        })

        self.assertEqual(action_chunk.shape, (1, 6, 22))
        np.testing.assert_array_equal(action_chunk[:, 0], action_chunk[:, 1])
        np.testing.assert_array_equal(action_chunk[:, 1], action_chunk[:, 2])
        np.testing.assert_array_equal(
            action_chunk[0, 0, 16:19],
            joint_pos[0, 16:19],
        )
        np.testing.assert_array_equal(action_chunk[0, 0, 19:], base_action[0, 0])


if __name__ == "__main__":
    unittest.main()
