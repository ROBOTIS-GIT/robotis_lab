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

"""Tests for metadata-driven model discovery and robot pose loading."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cyclo_arena.catalog import REGISTRY
from cyclo_arena.core.model_resolver import discover_models, resolve_model
from cyclo_arena.core.robot_pose import list_robot_poses, load_robot_pose
from cyclo_arena.core.server_state import load_server_port, write_server_state


def _write_checkpoint(
    checkpoint: Path,
    video_key: str = "cam_left_head",
    model_type: str = "Gr00tN1d6",
    action_horizon: int | None = None,
    nested_processor: bool = False,
) -> None:
    checkpoint.mkdir(parents=True)
    if action_horizon is None:
        action_horizon = 40 if model_type == "Gr00tN1d7" else 16
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": model_type}),
        encoding="utf-8",
    )
    modalities = {
        "video": {"delta_indices": [0], "modality_keys": [video_key]},
        "state": {
            "delta_indices": [0],
            "modality_keys": ["arm_left", "arm_right"],
        },
        "action": {
            "delta_indices": list(range(action_horizon)),
            "modality_keys": ["arm_left", "arm_right"],
            "action_configs": [
                {"rep": "ABSOLUTE"},
                {"rep": "ABSOLUTE"},
            ],
        },
        "language": {
            "delta_indices": [0],
            "modality_keys": ["annotation.human.task_description"],
        },
    }
    processor_directory = checkpoint / "processor" if nested_processor else checkpoint
    processor_directory.mkdir(exist_ok=True)
    (processor_directory / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_kwargs": {
                    "modality_configs": {"new_embodiment": modalities}
                }
            }
        ),
        encoding="utf-8",
    )
    (processor_directory / "statistics.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").touch()


class ModelResolverTest(unittest.TestCase):
    """Verify compatible checkpoints require no Python catalog entry."""

    def test_auto_adapter_resolves_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            checkpoint = Path(temp_directory) / "downloaded_model"
            _write_checkpoint(checkpoint)

            model = resolve_model(
                checkpoint=checkpoint,
                robot="ffw_sg2",
                adapter_name="auto",
                registry=REGISTRY,
            )

        self.assertEqual(model.name, "downloaded_model")
        self.assertEqual(model.adapter.name, "ffw_sg2_gr00t_n16")

    def test_incompatible_camera_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            checkpoint = Path(temp_directory) / "wrong_camera_model"
            _write_checkpoint(checkpoint, video_key="other_camera")

            with self.assertRaisesRegex(AssertionError, "video keys"):
                resolve_model(
                    checkpoint=checkpoint,
                    robot="ffw_sg2",
                    adapter_name="auto",
                    registry=REGISTRY,
                )

    def test_auto_adapter_resolves_n17_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            checkpoint = Path(temp_directory) / "n17_model"
            _write_checkpoint(checkpoint, model_type="Gr00tN1d7")

            model = resolve_model(
                checkpoint=checkpoint,
                robot="ffw_sg2",
                adapter_name="auto",
                registry=REGISTRY,
            )

        self.assertEqual(model.model_type, "Gr00tN1d7")
        self.assertEqual(model.adapter.name, "ffw_sg2_gr00t_n17")
        self.assertEqual(model.adapter.action_horizon, 40)
        self.assertEqual(model.adapter.server_image, "cyclo-gr00t:n1.7")

    def test_n17_training_output_supports_nested_processor_directory(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            checkpoint = Path(temp_directory) / "n17_training_output"
            _write_checkpoint(
                checkpoint,
                model_type="Gr00tN1d7",
                nested_processor=True,
            )

            model = resolve_model(
                checkpoint=checkpoint,
                robot="ffw_sg2",
                adapter_name="auto",
                registry=REGISTRY,
            )

        self.assertEqual(model.adapter.name, "ffw_sg2_gr00t_n17")

    def test_discovery_reports_compatible_and_incompatible_models(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            _write_checkpoint(root / "compatible")
            _write_checkpoint(root / "incompatible", video_key="other_camera")

            models = discover_models(REGISTRY, root)

        self.assertEqual(
            [model.checkpoint.name for model in models],
            ["compatible", "incompatible"],
        )
        self.assertEqual(models[0].compatible_adapters, ("ffw_sg2_gr00t_n16",))
        self.assertEqual(models[1].compatible_adapters, ())

    def test_prepared_server_state_is_shared_by_relative_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            model_root = Path(temp_directory)
            checkpoint = model_root / "downloaded_model"
            state_path = model_root / ".cyclo_arena" / "server.json"
            _write_checkpoint(checkpoint)
            environment = {
                "CYCLO_ARENA_MODEL_ROOT": str(model_root),
                "CYCLO_ARENA_SERVER_STATE": str(state_path),
            }
            with mock.patch.dict(os.environ, environment):
                model = resolve_model(
                    checkpoint=checkpoint,
                    robot="ffw_sg2",
                    adapter_name="auto",
                    registry=REGISTRY,
                )
                write_server_state(model, 61234, "cyclo-gr00t-test")

                self.assertEqual(load_server_port(model), 61234)


class RobotPoseTest(unittest.TestCase):
    """Verify model-independent FFW-SG2 poses are data files."""

    def test_ffw_sg2_pose_catalog(self):
        self.assertEqual(list_robot_poses("ffw_sg2"), ("cartoning", "recycling"))
        pose = load_robot_pose("ffw_sg2", "recycling")

        self.assertEqual(pose.joint_positions["lift_joint"], -0.2)
        self.assertEqual(pose.joint_positions["head_joint1"], 0.5)


if __name__ == "__main__":
    unittest.main()
