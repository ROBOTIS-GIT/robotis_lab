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

"""Tests for the simulator-independent Cyclo Arena workflow registry."""

import unittest

from cyclo_arena.core.workflows import (
    WORKFLOWS,
    WorkflowKind,
    WorkflowReadiness,
    WorkflowRegistry,
    WorkflowSpec,
    resolve_workflow,
)


class WorkflowRegistryTest(unittest.TestCase):
    """Verify canonical workflow metadata and backward-compatible lookup."""

    def test_current_and_planned_workflows_are_registered(self):
        self.assertEqual(
            set(WORKFLOWS),
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

    def test_inference_aliases_resolve_to_one_spec(self):
        infer = WORKFLOWS["infer"]

        for alias in ("run", "inference", "policy"):
            with self.subTest(alias=alias):
                self.assertIs(WORKFLOWS[alias], infer)
                self.assertEqual(WORKFLOWS.canonical_name(alias), "infer")
        self.assertIs(resolve_workflow("run"), infer)

    def test_legacy_mimic_commands_remain_aliases(self):
        self.assertIs(WORKFLOWS["annotate"], WORKFLOWS["mimic-annotate"])
        self.assertIs(WORKFLOWS["generate"], WORKFLOWS["mimic-generate"])
        self.assertEqual(WORKFLOWS.canonical_name("mimic.annotate"), "mimic-annotate")
        self.assertEqual(WORKFLOWS.canonical_name("mimic.generate"), "mimic-generate")

    def test_launch_kinds_and_upstream_targets_are_explicit(self):
        self.assertEqual(WORKFLOWS["infer"].kind, WorkflowKind.MODULE)
        self.assertEqual(
            WORKFLOWS["infer"].upstream_target,
            "isaaclab_arena.evaluation.policy_runner",
        )
        self.assertEqual(
            WORKFLOWS["infer"].launcher_target,
            "cyclo_arena.compat.policy_runner",
        )
        self.assertEqual(WORKFLOWS["infer"].executable_target, "cyclo_arena.compat.policy_runner")
        self.assertEqual(WORKFLOWS["rl-train"].kind, WorkflowKind.SCRIPT)
        self.assertEqual(WORKFLOWS["gr00t-server"].kind, WorkflowKind.SHELL)
        self.assertEqual(
            WORKFLOWS["test"].default_args,
            ("-q", "third_party/IsaacLab-Arena/isaaclab_arena/tests"),
        )

    def test_readiness_and_requirements_are_machine_readable(self):
        self.assertTrue(WORKFLOWS["infer"].is_ready)
        self.assertTrue(WORKFLOWS["teleop"].is_supported)
        self.assertFalse(WORKFLOWS["teleop"].is_ready)
        self.assertEqual(WORKFLOWS["teleop"].readiness, WorkflowReadiness.REQUIRES_SETUP)
        self.assertIn(
            "teleop_retargeter",
            {requirement.name for requirement in WORKFLOWS["teleop"].requirements},
        )
        for name in ("convert", "train"):
            with self.subTest(name=name):
                spec = WORKFLOWS[name]
                self.assertEqual(spec.readiness, WorkflowReadiness.UNSUPPORTED)
                self.assertFalse(spec.is_supported)
                self.assertIsNone(spec.upstream_target)
                self.assertTrue(spec.readiness_detail)

    def test_unknown_workflow_has_an_actionable_error(self):
        with self.assertRaisesRegex(KeyError, "Unknown workflow"):
            WORKFLOWS.resolve("missing")

    def test_registry_rejects_name_and_alias_collisions(self):
        first = WorkflowSpec(
            name="first",
            description="First workflow.",
            kind=WorkflowKind.MODULE,
            upstream_target="example.first",
            aliases=("shared",),
        )
        second = WorkflowSpec(
            name="second",
            description="Second workflow.",
            kind=WorkflowKind.MODULE,
            upstream_target="example.second",
            aliases=("shared",),
        )

        with self.assertRaisesRegex(AssertionError, "already registered"):
            WorkflowRegistry((first, second))

    def test_unsupported_workflow_cannot_expose_a_target(self):
        with self.assertRaisesRegex(AssertionError, "must not expose"):
            WorkflowSpec(
                name="future",
                description="Future workflow.",
                kind=WorkflowKind.MODULE,
                upstream_target="example.future",
                readiness=WorkflowReadiness.UNSUPPORTED,
                readiness_detail="Not implemented.",
            )


if __name__ == "__main__":
    unittest.main()
