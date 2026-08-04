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

"""Tests for Cyclo Arena's host-side submodule contracts."""

import subprocess
import unittest
from pathlib import Path
from unittest import mock

from cyclo_arena import doctor


class GitQueryTest(unittest.TestCase):
    """Verify Git metadata queries fail safely without mutating a checkout."""

    @mock.patch.object(doctor.subprocess, "run")
    def test_run_git_uses_safe_directory_and_returns_stdout(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout="revision\n", stderr="")
        root = Path("/workspace/cyclo_lab")

        result = doctor._run_git(root, "rev-parse", "HEAD")

        self.assertEqual(result, "revision")
        run.assert_called_once_with(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    @mock.patch.object(doctor.subprocess, "run", side_effect=FileNotFoundError)
    def test_run_git_returns_none_when_git_is_unavailable(self, _run):
        self.assertIsNone(doctor._run_git(Path("/workspace/cyclo_lab"), "rev-parse", "HEAD"))

    @mock.patch.object(doctor, "_run_git")
    def test_gitlink_entry_reads_the_committed_tree(self, run_git):
        superproject = Path("/workspace/cyclo_lab")
        submodule = superproject / "third_party" / "IsaacLab-Arena"
        run_git.return_value = "160000 commit aad4f25f44cc439c363c2b60f1fd170dfd10d2a5\tthird_party/IsaacLab-Arena"

        entry = doctor._gitlink_entry(superproject, submodule)

        self.assertEqual(
            entry,
            ("160000", "commit", "aad4f25f44cc439c363c2b60f1fd170dfd10d2a5"),
        )
        run_git.assert_called_once_with(
            superproject,
            "ls-tree",
            "HEAD",
            "--",
            "third_party/IsaacLab-Arena",
        )


class PinnedSubmoduleTest(unittest.TestCase):
    """Verify initialized submodules match the superproject gitlinks."""

    @mock.patch.object(doctor, "_submodule_commit", return_value="abc123")
    @mock.patch.object(doctor, "_gitlink_entry", return_value=("160000", "commit", "abc123"))
    def test_matching_gitlink_is_accepted(self, _entry, _commit):
        failures = []

        doctor._check_pinned_submodule(
            "Arena",
            Path("/workspace/cyclo_lab"),
            Path("/workspace/cyclo_lab/third_party/IsaacLab-Arena"),
            failures,
        )

        self.assertEqual(failures, [])

    @mock.patch.object(doctor, "_submodule_commit")
    @mock.patch.object(doctor, "_gitlink_entry", return_value=("040000", "tree", "abc123"))
    def test_non_gitlink_entry_is_rejected(self, _entry, commit):
        failures = []

        doctor._check_pinned_submodule(
            "Arena",
            Path("/workspace/cyclo_lab"),
            Path("/workspace/cyclo_lab/third_party/IsaacLab-Arena"),
            failures,
        )

        self.assertEqual(
            failures,
            [
                "Arena must be a 160000 gitlink, found mode=040000 type=tree: "
                "/workspace/cyclo_lab/third_party/IsaacLab-Arena"
            ],
        )
        commit.assert_not_called()

    @mock.patch.object(doctor, "_submodule_commit", return_value="working456")
    @mock.patch.object(doctor, "_gitlink_entry", return_value=("160000", "commit", "pinned123"))
    def test_checkout_drift_from_gitlink_is_rejected(self, _entry, _commit):
        failures = []

        doctor._check_pinned_submodule(
            "Isaac Lab",
            Path("/workspace/cyclo_lab"),
            Path("/workspace/cyclo_lab/third_party/IsaacLab"),
            failures,
        )

        self.assertEqual(
            failures,
            ["Expected Isaac Lab gitlink pinned123, found working456"],
        )


class ArenaGitmodulesTest(unittest.TestCase):
    """Verify Arena is sourced from the official compatibility branch."""

    @mock.patch.object(Path, "read_text")
    def test_gitmodule_values_parse_standard_git_config(self, read_text):
        read_text.return_value = """
[submodule "third_party/IsaacLab-Arena"]
    path = third_party/IsaacLab-Arena
    url = https://github.com/isaac-sim/IsaacLab-Arena.git
    branch = feature/arena_v0.2_on_lab_2.3
"""

        values = doctor._gitmodule_values(
            Path("/workspace/cyclo_lab/.gitmodules"),
            doctor.ARENA_SUBMODULE_NAME,
        )

        self.assertEqual(
            values,
            {
                "path": doctor.ARENA_SUBMODULE_PATH.as_posix(),
                "url": doctor.EXPECTED_ARENA_URL,
                "branch": doctor.EXPECTED_ARENA_BRANCH,
            },
        )

    @mock.patch.object(doctor, "_gitmodule_values")
    def test_official_arena_contract_is_accepted(self, values):
        values.return_value = {
            "path": doctor.ARENA_SUBMODULE_PATH.as_posix(),
            "url": doctor.EXPECTED_ARENA_URL,
            "branch": doctor.EXPECTED_ARENA_BRANCH,
        }
        failures = []

        doctor._check_arena_gitmodule(Path("/workspace/cyclo_lab"), failures)

        self.assertEqual(failures, [])

    @mock.patch.object(doctor, "_gitmodule_values")
    def test_nonofficial_url_and_branch_are_rejected(self, values):
        values.return_value = {
            "path": doctor.ARENA_SUBMODULE_PATH.as_posix(),
            "url": "https://example.com/fork/IsaacLab-Arena.git",
            "branch": "main",
        }
        failures = []

        doctor._check_arena_gitmodule(Path("/workspace/cyclo_lab"), failures)

        self.assertEqual(
            failures,
            [
                (
                    "Expected Arena .gitmodules url="
                    f"{doctor.EXPECTED_ARENA_URL}, found https://example.com/fork/IsaacLab-Arena.git"
                ),
                f"Expected Arena .gitmodules branch={doctor.EXPECTED_ARENA_BRANCH}, found main",
            ],
        )


class DependencyContractTest(unittest.TestCase):
    """Verify dependency checks retain Zenoh's fixed revision contract."""

    @mock.patch.object(doctor, "_check_submodule")
    @mock.patch.object(doctor, "_check_arena_gitmodule")
    @mock.patch.object(doctor, "_check_pinned_submodule")
    def test_dependency_contract_dispatch(self, pinned, arena_config, fixed):
        failures = []

        doctor._check_dependency_contracts(failures)

        self.assertEqual(
            pinned.call_args_list,
            [
                mock.call("Isaac Lab", doctor.CYCLO_LAB_ROOT, doctor.ISAACLAB_ROOT, failures),
                mock.call("Arena", doctor.CYCLO_LAB_ROOT, doctor.ARENA_ROOT, failures),
            ],
        )
        arena_config.assert_called_once_with(doctor.CYCLO_LAB_ROOT, failures)
        fixed.assert_called_once_with(
            "Zenoh ROS2 SDK",
            doctor.ZENOH_SDK_ROOT,
            doctor.EXPECTED_ZENOH_SDK_COMMIT,
            failures,
        )


if __name__ == "__main__":
    unittest.main()
