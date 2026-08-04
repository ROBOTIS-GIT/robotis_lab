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

"""Tests for Cyclo's thin upstream Arena policy-runner adapter."""

import types
import unittest
from unittest import mock

from cyclo_arena.compat import policy_runner


class CompatiblePolicyRunnerTest(unittest.TestCase):
    """Verify compatibility is installed only inside upstream runner execution."""

    def test_install_occurs_immediately_before_policy_registry_lookup(self):
        events = []

        def original_get_policy_cls(policy_type):
            events.append(("resolve", policy_type))
            return object

        upstream = types.SimpleNamespace(get_policy_cls=original_get_policy_cls)

        def upstream_main():
            events.append(("main", None))
            return upstream.get_policy_cls("example.Policy")

        upstream.main = upstream_main
        with mock.patch.object(policy_runner.importlib, "import_module", return_value=upstream):
            with mock.patch.object(
                policy_runner,
                "install_arena_runtime_compat",
                side_effect=lambda: events.append(("install", None)),
            ):
                result = policy_runner.main()

        self.assertIs(result, object)
        self.assertEqual(
            events,
            [
                ("main", None),
                ("install", None),
                ("resolve", "example.Policy"),
            ],
        )
        self.assertIs(upstream.get_policy_cls, original_get_policy_cls)


if __name__ == "__main__":
    unittest.main()

