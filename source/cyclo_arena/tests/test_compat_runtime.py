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

"""Tests for the aggregate Cyclo Arena runtime compatibility installer."""

import unittest
from unittest import mock

from cyclo_arena.compat import runtime


class RuntimeCompatibilityTest(unittest.TestCase):
    """Verify every narrow compatibility boundary is installed in order."""

    def test_combines_installed_feature_names(self):
        with mock.patch.object(runtime, "install_isaaclab_23_compat", return_value=("lab",)):
            with mock.patch.object(runtime, "install_isaac_sim_51_compat", return_value=("sim",)):
                installed = runtime.install_arena_runtime_compat()

        self.assertEqual(installed, ("lab", "sim"))


if __name__ == "__main__":
    unittest.main()

