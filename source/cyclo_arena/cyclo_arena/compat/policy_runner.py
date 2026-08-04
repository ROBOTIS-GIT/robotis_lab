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

"""Launch Arena's policy runner with Cyclo's deferred Isaac Lab compatibility bridge."""

from __future__ import annotations

import importlib
from typing import Any

from cyclo_arena.compat.runtime import install_arena_runtime_compat

UPSTREAM_POLICY_RUNNER = "isaaclab_arena.evaluation.policy_runner"


def main() -> Any:
    """Delegate to Arena's policy runner and install compatibility after app startup."""
    policy_runner = importlib.import_module(UPSTREAM_POLICY_RUNNER)
    original_get_policy_cls = policy_runner.get_policy_cls

    def get_policy_cls_with_compatibility(policy_type: str):
        install_arena_runtime_compat()
        return original_get_policy_cls(policy_type)

    policy_runner.get_policy_cls = get_policy_cls_with_compatibility
    try:
        return policy_runner.main()
    finally:
        policy_runner.get_policy_cls = original_get_policy_cls


if __name__ == "__main__":
    main()
