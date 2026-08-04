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

"""Install the compatibility boundaries required by Cyclo's simulation pins."""

from cyclo_arena.compat.isaac_sim_51 import install_isaac_sim_51_compat
from cyclo_arena.compat.isaaclab_23 import install_isaaclab_23_compat


def install_arena_runtime_compat() -> tuple[str, ...]:
    """Install every required feature-detected runtime compatibility bridge."""
    return (
        *install_isaaclab_23_compat(),
        *install_isaac_sim_51_compat(),
    )

