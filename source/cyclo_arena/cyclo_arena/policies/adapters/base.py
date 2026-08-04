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

"""Transport-independent robot contract for GR00T inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np


class Gr00tRobotAdapter(ABC):
    """Translate one robot's Arena observations and GR00T actions."""

    @property
    @abstractmethod
    def observation_keys(self) -> list[str]:
        """Return dotted Arena observation keys requested by the server."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Return the action dimension exposed to the Arena environment."""

    @property
    @abstractmethod
    def model_action_horizon(self) -> int:
        """Return the native action horizon read from the checkpoint."""

    @abstractmethod
    def build_policy_observation(
        self,
        observation: Mapping[str, Any],
        task_description: str,
    ) -> dict[str, Any]:
        """Convert one Arena observation into the checkpoint schema."""

    @abstractmethod
    def build_action_chunk(
        self,
        policy_action: Mapping[str, Any],
    ) -> np.ndarray:
        """Convert a checkpoint action dictionary into Arena action chunks."""

    @abstractmethod
    def reset(self) -> None:
        """Clear observation history owned by the adapter."""
