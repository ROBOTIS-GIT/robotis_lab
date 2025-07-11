#!/usr/bin/env python3

# Copyright 2025 ROBOTIS CO., LTD.
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
# Author: Taehyeong Kim

# Modified from original code by Louis Le Lay
# Original source: https://github.com/louislelay/kinova_isaaclab_sim2real

import io

import numpy as np
import torch

from .config_loader import parse_env_config, get_physics_properties, get_robot_joint_properties, get_action_scale

class PolicyExecutor:
    """A controller that loads and executes a policy from a file."""

    def __init__(self) -> None:
        pass

    def load_policy(self, policy_file_path, policy_env_path, joint_names) -> None:
        """
        Load a TorchScript *policy* plus its environment metadata.

        Parameters
        ----------
        model_path : str | Path
            Path to a ``.pt`` / ``.pth`` TorchScript file.
        env_path   : str | Path
            Path to the corresponding ``env.yaml``.
        """

        print("\n=== Policy Loading ===")
        print(f"{'Model path:':<18} {policy_file_path}")
        print(f"{'Environment path:':<18} {policy_env_path}")

        with open(policy_file_path, "rb") as f:
            file = io.BytesIO(f.read())
        self.policy = torch.jit.load(file)
        self.policy_env_params = parse_env_config(policy_env_path)

        self.decimation, self.dt, self.render_interval = get_physics_properties(self.policy_env_params)

        print("\n--- Physics properties ---")
        print(f"{'Decimation:':<18} {self.decimation}")
        print(f"{'Timestep (dt):':<18} {self.dt}")
        print(f"{'Render interval:':<18} {self.render_interval}")

        self.action_scale = get_action_scale(self.policy_env_params)

        print(f"{'Action scale:':<18} {self.action_scale}")

        self.stiffness, self.damping, self.default_pos, self.default_vel = get_robot_joint_properties(
            self.policy_env_params, joint_names
        )
        self.num_joints = len(joint_names)

        print("\n--- Robot joint properties ---")
        print(f"{'Number of joints:':<18} {self.num_joints}")
        print(f"{'Stifness:':<18} {self.stiffness}")
        print(f"{'Damping:':<18} {self.damping}")
        print(f"{'Default position:':<18} {self.default_pos}")
        print(f"{'Default velocity:':<18} {self.default_vel}")

        print("\n=== Policy Loaded ===\n")

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """

        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def compute_observation(self) -> NotImplementedError:
        """Build an observation, must be overridden."""

        raise NotImplementedError(
            "Compute observation need to be implemented, expects np.ndarray in the structure specified by env yaml"
        )

    def forward(self) -> NotImplementedError:
        """Return the next command, must be overridden."""

        raise NotImplementedError(
            "Forward needs to be implemented to compute and apply robot control from observations"
        )
