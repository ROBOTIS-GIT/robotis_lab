#!/usr/bin/env python3
"""
policy_controller.py
--------------------

Minimal wrapper to load a TorchScript policy and expose helpers to:

* build observations  – implemented in subclasses
* compute actions     – via `_compute_action()`

The class also extracts physics and robot-joint parameters from the
`env.yaml` that accompanies each policy.

Sub-classes must set `self.dof_names` before calling `load_policy`
(in `__init__`) and implement:

* `_compute_observation()`
* `forward()`

Author: Louis Le Lay
"""

import io

import numpy as np
import torch

from utils.config_loader import parse_env_config, get_physics_properties, get_robot_joint_properties

class PolicyController:
    """A controller that loads and executes a policy from a file."""

    def __init__(self) -> None:
        pass

    def load_policy(self, policy_file_path, policy_env_path) -> None:
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

        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)

        print("\n--- Physics properties ---")
        print(f"{'Decimation:':<18} {self._decimation}")
        print(f"{'Timestep (dt):':<18} {self._dt}")
        print(f"{'Render interval:':<18} {self.render_interval}")

        self._max_effort, self._max_vel, self._stiffness, self._damping, self.default_pos, self.default_vel = get_robot_joint_properties(
            self.policy_env_params, self.dof_names
        )
        self.num_joints = len(self.dof_names)

        print("\n--- Robot joint properties ---")
        print(f"{'Number of joints:':<18} {self.num_joints}")
        print(f"{'Max effort:':<18} {self._max_effort}")
        print(f"{'Max velocity:':<18} {self._max_vel}")
        print(f"{'Stifness:':<18} {self._stiffness}")
        print(f"{'Damping:':<18} {self._damping}")
        print(f"{'Default position:':<18} {self.default_pos}")
        print(f"{'Default velocity:':<18} {self.default_vel}")

        print("\n=== Policy Loaded ===\n")

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
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

    def _compute_observation(self) -> NotImplementedError:
        """Build an observation, must be overridden."""

        raise NotImplementedError(
            "Compute observation need to be implemented, expects np.ndarray in the structure specified by env yaml"
        )

    def forward(self) -> NotImplementedError:
        """Return the next command, must be overridden."""

        raise NotImplementedError(
            "Forward needs to be implemented to compute and apply robot control from observations"
        )