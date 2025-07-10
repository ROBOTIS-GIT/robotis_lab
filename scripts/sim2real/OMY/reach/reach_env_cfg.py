# robotis_lab/sim2real/robot/OMY/reach/config.py

import math
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path


class ReachEnvConfig:
    def __init__(self, model_dir: str):
        self.joint_names = [f"joint{i}" for i in range(1, 7)]
        self.step_size = 1.0 / 1000  # 1000Hz
        self.trajectory_time_from_start = 1.0/20 # 20Hz
         # seconds
        self.send_command_interval = 3.0 # seconds

        self.joint_state_topic = "/arm_controller/controller_state"
        self.joint_trajectory_topic = "/leader/joint_trajectory"

        repo_root = Path(__file__).resolve().parents[4]
        self.policy_path = repo_root / "logs/rsl_rl/reach_omy" / model_dir / "exported/policy.pt"
        self.env_yaml_path = repo_root / "logs/rsl_rl/reach_omy" / model_dir / "params/env.yaml"


    def sample_random_pose(self) -> np.ndarray:
        """Return a random 6D target pose: [x, y, z, qw, qx, qy, qz]."""
        pos = np.random.uniform([0.25, -0.2, 0.3], [0.45, 0.2, 0.45])
        roll = np.random.uniform(-math.pi / 4, math.pi / 4)
        pitch = 0.0 
        yaw = np.random.uniform(math.pi / 4, math.pi * 3 / 4)
        quat = Rotation.from_euler("zyx", [yaw, pitch, roll]).as_quat()  # [x, y, z, w]
        return np.concatenate([pos, [quat[3], quat[0], quat[1], quat[2]]]) # [x, y, z, qw, qx, qy, qz]
