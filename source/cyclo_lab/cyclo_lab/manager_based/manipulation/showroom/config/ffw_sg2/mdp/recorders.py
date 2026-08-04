"""Recorder helpers for SG2 showroom demonstrations."""

from __future__ import annotations

import contextlib

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import (
    InitialStateRecorderCfg,
    PostStepProcessedActionsRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
    PreStepFlatPolicyObservationsRecorderCfg,
)
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


SHOWROOM_CAMERA_NAMES = ("cam_head", "cam_wrist_left", "cam_wrist_right")


def _profile_time(env, name: str):
    recorder_manager = getattr(env, "recorder_manager", None)
    profiler = getattr(recorder_manager, "profiler", None)
    if profiler is None:
        return contextlib.nullcontext()
    return profiler.time(name)


def camera_image_cpu(env, sensor_name: str, data_type: str = "rgb", sensor_data=None) -> torch.Tensor:
    """Read a camera sensor as a CPU uint8 tensor for HDF5 recording."""
    if sensor_data is None:
        sensor_data = env.scene.sensors[sensor_name].data
    images = sensor_data.output[data_type].detach().to(device="cpu", copy=True).contiguous()

    if data_type == "rgb" and images.dtype != torch.uint8:
        if images.numel() > 0 and float(images.max()) <= 1.0:
            images = images * 255.0
        images = images.clamp(0, 255).to(torch.uint8)
    return images


class PreStepShowroomCameraObservationsRecorder(RecorderTerm):
    """Record showroom camera observations without putting images in policy observations."""

    cfg: "PreStepShowroomCameraObservationsRecorderCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._cached_images: dict[str, torch.Tensor] = {}
        self._step_count = 0

    def record_post_reset(self, env_ids):
        self._cached_images.clear()
        self._step_count = 0
        return None, None

    def record_pre_step(self):
        observations = {}
        capture_frame = self._step_count % self.cfg.capture_interval_steps == 0
        for camera_name in self.cfg.camera_names:
            if camera_name not in self._env.scene.sensors:
                continue
            with _profile_time(self._env, f"recorder_camera_{camera_name}"):
                image = self._cached_images.get(camera_name)
                if capture_frame or image is None:
                    sensor_data = self._env.scene.sensors[camera_name].data
                    with _profile_time(self._env, f"recorder_camera_copy_{camera_name}"):
                        image = camera_image_cpu(self._env, camera_name, sensor_data=sensor_data)
                    self._cached_images[camera_name] = image
                observations[camera_name] = image
        self._step_count += 1
        return "obs", observations


@configclass
class PreStepShowroomCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the showroom camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepShowroomCameraObservationsRecorder
    camera_names: tuple[str, ...] = SHOWROOM_CAMERA_NAMES
    capture_interval_steps: int = 2
    owned_cpu_tensor_keys: tuple[str, ...] = tuple(f"obs/{name}" for name in SHOWROOM_CAMERA_NAMES)


@configclass
class ShowroomRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder layout for SG2 showroom datasets.

    Policy observations stay low-dimensional. Camera frames are recorded by a
    dedicated term so IsaacLab's observation manager does not cache image tensors
    in ``obs_buf``.
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()
    record_pre_step_camera_observations = PreStepShowroomCameraObservationsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()
