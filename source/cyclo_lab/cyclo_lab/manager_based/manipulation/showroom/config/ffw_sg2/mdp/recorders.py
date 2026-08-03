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


def camera_image_cpu(env, sensor_name: str, data_type: str = "rgb") -> torch.Tensor:
    """Read a camera sensor as a CPU uint8 tensor for HDF5 recording."""
    sensor = env.scene.sensors[sensor_name]
    images = sensor.data.output[data_type].detach().to(device="cpu", copy=True).contiguous()

    if data_type == "rgb" and images.dtype != torch.uint8:
        if images.numel() > 0 and float(images.max()) <= 1.0:
            images = images * 255.0
        images = images.clamp(0, 255).to(torch.uint8)
    return images


class PreStepShowroomCameraObservationsRecorder(RecorderTerm):
    """Record showroom camera observations without putting images in policy observations."""

    cfg: "PreStepShowroomCameraObservationsRecorderCfg"

    def record_pre_step(self):
        observations = {}
        for camera_name in self.cfg.camera_names:
            if camera_name not in self._env.scene.sensors:
                continue
            with _profile_time(self._env, f"recorder_camera_{camera_name}"):
                observations[camera_name] = camera_image_cpu(self._env, camera_name)
        return "obs", observations


@configclass
class PreStepShowroomCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the showroom camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepShowroomCameraObservationsRecorder
    camera_names: tuple[str, ...] = SHOWROOM_CAMERA_NAMES


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
