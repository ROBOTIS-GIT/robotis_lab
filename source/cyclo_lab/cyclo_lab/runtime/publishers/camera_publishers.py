"""Compressed image publishing helpers for IsaacLab camera sensors."""

from __future__ import annotations

from collections.abc import Callable
import time

import cv2
import torch

from cyclo_lab.runtime.transport.ros2_zenoh import (
    COMPRESSED_IMAGE,
    create_publisher,
    make_compressed_image_kwargs,
    now_time_msg,
)


def publish_compressed_camera(
    camera_name: str,
    camera,
    writer,
    *,
    frame_id: str | None = None,
    stamp_fn: Callable | None = None,
    image_rotation_quarter_turns: int = 0,
) -> None:
    img = camera.data.output["rgb"][0].detach()
    quarter_turns = int(image_rotation_quarter_turns) % 4
    if quarter_turns:
        img = torch.rot90(img, k=quarter_turns, dims=(0, 1))
    img = img.contiguous().cpu().numpy()
    if img.dtype != "uint8":
        max_value = float(img.max()) if img.size else 0.0
        if max_value <= 1.0:
            img = img * 255.0
        img = img.clip(0, 255).astype("uint8")
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", img_bgr)
    if not success:
        raise RuntimeError("cv2.imencode('.jpg', image) failed")

    stamp = stamp_fn() if stamp_fn is not None else now_time_msg()
    writer.publish(
        **make_compressed_image_kwargs(
            data=buffer.tobytes(),
            frame_id=frame_id or camera_name,
            fmt="jpeg",
            stamp=stamp,
        )
    )


class CompressedCameraPublishers:
    """Publish available IsaacLab RGB sensors to ROS2-compatible image topics."""

    def __init__(
        self,
        scene,
        camera_topics: dict[str, str],
        publish_hz: float | None,
        *,
        image_rotations: dict[str, int] | None = None,
    ) -> None:
        self._scene = scene
        self._publish_hz = publish_hz
        self._image_rotations = {
            camera_name: int(quarter_turns) % 4
            for camera_name, quarter_turns in (image_rotations or {}).items()
        }
        self._last_publish_time = 0.0
        self._warned_camera_publish_errors: set[str] = set()
        available_cameras = set(scene.sensors)
        if publish_hz == 0.0:
            self.writers = {}
        else:
            self.writers = {
                camera_name: create_publisher(topic, COMPRESSED_IMAGE)
                for camera_name, topic in camera_topics.items()
                if camera_name in available_cameras
            }

    @property
    def endpoints(self) -> tuple:
        return tuple(self.writers.values())

    def publish(self) -> None:
        if not self.writers:
            return
        if self._publish_hz is not None and self._publish_hz > 0.0:
            now = time.monotonic()
            publish_period = 1.0 / self._publish_hz
            if now - self._last_publish_time < publish_period * 0.95:
                return
            self._last_publish_time = now

        for camera_name, writer in self.writers.items():
            try:
                publish_compressed_camera(
                    camera_name,
                    self._scene[camera_name],
                    writer,
                    image_rotation_quarter_turns=self._image_rotations.get(camera_name, 0),
                )
            except Exception as exc:
                if camera_name not in self._warned_camera_publish_errors:
                    self._warned_camera_publish_errors.add(camera_name)
                    print(f"[Zenoh ROS2] camera publish error for {camera_name}: {exc}")
