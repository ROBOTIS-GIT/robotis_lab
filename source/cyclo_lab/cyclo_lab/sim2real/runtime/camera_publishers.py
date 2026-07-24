"""Compressed image publishing helpers for IsaacLab camera sensors."""

from __future__ import annotations

from collections.abc import Callable

import cv2

from cyclo_lab.sim2real.transport.ros2_zenoh import make_compressed_image_kwargs, now_time_msg


def publish_compressed_camera(
    camera_name: str,
    camera,
    writer,
    *,
    frame_id: str | None = None,
    stamp_fn: Callable | None = None,
) -> None:
    img = camera.data.output["rgb"][0].detach().cpu().numpy()
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

