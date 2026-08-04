"""Non-blocking RGB camera preview backed by POSIX shared memory."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import struct
import subprocess
from multiprocessing import shared_memory

import numpy as np


_FRAME_OFFSET = 64
_SEQUENCE = struct.Struct("<Q")
_STOP_SEQUENCE = (1 << 64) - 1


class SharedMemoryCameraPreview:
    """Send the latest RGB frame to an isolated lightweight viewer process."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        window_title: str = "SG2 Overhead Camera",
        window_size: int = 720,
        window_position: tuple[int, int] | None = None,
        panel_labels: tuple[tuple[str, int, int], ...] = (),
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self._sequence = 0
        self._closed = False
        self._shared_memory = shared_memory.SharedMemory(
            create=True,
            size=_FRAME_OFFSET + self.width * self.height * 3,
        )
        self._frame = np.ndarray(
            (self.height, self.width, 3),
            dtype=np.uint8,
            buffer=self._shared_memory.buf,
            offset=_FRAME_OFFSET,
        )
        self._frame.fill(0)
        _SEQUENCE.pack_into(self._shared_memory.buf, 0, self._sequence)

        viewer_python = os.environ.get("CYCLO_CAMERA_VIEWER_PYTHON", "/opt/cyclo-camera-viewer/bin/python")
        if not Path(viewer_python).is_file():
            self._cleanup_shared_memory()
            raise RuntimeError(
                f"Camera viewer Python not found: {viewer_python}. Rebuild the cyclo-lab image "
                "or set CYCLO_CAMERA_VIEWER_PYTHON."
            )

        viewer_script = Path(__file__).with_name("camera_preview_process.py")
        viewer_env = os.environ.copy()
        for variable in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "PYTHONPATH"):
            viewer_env.pop(variable, None)
        viewer_env.setdefault("QT_X11_NO_MITSHM", "1")
        self._process = subprocess.Popen(
            [
                viewer_python,
                str(viewer_script),
                "--shared-memory",
                self._shared_memory.name,
                "--width",
                str(self.width),
                "--height",
                str(self.height),
                "--window-size",
                str(int(window_size)),
                "--title",
                window_title,
                "--parent-pid",
                str(os.getpid()),
                "--panel-labels",
                json.dumps(panel_labels),
                *(
                    ["--window-x", str(window_position[0]), "--window-y", str(window_position[1])]
                    if window_position is not None
                    else []
                ),
            ],
            env=viewer_env,
            start_new_session=True,
        )
        atexit.register(self.close)

    @property
    def is_open(self) -> bool:
        return not self._closed and self._process.poll() is None

    def update(self, image) -> None:
        """Publish one RGB frame, replacing any frame the viewer has not consumed."""
        if not self.is_open:
            return
        if hasattr(image, "detach"):
            image = image.detach()
            if getattr(image, "ndim", 0) == 4:
                image = image[0]
            image = image.to(device="cpu", copy=True).contiguous().numpy()
        else:
            image = np.asarray(image)

        if image.ndim != 3 or image.shape[0:2] != (self.height, self.width):
            raise ValueError(
                f"Preview frame must have shape ({self.height}, {self.width}, C), got {image.shape}."
            )
        if image.shape[2] < 3:
            raise ValueError(f"Preview frame must have at least 3 channels, got {image.shape[2]}.")
        image = image[..., :3]
        if image.dtype != np.uint8:
            if image.size and float(image.max()) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        _SEQUENCE.pack_into(self._shared_memory.buf, 0, self._sequence + 1)
        np.copyto(self._frame, image)
        self._sequence += 2
        _SEQUENCE.pack_into(self._shared_memory.buf, 0, self._sequence)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            _SEQUENCE.pack_into(self._shared_memory.buf, 0, _STOP_SEQUENCE)
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=1.0)
        self._cleanup_shared_memory()
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def _cleanup_shared_memory(self) -> None:
        try:
            self._shared_memory.close()
        except Exception:
            pass
        try:
            self._shared_memory.unlink()
        except FileNotFoundError:
            pass
