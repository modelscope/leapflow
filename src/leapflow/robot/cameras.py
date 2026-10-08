# Copyright (c) Alibaba, Inc. and its affiliates.
"""Camera abstractions for LeapRobot.

Defines the ``Camera`` Protocol and a concrete ``OpenCVCamera``
implementation.  opencv-python is an optional dependency — the module
is importable without it, but ``OpenCVCamera`` requires it at runtime.
"""

from __future__ import annotations

import logging
import math
import platform
import time
from threading import Event, Lock, Thread
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from numpy.typing import NDArray

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]
    NDArray = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Camera(Protocol):
    """Protocol for camera devices.

    Any object satisfying this protocol can be used as a camera source
    within the LeapFlow hardware stack.
    """

    @property
    def is_connected(self) -> bool:
        """Whether the camera is currently connected and ready."""
        ...

    def connect(self, warmup: bool = True) -> None:
        """Establish connection to the camera.

        Args:
            warmup: If ``True``, capture a warm-up frame before
                returning.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from the camera and release resources."""
        ...

    def read(self) -> Any:
        """Capture and return a single frame synchronously (blocking)."""
        ...

    def async_read(self, timeout_ms: float = 200) -> Any:
        """Return the most recent new frame (may block up to *timeout_ms*)."""
        ...


# ---------------------------------------------------------------------------
# CameraConfig
# ---------------------------------------------------------------------------

class CameraConfig:
    """Minimal camera configuration.

    Replaces the draccus-based config from the upstream project with a
    plain Python class.
    """

    def __init__(
        self,
        index_or_path: int | str = 0,
        *,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        color_mode: str = "rgb",
        rotation: int | None = None,
        warmup_s: float = 1.0,
    ) -> None:
        self.index_or_path = index_or_path
        self.fps = fps
        self.width = width
        self.height = height
        self.color_mode = color_mode
        self.rotation = rotation
        self.warmup_s = warmup_s


# ---------------------------------------------------------------------------
# OpenCVCamera
# ---------------------------------------------------------------------------

def _get_cv2_rotation(degrees: int | None) -> int | None:
    """Map rotation degrees to an OpenCV constant (or ``None``)."""
    if degrees is None:
        return None
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return None
    mapping = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return mapping.get(degrees)


class OpenCVCamera:
    """Camera implementation backed by OpenCV's ``VideoCapture``.

    Supports synchronous and asynchronous (background-thread) frame
    capture.  The heavy ``cv2`` import happens lazily at connect time so
    the module can be imported in environments where OpenCV is absent.
    """

    def __init__(self, config: CameraConfig | None = None, **kwargs: Any) -> None:
        cfg = config or CameraConfig(**kwargs)
        self._index_or_path = cfg.index_or_path
        self._target_fps = cfg.fps
        self._target_width = cfg.width
        self._target_height = cfg.height
        self._color_mode = cfg.color_mode.lower()
        self._warmup_s = cfg.warmup_s
        self._rotation = _get_cv2_rotation(cfg.rotation)

        # Runtime state
        self._cap: Any = None  # cv2.VideoCapture
        self._thread: Thread | None = None
        self._stop_event: Event | None = None
        self._frame_lock = Lock()
        self._latest_frame: Any = None
        self._latest_timestamp: float | None = None
        self._new_frame = Event()

        # Resolved after connect
        self.fps: int | float | None = cfg.fps
        self.width: int | None = cfg.width
        self.height: int | None = cfg.height

    def __str__(self) -> str:
        return f"OpenCVCamera({self._index_or_path})"

    # -- Protocol properties / methods ------------------------------------

    @property
    def is_connected(self) -> bool:
        try:
            import cv2
        except ImportError:  # pragma: no cover
            return False
        return isinstance(self._cap, cv2.VideoCapture) and self._cap.isOpened()

    def connect(self, warmup: bool = True) -> None:
        """Open the camera and start the background capture thread."""
        import cv2

        if self.is_connected:
            return

        cv2.setNumThreads(1)
        self._cap = cv2.VideoCapture(self._index_or_path)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise ConnectionError(
                f"Failed to open camera at {self._index_or_path}."
            )

        try:
            self._apply_settings(cv2)
            self._start_thread()

            if warmup and self._warmup_s > 0:
                deadline = time.time() + self._warmup_s
                while time.time() < deadline:
                    try:
                        self.async_read(timeout_ms=self._warmup_s * 1000)
                    except TimeoutError:
                        pass
                    time.sleep(0.1)
        except BaseException:
            self._cleanup()
            raise

        logger.info("%s connected.", self)

    def disconnect(self) -> None:
        """Stop capture and release the camera."""
        if not self.is_connected and self._thread is None:
            return
        self._cleanup()
        logger.info("%s disconnected.", self)

    def read(self) -> Any:
        """Synchronous blocking read (delegates to async_read)."""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        self._new_frame.clear()
        return self.async_read(timeout_ms=10_000)

    def async_read(self, timeout_ms: float = 200) -> Any:
        """Return the latest frame, waiting up to *timeout_ms*."""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"{self} capture thread is not running.")

        if not self._new_frame.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"No frame from {self} within {timeout_ms} ms."
            )
        with self._frame_lock:
            frame = self._latest_frame
            self._new_frame.clear()

        if frame is None:
            raise RuntimeError(f"Event set but no frame for {self}.")
        return frame

    # -- Context manager --------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disconnect()

    # -- internals --------------------------------------------------------

    def _apply_settings(self, cv2_mod: Any) -> None:
        """Apply FPS / resolution to the opened capture."""
        cap = self._cap
        if self._target_width is not None:
            cap.set(cv2_mod.CAP_PROP_FRAME_WIDTH, float(self._target_width))
        if self._target_height is not None:
            cap.set(cv2_mod.CAP_PROP_FRAME_HEIGHT, float(self._target_height))
        if self._target_fps is not None:
            cap.set(cv2_mod.CAP_PROP_FPS, float(self._target_fps))

        # Read back the actual values
        self.width = int(round(cap.get(cv2_mod.CAP_PROP_FRAME_WIDTH)))
        self.height = int(round(cap.get(cv2_mod.CAP_PROP_FRAME_HEIGHT)))
        self.fps = cap.get(cv2_mod.CAP_PROP_FPS)

    def _start_thread(self) -> None:
        self._stop_thread()
        self._stop_event = Event()
        self._thread = Thread(
            target=self._capture_loop, daemon=True,
            name=f"{self}_capture",
        )
        self._thread.start()
        time.sleep(0.1)

    def _stop_thread(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._stop_event = None
        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = None
            self._new_frame.clear()

    def _cleanup(self) -> None:
        thread = self._thread
        cap = self._cap
        try:
            self._stop_thread()
        finally:
            self._cap = None
            if cap is not None:
                cap.release()
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

    def _capture_loop(self) -> None:
        """Background loop that reads frames from hardware."""
        import cv2

        stop = self._stop_event
        failures = 0
        while stop is not None and not stop.is_set():
            try:
                ret, frame = self._cap.read()
                if not ret:
                    raise RuntimeError("read() returned False")

                # Color conversion
                if self._color_mode == "rgb":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Rotation
                if self._rotation is not None:
                    frame = cv2.rotate(frame, self._rotation)

                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_timestamp = time.perf_counter()
                self._new_frame.set()
                failures = 0
            except Exception as exc:
                failures += 1
                if failures > 10:
                    logger.error(
                        "%s exceeded max consecutive failures: %s", self, exc,
                    )
                    break
                logger.warning("%s capture error: %s", self, exc)

    # -- static discovery -------------------------------------------------

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detect available OpenCV cameras on the system."""
        try:
            import cv2
        except ImportError:  # pragma: no cover
            return []

        from pathlib import Path

        found: list[dict[str, Any]] = []
        if platform.system() == "Linux":
            targets: list[str | int] = [
                str(p) for p in sorted(Path("/dev").glob("video*"))
            ]
        else:
            targets = list(range(60))

        for target in targets:
            cap = cv2.VideoCapture(target)
            try:
                if cap.isOpened():
                    found.append({
                        "type": "OpenCV",
                        "id": target,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                    })
            finally:
                cap.release()
        return found


__all__ = [
    "Camera",
    "CameraConfig",
    "OpenCVCamera",
]
