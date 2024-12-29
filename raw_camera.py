import os
import platform
import ctypes
import logging
import subprocess
from typing import Optional, Dict
import cv2

logger = logging.getLogger(__name__)


class RawCamera:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device_fd = None
        self.buffers = {}
        self.cap = None
        self.system = platform.system()
        self._initialize_device()

    def _initialize_device(self) -> None:
        try:
            if self.system == "Darwin":
                self._initialize_macos()
            elif self.system == "Linux":
                self._initialize_linux()
            elif self.system == "Windows":
                self._initialize_windows()
            else:
                raise RuntimeError(f"Unsupported operating system: {self.system}")

        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.cleanup()
            raise

    def _initialize_macos(self) -> None:
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            logger.info("Successfully initialized camera on macOS")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera on macOS: {e}")

    def _initialize_linux(self) -> None:
        device_path = f"/dev/video{self.device_id}"
        if not os.path.exists(device_path):
            raise RuntimeError(f"Camera device not found: {device_path}")

        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera device: {device_path}")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            logger.info(f"Successfully initialized camera on Linux: {device_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera on Windows: {e}")

    def _initialize_windows(self) -> None:
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera (Windows)")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            logger.info("Successfully initialized camera on Windows")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera on Windows: {e}")

    def read_frame(self) -> Optional[bytes]:
        try:
            if self.system == "Darwin":
                return self._read_frame_macos()
            elif self.system == "Linux":
                return self._read_frame_linux()
            elif self.system == "Windows":
                return self._read_frame_windows()

        except Exception as e:
            logger.error(f"Failed to read frame: {e}")
            return None

    def _read_frame_macos(self) -> Optional[bytes]:
        try:
            if self.cap is None:
                return None

            print("[Camera] Starting frame capture")
            ret, frame = self.cap.read()
            if not ret:
                print("[Camera] Failed to capture frame")
                return None

            print(f"[Camera] Frame captured successfully: shape={frame.shape}")
            return frame.tobytes()

        except Exception as e:
            logger.error(f"[Camera] Failed to read read frame: {e}")
            return None

    def _read_frame_linux(self) -> Optional[bytes]:
        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.device_id)
                if not self.cap.isOpened():
                    raise RuntimeError("[Camera] Failed to open camera (Linux)")

            print("[Camera] Srarting frame capture (Linux)")
            ret, frame = self.cap.read()
            if not ret:
                print("[Camera] Failed to capture frame")
                return None

            print(f"[Camera] Frame captured successfully (Linux): shape={frame.shape}")
            return frame.tobytes()

        except Exception as e:
            logger.error(f"[Camera] Failed to read read frame (Linux): {e}")
            return None

    def _read_frame_windows(self) -> Optional[bytes]:
        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.device_id)
                if not self.cap.isOpened():
                    raise RuntimeError("Could not open camera")

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            print("[Camera] Starting frame capture (Windows)")
            ret, frame = self.cap.read()
            if not ret:
                print("[Camera] Failed to capture frame (Windows)")
                return None

            print(
                f"[Camera] Frame captured successfully (Windows): shape={frame.shape}"
            )
            return frame.tobytes()

        except Exception as e:
            logger.error(f"[Camera] Failed to read read frame (Windows): {e}")
            return None

    def start_streaming(self) -> None:
        pass

    def stop_streaming(self) -> None:
        if self.system == "Darwin":
            if self.cap is not None:
                self.cap.release()

    def cleanup(self) -> None:
        try:
            self.stop_streaming()

            if self.cap is not None:
                self.cap.release()
                self.cap = None

            if self.device_fd is not None:
                os.close(self.device_fd)
                self.device_fd = None

            self.buffers.clear()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
