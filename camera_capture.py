import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VideoDevice:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            logger.info("Successfully initialized video capture")

        except Exception as e:
            logger.error(f"Failed to initialize video device: {e}")
            self.cleanup()
            raise

    def read_frame(self) -> Optional[bytes]:
        try:
            if self.cap is None:
                return None

            print("[Camera] Starting frame capture")
            ret, frame = self.cap.read()
            if not ret:
                print("[Camera] Failed to capture frame")
                return None

            print(f"[Camera] Frame captured successfully: shape={frame.shape}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0

            return frame.tobytes()

        except Exception as e:
            logger.error(f"Failed to read frame: {e}")
            return None

    def cleanup(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
