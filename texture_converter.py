import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class TextureConverter:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3

    def process_frame(self, frame_data: bytes) -> Tuple[bool, np.ndarray]:
        try:
            print("[Converter] Starting frame conversion")
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)

            if len(frame_array) == 230400:
                print("[Converter] Processing YUV format")
                frame_array = frame_array.reshape((self.height, self.width, 2))
                return True, self._yuv_to_rgb(frame_array)
            elif len(frame_array) == self.width * self.height * 3:
                print("[Converter] Processing RGB format")
                frame_array = frame_array.reshape((self.height, self.width, 3))
                frame_array = frame_array[:, :, ::-1].copy()
                print("[Converter] Conversion complete")
                return True, frame_array.astype(np.float32) / 255.0
            else:
                logger.error(f"Unexpected frame size: {len(frame_array)}")
                return False, np.zeros((self.height, self.width, 3), dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to process frame: {e}")
            return False, np.zeros((self.height, self.width, 3), dtype=np.float32)

    def _yuv_to_rgb(self, yuv: np.ndarray) -> np.ndarray:
        try:
            y = yuv[:, :, 0].astype(np.float32)
            uv = yuv[:, :, 1].astype(np.float32)

            y = (y - 16) / 219.0
            uv = (uv - 128) / 224.0

            r = np.clip(y + 1.402 * uv, 0, 1)
            g = np.clip(y - 0.344136 * uv - 0.714136 * uv, 0, 1)
            b = np.clip(y + 1.772 * uv, 0, 1)

            rgb = np.stack([r, g, b], axis=2)

            return rgb.astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to convert YUV to RGB: {e}")
            return np.zeros((self.height, self.width, 3), dtype=np.float32)
