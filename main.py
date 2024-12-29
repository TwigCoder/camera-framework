import pygame
import sys
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import logging
from camera_capture import VideoDevice
from gl_core import GLCore
from texture_converter import TextureConverter
import threading
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraApp:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.running = False

        pygame.init()
        pygame.display.set_caption("High-Performance Custom Camera Framework")
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )

        self.screen = pygame.display.set_mode(
            (width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        pygame.display.set_caption("Camera Feed")

        self.camera = VideoDevice()
        self.gl_core = GLCore(width, height)
        self.converter = TextureConverter(width, height)

        self.frame_queue = queue.Queue(maxsize=2)
        self.frame_thread = None

    def _process_frames(self):
        while self.running:
            try:
                frame_data = self.camera.read_frame()
                if frame_data is not None:
                    print("[Converter] Starting frame conversion")
                    frame_array = np.frombuffer(frame_data, dtype=np.float32)
                    frame_array = frame_array.reshape((480, 640, 3))
                    print("[Converter] Processing RGB format")
                    print("[Converter] Conversion complete")

                    if not self.frame_queue.full():
                        print("[GL] Starting texture update")
                        self.frame_queue.put(frame_array)
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                continue

    def start(self):
        try:
            self.running = True

            self.frame_thread = threading.Thread(target=self._process_frames)
            self.frame_thread.daemon = True
            self.frame_thread.start()

            clock = pygame.time.Clock()
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.width, self.height = event.size
                        self.gl_core.resize(self.width, self.height)

                try:
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get_nowait()
                        self.gl_core.update_texture(frame)

                    self.gl_core.render()
                    pygame.display.flip()
                    clock.tick(60)

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Render error: {e}")
                    continue

        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False
        if self.frame_thread:
            self.frame_thread.join(timeout=1.0)

        self.camera.cleanup()
        self.gl_core.cleanup()
        pygame.quit()


def main():
    try:
        app = CameraApp()
        app.start()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
