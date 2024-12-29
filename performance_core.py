import ctypes
import numpy as np
from typing import Optional, List, Dict, Any
import threading
import queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
import mmap
import os

logger = logging.getLogger(__name__)


class MemoryAlignment(Enum):

    ALIGN_16 = 16
    ALIGN_32 = 32
    ALIGN_64 = 64
    ALIGN_128 = 128
    ALIGN_256 = 256


@dataclass
class BufferSpec:

    size: int
    alignment: MemoryAlignment
    usage: str
    flags: int = 0


class AlignedBuffer:

    def __init__(self, spec: BufferSpec):
        self.spec = spec
        self.raw_buffer: Optional[memoryview] = None
        self.aligned_ptr: int = 0
        self._allocate()

    def _allocate(self) -> None:
        raw_size = self.spec.size + self.spec.alignment.value - 1

        raw_buffer = mmap.mmap(-1, raw_size)
        self.raw_buffer = memoryview(raw_buffer)

        addr = ctypes.addressof(ctypes.c_void_p.from_buffer(raw_buffer))
        mask = self.spec.alignment.value - 1
        self.aligned_ptr = (addr + mask) & ~mask

    def get_buffer(self) -> memoryview:
        if self.raw_buffer is None:
            raise RuntimeError("Buffer not allocated")
        offset = self.aligned_ptr - ctypes.addressof(
            ctypes.c_void_p.from_buffer(self.raw_buffer)
        )
        return self.raw_buffer[offset : offset + self.spec.size]

    def cleanup(self) -> None:
        if self.raw_buffer is not None:
            self.raw_buffer.release()
            self.raw_buffer = None


class TextureUpdateTask:

    def __init__(self, texture_id: int, data: np.ndarray, width: int, height: int):
        self.texture_id = texture_id
        self.data = data
        self.width = width
        self.height = height
        self.timestamp = time.time()


class AsyncTextureManager:

    def __init__(self, max_workers: int = 4, queue_size: int = 8):
        self.update_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        self.pending_updates: Dict[int, concurrent.futures.Future] = {}
        self._start_worker()

    def _start_worker(self) -> None:
        self.worker_thread = threading.Thread(target=self._process_updates)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _process_updates(self) -> None:
        while self.running:
            try:
                task = self.update_queue.get(timeout=0.1)
                future = self.thread_pool.submit(self._update_texture_data, task)
                self.pending_updates[task.texture_id] = future

            except queue.Empty:
                continue

    def _update_texture_data(self, task: TextureUpdateTask) -> None:
        try:
            print("[Performance] Starting async texture update")
            buffer_spec = BufferSpec(
                size=task.data.nbytes,
                alignment=MemoryAlignment.ALIGN_256,
                usage="texture",
            )

            print("[Performance] Creating aligned buffer")
            aligned_buffer = AlignedBuffer(buffer_spec)
            buffer_view = aligned_buffer.get_buffer()

            print("[Performance] Copying texture data")
            np.copyto(
                np.frombuffer(buffer_view, dtype=task.data.dtype).reshape(
                    task.data.shape
                ),
                task.data,
            )

            print("[Performance] Async update complete")
            self.update_queue.task_done()

        except Exception as e:
            logger.error(f"Texture update failed: {e}")
            raise

    def cleanup(self) -> None:
        try:
            self.running = False
            while not self.update_queue.empty():
                try:
                    self.update_queue.get_nowait()
                except queue.Empty:
                    break

            self.thread_pool.shutdown(wait=True)

            self.pending_updates.clear()

        except Exception as e:
            logger.error(f"Failed to cleanup AsyncTextureManager: {e}")
