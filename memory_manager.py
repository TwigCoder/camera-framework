import mmap
import ctypes
import numpy as np
from typing import Optional, Dict, List
import logging
from enum import Enum, auto
from dataclasses import dataclass
import threading
import queue
import time

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    CPU = auto()
    GPU = auto()
    UNIFIED = auto()


@dataclass
class BufferSpec:
    size: int
    alignment: int
    memory_type: MemoryType
    flags: int = 0
    name: str = "unnamed"


class MemoryBlock:
    def __init__(self, size: int, alignment: int):
        self.size = size
        self.alignment = alignment
        self.raw_ptr: Optional[int] = None
        self.aligned_ptr: Optional[int] = None
        self.buffer: Optional[mmap.mmap] = None
        self._allocate()

    def _allocate(self) -> None:
        try:
            total_size = self.size + self.alignment

            self.buffer = mmap.mmap(
                -1,
                total_size,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )

            self.raw_ptr = ctypes.addressof(ctypes.c_void_p.from_buffer(self.buffer))

            mask = self.alignment - 1
            self.aligned_ptr = (self.raw_ptr + mask) & ~mask

        except Exception as e:
            logger.error(f"Failed to allocate memory block: {e}")
            self.cleanup()
            raise

    def get_view(self) -> memoryview:
        if not self.buffer:
            raise RuntimeError("Buffer not allocated")

        offset = self.aligned_ptr - self.raw_ptr
        return memoryview(self.buffer)[offset : offset + self.size]

    def cleanup(self) -> None:
        if self.buffer:
            self.buffer.close()
            self.buffer = None
            self.raw_ptr = None
            self.aligned_ptr = None


class MemoryPool:
    def __init__(self, block_sizes: List[int], blocks_per_size: int = 4):
        self.blocks: Dict[int, List[MemoryBlock]] = {}
        self.in_use: Dict[int, List[bool]] = {}
        self.lock = threading.Lock()

        for size in block_sizes:
            self.blocks[size] = []
            self.in_use[size] = []
            for _ in range(blocks_per_size):
                block = MemoryBlock(size, 256)
                self.blocks[size].append(block)
                self.in_use[size].append(False)

    def acquire(self, size: int) -> Optional[memoryview]:
        with self.lock:
            print(f"[Memory] Requesting block of size {size}")
            if size not in self.blocks:
                print(f"[Memory] No pool for size {size}")
                return None

            for i, in_use in enumerate(self.in_use[size]):
                if not in_use:
                    self.in_use[size][i] = True
                    print(f"[Memory] Allocated block {i}")
                    return self.blocks[size][i].get_view()

            logger.warning(f"No available blocks for size {size}")
            return None

    def release(self, size: int, view: memoryview) -> None:
        with self.lock:
            if size not in self.blocks:
                return

            for i, block in enumerate(self.blocks[size]):
                if block.aligned_ptr == ctypes.addressof(
                    ctypes.c_void_p.from_buffer(view.obj)
                ):
                    self.in_use[size][i] = False
                    break

    def cleanup(self) -> None:
        with self.lock:
            for size_blocks in self.blocks.values():
                for block in size_blocks:
                    block.cleanup()
            self.blocks.clear()
            self.in_use.clear()
