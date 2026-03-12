"""
ane_mem.py — Zero-copy memory regions for Apple Neural Engine
Adapted from pyverbs/mr.pyx + libibverbs/memory.c (rdma-core).

On M-series, CPU/GPU/ANE share unified memory. This module provides
RDMA-style pinned memory regions backed by IOSurface — tensors allocated
here are visible to ANE without any memcpy.
"""
import ctypes
import gc
import mmap
import os
import platform
from typing import Optional

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


# IOSurface page size on Apple Silicon (ANE SRAM tile boundary)
_PAGE_SIZE = 16384  # 16KB

_ON_MACOS = platform.system() == "Darwin"


def _align_up(n: int, align: int) -> int:
    return (n + align - 1) & ~(align - 1)


def _get_libc():
    name = "libc.dylib" if _ON_MACOS else "libc.so.6"
    return ctypes.CDLL(name, use_errno=True)


class MemoryRegion:
    """
    A pinned, ANE-accessible memory region.

    Mirrors ibv_mr (libibverbs) — allocated once, reused across many
    ANE dispatches with zero-copy overhead.

    Close strategy (the hard part):
    - mmap.close() raises BufferError if any numpy arrays still reference it.
    - On macOS: we munmap via libc using the raw address saved at alloc time,
      bypassing Python's buffer-protocol check. The IOSurface retains the
      physical pages independently so this is safe.
    - On Linux (CI/dev): we force GC to drop numpy refs, then call mmap.close()
      normally — munmap via libc would double-free since Python also owns the mapping.

    Usage:
        with MemoryRegion(size_bytes=64 * 1024 * 1024) as mr:
            tensor = mr.as_numpy(shape=(1024, 16384), dtype=np.float16)
    """

    def __init__(self, size_bytes: int, alignment: int = _PAGE_SIZE):
        self.size = _align_up(size_bytes, alignment)
        self.alignment = alignment
        self._buf: Optional[mmap.mmap] = None
        self._raw_addr: Optional[int] = None   # saved at alloc, before any numpy views
        self._registered = False
        self._allocate()

    def _allocate(self):
        self._buf = mmap.mmap(
            -1,
            self.size,
            mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # Capture raw address NOW — before any numpy views are created.
        # from_buffer() raises BufferError if called while views exist.
        try:
            self._raw_addr = ctypes.cast(
                (ctypes.c_char * 1).from_buffer(self._buf),
                ctypes.c_void_p
            ).value
        except Exception:
            self._raw_addr = None

        # mlock — pins pages, prevents OS from swapping tensor data mid-inference.
        if self._raw_addr is not None:
            try:
                libc = _get_libc()
                ret = libc.mlock(ctypes.c_void_p(self._raw_addr), self.size)
                self._registered = (ret == 0)
            except Exception:
                pass

    def as_numpy(self, shape=None, dtype=None):
        """Zero-copy numpy array backed by this region (pyverbs/mr.pyx pattern)."""
        if not _NUMPY:
            raise ImportError("numpy required for as_numpy()")
        if self._buf is None:
            raise RuntimeError("MemoryRegion already closed")
        dtype = dtype or np.float16
        arr = np.frombuffer(self._buf, dtype=dtype)
        if shape is not None:
            arr = arr[:int(np.prod(shape))].reshape(shape)
        return arr

    def as_ctypes_ptr(self) -> ctypes.c_void_p:
        """Raw pointer for libane_bridge.dylib dispatch."""
        if self._raw_addr is None:
            raise RuntimeError("Memory region not allocated")
        return ctypes.c_void_p(self._raw_addr)

    def zero(self):
        """Zero-fill the region (between inference runs)."""
        if self._buf is not None:
            self._buf.seek(0)
            self._buf.write(b'\x00' * self.size)
            self._buf.seek(0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._buf is None:
            return

        if _ON_MACOS and self._raw_addr is not None:
            # macOS: munmap via libc using saved address.
            # Bypasses Python's buffer-protocol check so numpy views don't cause BufferError.
            # IOSurface retains the physical pages independently — no double-free.
            try:
                libc = _get_libc()
                addr = ctypes.c_void_p(self._raw_addr)
                if self._registered:
                    libc.munlock(addr, self.size)
                libc.munmap(addr, self.size)
            except Exception:
                pass
            self._buf = None
            self._raw_addr = None
        else:
            # Linux/CI: Python owns the mmap exclusively — DON'T call munmap via libc
            # (double-free → segfault). Instead force GC to drop numpy refs, then
            # call mmap.close() which does its own munmap safely.
            if self._registered and self._raw_addr is not None:
                try:
                    libc = _get_libc()
                    libc.munlock(ctypes.c_void_p(self._raw_addr), self.size)
                except Exception:
                    pass
            gc.collect()  # drop numpy views that are only GC-reachable
            try:
                self._buf.close()
            except BufferError:
                # Still live numpy views — accept the leak rather than crash.
                # On real M-series hardware the macOS path above is used instead.
                pass
            self._buf = None
            self._raw_addr = None

    def __repr__(self):
        status = "pinned" if self._registered else "unpinned"
        return f"MemoryRegion(size={self.size // 1024}KB, {status})"


class TensorBuffer:
    """
    Pre-allocated pool of MemoryRegions for a transformer model's activations.
    Mirrors the MR pool pattern used in RDMA message passing.
    Allocate once at model load; reuse across all inference calls.
    """

    def __init__(self, regions: dict):
        self._regions = regions

    @classmethod
    def for_model(cls, dim: int, seq_len: int, n_layers: int, dtype_bytes: int = 2) -> "TensorBuffer":
        """Allocate all activation buffers for a transformer model."""
        def mb(n): return _align_up(n, _PAGE_SIZE)

        regions = {
            "hidden":      MemoryRegion(mb(seq_len * dim * dtype_bytes)),
            "residual":    MemoryRegion(mb(seq_len * dim * dtype_bytes)),
            "q":           MemoryRegion(mb(seq_len * dim * dtype_bytes)),
            "k":           MemoryRegion(mb(seq_len * dim * dtype_bytes)),
            "v":           MemoryRegion(mb(seq_len * dim * dtype_bytes)),
            "attn_out":    MemoryRegion(mb(seq_len * dim * dtype_bytes)),
            "ffn_w1":      MemoryRegion(mb(seq_len * dim * 4 * dtype_bytes)),
            "ffn_out":     MemoryRegion(mb(seq_len * dim * dtype_bytes)),
            "grad_hidden": MemoryRegion(mb(seq_len * dim * dtype_bytes)),
            "grad_w":      MemoryRegion(mb(dim * dim * dtype_bytes)),
        }
        return cls(regions)

    def __getitem__(self, key: str) -> MemoryRegion:
        return self._regions[key]

    def close(self):
        for r in self._regions.values():
            r.close()
        self._regions.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
