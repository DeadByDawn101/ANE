"""
ane_queue.py — Async dispatch and completion queue for ANE kernels.
Adapted from pyverbs/cq.pyx + pyverbs/qp.pyx (rdma-core).

Maps the RDMA completion queue model to ANE kernel dispatch:
  - ibv_post_send()  → ane_queue.submit()
  - ibv_poll_cq()    → ane_queue.poll()
  - ibv_req_notify() → ane_queue.wait()
"""
import ctypes
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_BRIDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "bridge", "libane_bridge.dylib")


@dataclass
class ANEWorkRequest:
    """
    Mirrors ibv_send_wr (work request) from libibverbs.
    Represents one ANE kernel dispatch.
    """
    kernel_name: str
    input_ptr: Optional[int] = None      # ctypes void* as int
    output_ptr: Optional[int] = None
    input_size: int = 0
    output_size: int = 0
    wr_id: Optional[int] = None              # user-defined ID (like ib_wr_id)
    callback: Optional[Callable] = None  # called on completion


@dataclass
class ANECompletion:
    """
    Mirrors ibv_wc (work completion) from libibverbs.
    Returned by poll() when a kernel finishes.
    """
    wr_id: int
    status: int          # 0 = success
    elapsed_ms: float
    output_ptr: Optional[int] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 0


class ANECompletionQueue:
    """
    Completion queue for ANE kernel dispatches.
    Mirrors ibv_cq from libibverbs — poll-based completion notification.
    """

    def __init__(self, depth: int = 64):
        self.depth = depth
        self._cq: queue.Queue[ANECompletion] = queue.Queue(maxsize=depth)
        self._pending = 0
        self._lock = threading.Lock()

    def post_completion(self, wc: ANECompletion):
        """Called internally when a kernel finishes."""
        self._cq.put(wc, block=False)
        with self._lock:
            self._pending = max(0, self._pending - 1)

    def poll(self, num_entries: int = 1, timeout: float = 0.0) -> list[ANECompletion]:
        """
        Poll for completions. Non-blocking by default (timeout=0).
        Mirrors ibv_poll_cq().
        """
        results = []
        deadline = time.monotonic() + timeout
        while len(results) < num_entries:
            try:
                t = max(0.0, deadline - time.monotonic()) if timeout > 0 else None
                wc = self._cq.get(block=(timeout > 0), timeout=t)
                results.append(wc)
            except queue.Empty:
                break
        return results

    def wait_all(self, timeout: float = 30.0) -> list[ANECompletion]:
        """Wait until all pending work requests complete."""
        return self.poll(num_entries=self.depth, timeout=timeout)


class ANEQueuePair:
    """
    ANE dispatch queue. Mirrors ibv_qp (queue pair) from libibverbs.
    
    Wraps libane_bridge.dylib dispatch with:
    - Async submission (non-blocking like ibv_post_send)
    - GCD-style overlap: CPU cblas gradients run while ANE forward pass executes
    - exec() restart when ANE kernel limit (~119) is approached
    """

    _KERNEL_LIMIT = 100  # restart before hitting the ~119 ANE compile limit

    def __init__(self, cq: ANECompletionQueue, lib=None):
        self._cq = cq
        self._lib = lib
        self._submitted = 0
        self._exec_count = 0
        self._thread_pool = []
        self._wr_counter = 0
        self._lock = threading.Lock()

    def post_send(self, wr: ANEWorkRequest) -> int:
        """
        Submit a work request to ANE. Non-blocking.
        Mirrors ibv_post_send().
        
        Returns wr_id for tracking completion via poll().
        """
        with self._lock:
            self._wr_counter += 1
            wr_id = wr.wr_id if wr.wr_id is not None else self._wr_counter
            self._submitted += 1
            self._exec_count += 1

            # Restart exec() to bypass ANE 119-kernel compile limit
            if self._exec_count >= self._KERNEL_LIMIT:
                self._restart_exec()

        # Dispatch asynchronously (mirrors GCD async dispatch in ANE bridge)
        t = threading.Thread(
            target=self._dispatch,
            args=(wr, wr_id),
            daemon=True,
        )
        t.start()
        self._thread_pool.append(t)
        return wr_id

    def _dispatch(self, wr: ANEWorkRequest, wr_id: int):
        """Execute the ANE kernel and post completion."""
        t0 = time.monotonic()
        status = 0
        error = None

        try:
            if self._lib is not None:
                # Real dispatch via libane_bridge.dylib
                fn = getattr(self._lib, f"ane_{wr.kernel_name}", None)
                if fn:
                    ret = fn(
                        ctypes.c_void_p(wr.input_ptr),
                        ctypes.c_void_p(wr.output_ptr),
                        ctypes.c_size_t(wr.input_size),
                        ctypes.c_size_t(wr.output_size),
                    )
                    status = int(ret)
                else:
                    # Kernel not in dylib — simulate
                    time.sleep(0.001)
            else:
                # No dylib loaded — simulation mode
                time.sleep(0.001)

        except Exception as e:
            status = 1
            error = str(e)

        elapsed = (time.monotonic() - t0) * 1000
        wc = ANECompletion(
            wr_id=wr_id,
            status=status,
            elapsed_ms=elapsed,
            output_ptr=wr.output_ptr,
            error=error,
        )
        self._cq.post_completion(wc)
        if wr.callback:
            wr.callback(wc)

    def _restart_exec(self):
        """
        Restart the ANE execution context to bypass the ~119 kernel compile limit.
        Mirrors the exec() restart pattern from inmem_bench.m.
        """
        self._exec_count = 0
        if self._lib is not None:
            restart_fn = getattr(self._lib, "ane_restart_exec", None)
            if restart_fn:
                restart_fn()

    def drain(self, timeout: float = 30.0):
        """Wait for all in-flight work requests to complete."""
        for t in self._thread_pool:
            t.join(timeout=timeout)
        self._thread_pool.clear()
        self._submitted = 0
