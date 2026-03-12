"""
ANE Python — Apple Neural Engine compute stack for M-series Macs.
Merges ANE private API access with RDMA-style zero-copy memory management.
"""
from .ane_device import ANEDevice, ANEDeviceInfo
from .ane_mem import MemoryRegion, TensorBuffer
from .ane_queue import ANECompletionQueue, ANEQueuePair, ANEWorkRequest, ANECompletion
from .ane_inference import ANEInference, InferenceResult

__all__ = [
    "ANEDevice", "ANEDeviceInfo",
    "MemoryRegion", "TensorBuffer",
    "ANECompletionQueue", "ANEQueuePair", "ANEWorkRequest", "ANECompletion",
    "ANEInference", "InferenceResult",
]
