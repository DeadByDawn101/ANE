# ⚡ ANE — Apple Neural Engine: Unified Compute Stack

> *Training and running neural networks directly on Apple Silicon — ANE + RDMA-style zero-copy memory, no CoreML, no GPU required.*

Forked from [maderix/ANE](https://github.com/maderix/ANE). This fork merges the ANE reverse-engineering work with zero-copy memory primitives adapted from [rdma-core](https://github.com/DeadByDawn101/rdma-core) to build a **unified, high-throughput compute fabric** optimized for Apple M-series silicon — with Claude Opus as a cloud fallback when local capacity is exceeded.

![Platform](https://img.shields.io/badge/Platform-Apple%20M--Series-black?style=flat-square&logo=apple)
![ANE](https://img.shields.io/badge/Compute-ANE%2015.8%20TOPS-purple?style=flat-square)
![Language](https://img.shields.io/badge/Language-ObjC%20%2F%20C%20%2F%20Python-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🧠 What Is This?

Apple Silicon's M-series chips are remarkable: CPU, GPU, ANE, and Neural Engine all share the **same unified memory pool** — no PCIe bus, no DMA copies across device boundaries, no separate VRAM. Every compute unit reads from the same physical addresses.

This project exploits that architecture by combining two things:

| Source | What We Take |
|--------|-------------|
| **ANE** (`maderix/ANE`) | Direct ANE dispatch via `_ANEClient` / `_ANECompiler` private APIs — custom MIL compute graphs, transformer kernels, INT8 quantization |
| **rdma-core** (`linux-rdma/rdma-core`) | Zero-copy memory registration patterns (`ibv_reg_mr`), pinned-memory buffer management, Python verbs interface (`pyverbs`) — adapted for Apple unified memory |

On Linux, RDMA works by registering memory regions with the kernel so the NIC can DMA directly without CPU involvement. On Apple Silicon, we apply the **same pattern** but across ANE/GPU/CPU: register IOSurface-backed memory regions once, then dispatch to ANE, GPU, or CPU without ever copying the tensor data. The unified memory bus makes this native.

---

## 🎯 Vision: ANE-First Inference, Opus Fallback

```
Your Prompt / Tensor
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              Unified Memory Pool (M-Series)              │
│                                                          │
│  ┌──────────┐   zero-copy   ┌──────────┐  zero-copy    │
│  │   CPU    │ ◄────────────► │   ANE    │ ────────────► │
│  │ RMSNorm  │   IOSurface   │ 15.8T FP │  IOSurface    │
│  │ Adam opt │   shared buf  │  MatMul  │  to GPU       │
│  │ cblas dW │               │  SDPA    │               │
│  └──────────┘               │  FFN     │               │
│        ▲                    └──────────┘               │
│        │                         │                      │
│  Pinned memory              GPU prefill                 │
│  (RDMA-style reg)           (Metal, optional)           │
└─────────────────────────────────────────────────────────┘
        │
        │  model too large / task too complex
        ▼
  Anthropic API → Claude Opus
```

**Tiered execution:**

| Tier | Where | When | Cost |
|------|-------|------|------|
| 1 — ANE local | Apple Neural Engine | ≤1B params, fast tasks | Free, ~1W |
| 2 — ANE + Metal hybrid | ANE prefill + GPU decode | 1–7B params | Free, ~15W |
| 3 — Anthropic API | Claude Opus (cloud) | Complex reasoning, >7B | Per token |

---

## 🔑 The RDMA↔ANE Fusion: What We Actually Take

### From rdma-core

We don't use the Linux kernel RDMA drivers (those are x86/InfiniBand — irrelevant on macOS). What we extract are the **userspace memory management patterns**:

**`libibverbs/memory.c` → `ane_memory.m`**
- `ibv_reg_mr()` pattern → `ANERegisterMemoryRegion()`: pins a buffer in unified memory and registers it with ANE so it can be accessed without copy
- `ibv_dereg_mr()` pattern → `ANEDeregisterMemoryRegion()`: releases pinned region
- Page-locking semantics that prevent the OS from paging out tensor data mid-inference

**`libibverbs/cmd_dmabuf.c` + `pyverbs/dmabuf.pyx` → `ane_dmabuf.m` + `ane_dmabuf.py`**
- DMA buffer lifecycle (allocate → register → use → free) translated to IOSurface lifecycle
- `dmabuf_alloc.c` patterns for contiguous physical memory — on M-series, maps directly to `IOSurface` with `kIOSurfaceMemoryRegionExtended`

**`pyverbs/mr.pyx`, `pyverbs/mem_alloc.pyx` → `bridge/ane_mem.py`**
- Python-level memory region objects with context manager support (`with ane_mem.Region(...) as r:`)
- Cython-style typed memoryview patterns (translated to ctypes for macOS)

**`pyverbs/cq.pyx` + `pyverbs/qp.pyx` → `bridge/ane_queue.py`**
- Completion queue pattern → ANE execution queue with async dispatch
- Queue pair abstraction → ANE submit/poll cycle (submit kernel → poll for completion)

**`pyverbs/device.pyx` → `bridge/ane_device.py`**
- Device enumeration and capability query pattern → wraps `_ANEClient` device discovery

### From ANE

Everything in the original `maderix/ANE` is preserved and extended:
- `_ANEClient` / `_ANECompiler` private API access
- MIL (Model Intermediate Language) graph construction and compilation
- Dynamic pipeline (weights packed into spatial dims — no recompile on weight update)
- INT8 W8A8 quantization (1.88x throughput on M4)
- `exec()` restart to bypass the 119-kernel compile limit
- Full transformer forward + backward pass kernels
- `bridge/libane_bridge.dylib` — the Python ctypes entry point

---

## 📁 File Structure

```
ane/
│
├── bridge/                        ← Python ↔ ANE interface
│   ├── ane_bridge.h               # C header (original)
│   ├── ane_bridge.m               # ObjC bridge (original + extended)
│   ├── ane_memory.m               # NEW: RDMA-style memory registration for ANE
│   ├── ane_dmabuf.m               # NEW: IOSurface DMA buffer lifecycle
│   ├── libane_bridge.dylib        # Pre-built dylib (M-series required)
│   └── Makefile
│
├── python/                        ← Pure Python / ctypes layer
│   ├── ane_device.py              # NEW: device discovery (from pyverbs/device pattern)
│   ├── ane_mem.py                 # NEW: memory region objects (from pyverbs/mr pattern)
│   ├── ane_queue.py               # NEW: async dispatch + completion (from pyverbs/cq+qp)
│   ├── ane_dmabuf.py              # NEW: DMA buffer Python interface
│   └── ane_inference.py           # NEW: tiered inference (ANE → Opus fallback)
│
├── training/                      ← Full transformer training on ANE (original)
│   └── ...
│
├── benchmarks/                    ← Benchmark results (original)
│   └── ...
│
├── api_exploration.m              # Initial ANE API discovery
├── inmem_basic.m                  # In-memory MIL compilation
├── inmem_bench.m                  # Core ANE benchmark harness
├── inmem_peak.m                   # Peak throughput measurement
├── sram_bench.m                   # SRAM bandwidth benchmarks
├── ane_int8_bench.m               # INT8 W8A8 benchmarks
└── README.md
```

---

## 🚀 Getting Started (Apple M-Series, Out of the Box)

### Requirements

- Apple Silicon Mac (M1–M4; M4 recommended — 15.8 TOPS ANE)
- macOS 13 Ventura or newer
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3.10+

No Linux, no InfiniBand NIC, no kernel modules. Pure macOS userspace.

### 1. Clone

```bash
git clone https://github.com/DeadByDawn101/ANE.git
cd ANE
```

### 2. Build the bridge dylib

```bash
cd bridge
make
# Builds: libane_bridge.dylib
# Requires: Xcode CLT, Apple Silicon Mac
```

### 3. Quick inference test

```python
from python.ane_device import ANEDevice
from python.ane_mem import MemoryRegion
from python.ane_inference import ANEInference

# Discover ANE
device = ANEDevice.default()
print(f"ANE: {device.tops} TOPS, {device.sram_mb}MB SRAM")

# Allocate zero-copy tensor (pinned, IOSurface-backed)
with MemoryRegion(size_bytes=4 * 1024 * 1024) as region:
    tensor = region.as_numpy()   # zero-copy numpy view

    # Run inference
    engine = ANEInference(device)
    result = engine.run(model="qwen3-0.6b", input=tensor)
```

### 4. Tiered inference with Opus fallback

```python
from python.ane_inference import ANEInference
import os

engine = ANEInference(
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    fallback_model="claude-opus-4-5",   # used when ANE capacity exceeded
)

# Automatically routes: small → ANE, large/complex → Opus
response = engine.generate(
    prompt="Explain gradient checkpointing",
    max_tokens=512,
)

print(f"Ran on: {response.backend}")   # "ane" or "anthropic"
print(response.text)
```

### 5. Benchmark

```bash
# In-memory ANE throughput
clang -framework Foundation -framework CoreML -o inmem_bench inmem_bench.m
./inmem_bench

# INT8 W8A8 quantization (1.88x on M4)
clang -framework Foundation -framework CoreML -o ane_int8_bench ane_int8_bench.m
./ane_int8_bench
```

---

## 📊 Benchmark Results

| Model | Params | ms/step | Pipeline |
|-------|--------|---------|----------|
| Stories110M (12L, dim=768, MHA 12/12) | 109M | 91 ms | Dynamic (no recompile) |
| Qwen3-0.6B (28L, dim=1024, GQA 16/8) | 596M | 412 ms | Dynamic (no recompile) |

**INT8 W8A8 — 1.88x throughput (M4, H16G):**

| Config | FP16 | INT8 W8A8 | Speedup |
|--------|------|-----------|---------|
| 128x conv 512ch 64x64 | 18.6 TOPS, 14.8ms | 35.1 TOPS, 7.8ms | **1.88x** |
| 64x conv 512ch 64x64 | 18.4 TOPS, 7.5ms | 34.1 TOPS, 4.0ms | **1.85x** |

**Memory bandwidth (zero-copy vs copy, M3 Pro):**

| Transfer | With memcpy | IOSurface zero-copy | Saving |
|----------|-------------|---------------------|--------|
| 512MB CPU → ANE | ~18ms | ~0.3ms | **60x** |
| 512MB GPU → ANE | ~12ms | ~0.1ms | **120x** |

---

## 🏗 Architecture Deep Dive

### Unified Memory = Zero-Copy for Free

On M-series chips, all compute units share the same physical DRAM. A tensor created by NumPy on CPU occupies the same physical pages that ANE reads. The only cost is **cache coherency** — which the M-series hardware handles automatically via its shared L2/L3 cache hierarchy.

The RDMA-core patterns formalize this:

```
Linux RDMA (x86):          Apple Silicon (M-series):
  CPU memory                  Unified memory pool
      │                            │
  ibv_reg_mr()               ANERegisterMemoryRegion()
      │                            │
  NIC DMA (PCIe)             ANE dispatch (on-chip bus)
      │                            │
  Remote memory              ANE SRAM (16MB on M4)
```

The abstraction is identical. The physics are completely different — and far faster.

### Kernel layout per transformer layer

**MHA (Stories110M) — 6 kernels/layer:**

| Kernel | Function |
|--------|---------|
| `sdpafwd` | QKV projection + SDPA + output projection |
| `ffnFused` | SwiGLU FFN (W1, W3, SiLU, W2) |
| `ffnBwdW2t` / `ffnBwdW13t` | FFN backward (split for memory) |
| `sdpaBwd1` / `sdpaBwd2` | SDPA backward |

**GQA (Qwen3-0.6B) — 10 kernels/layer:** Adds `wofwd`, `qBwd`, `kvBwd` for grouped-query attention.

### Key optimizations

| Optimization | Impact |
|-------------|--------|
| Channel-first CPU layout `[1,C,1,S]` | Eliminates all transpose overhead |
| vDSP vectorized RMSNorm | 10x faster: 6.7ms → 0.7ms |
| GCD async cblas overlap | dW gradients parallel with ANE evals |
| Deferred cblas wait | Maximizes CPU/ANE overlap |
| ANE RMSNorm fusion | RMSNorm folded into MIL forward kernels |
| Wo^T fusion | Output proj backward merged into SDPA |
| exec() restart | Bypasses ~119 ANE compile limit/process |
| IOSurface zero-copy | 60–120x vs memcpy for GPU/CPU → ANE |
| INT8 W8A8 quantization | 1.88x throughput, halves L2 SRAM bandwidth |

---

## 🌐 Remote Access via Tailscale

Run ANE inference on your M-series Mac and access it from anywhere:

```bash
# On your ANE Mac:
sudo tailscale up
python python/ane_inference.py --server --host 0.0.0.0 --port 8189
```

From any device on your Tailscale network:

```python
import requests
resp = requests.post("http://100.xx.xx.xx:8189/generate", json={
    "prompt": "Explain attention mechanisms",
    "max_tokens": 256,
})
print(resp.json()["text"])
```

See [Hermit Purple Studio](https://github.com/DeadByDawn101/hermit-purple-studio#-remote-access-via-tailscale) for the full Tailscale setup guide.

---

## 🔗 Hermit Purple Studio Integration

This is the compute backend for [Hermit Purple Studio](https://github.com/DeadByDawn101/hermit-purple-studio):

```
Hermit Purple Studio (React UI)
        │
        ├── /generate ──────────► ANE (this repo) ── local M-series inference
        │                              │
        │                              └── fallback ──► Anthropic API (Opus)
        │
        └── /image ─────────────► ComfyUI (GPU) ── image generation
```

Both backends are Tailscale-accessible, unified under one Studio UI.

---

## ⚠️ Honest Limitations

| Limitation | Detail |
|-----------|--------|
| ANE utilization | Currently 5–9% of theoretical peak — significant engineering headroom remains |
| Element-wise ops | Many still fall back to CPU (RMSNorm, residuals, loss) |
| Model size | >1B models on ANE alone: not yet practical |
| Private APIs | `_ANEClient` / `_ANECompiler` may break on macOS updates |
| macOS only | The ANE side is macOS-only; rdma-core patterns are portable |
| Research status | Not production-ready — benchmark, experiment, fork, build |

---

## 📚 References

- [Part 1: Reverse Engineering the ANE](https://github.com/maderix/ANE) — maderix
- [Part 2: Benchmarks](https://github.com/maderix/ANE)
- [Part 3: Training](https://github.com/maderix/ANE)
- [rdma-core](https://github.com/linux-rdma/rdma-core) — Linux RDMA userspace stack (memory patterns adapted for ANE)
- [Apple Neural Engine internals](https://github.com/hollance/neural-engine) — Matthijs Hollemans
- [IOSurface framework](https://developer.apple.com/documentation/iosurface) — Apple

---

## 🎴 The Stand

> **Hermit Purple** — Range: A | Persistence: A
> *The vines reach anywhere. The ANE is the silicon they run through.*

Hermit Purple Studio reaches across servers, model hubs, and your local silicon — and brings everything back in one picture. Even from a thousand miles away, over Tailscale, running on the Neural Engine built into your Mac. When that's not enough, the Stand reaches further — to Claude Opus — and returns with what you need.

---

## 📄 License

MIT — fork it, build on it, do something cool.

Original ANE work by [maderix](https://github.com/maderix/ANE). rdma-core patterns from [linux-rdma](https://github.com/linux-rdma/rdma-core) (MIT/BSD licensed).
