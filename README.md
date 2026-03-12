# ⚡ ANE — Apple Neural Engine Training & Inference Bridge

> *Training and running neural networks directly on Apple Silicon's Neural Engine — no CoreML, no GPU required.*

Forked from [maderix/ANE](https://github.com/maderix/ANE). This fork extends the project toward **practical LLM inference acceleration** on Apple Silicon, with a focus on running large language models (including Claude Opus-class models) using ANE as the primary compute engine — with the Anthropic API as a fallback when local capacity is exceeded.

![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-black?style=flat-square&logo=apple)
![Language](https://img.shields.io/badge/Language-Objective--C%20%2F%20C-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🧠 What Is the Apple Neural Engine?

The ANE is a dedicated neural accelerator built into every Apple Silicon chip (M1–M4, A-series). Apple exposes it only through CoreML for inference — and not at all for training. This project reverse-engineers the private `_ANEClient` / `_ANECompiler` APIs to run custom compute graphs — including full transformer forward + backward passes — directly on ANE hardware.

**ANE specs (M4):** 15.8 TOPS FP16 inference accelerator. Runs at extremely low power vs. GPU. Always-on, no warm-up.

---

## 🎯 Vision: ANE-First LLM Inference with Opus Fallback

The goal of this fork is to build a **tiered inference system** for use with [Hermit Purple Studio](https://github.com/DeadByDawn101/hermit-purple-studio) and standalone use:

```
Prompt → ANE Local Inference (fast, free, private)
              │
              ├─ Fits in ANE/RAM? → Run locally on ANE
              │
              └─ Too large / complex? → Route to Anthropic API (Claude Opus)
```

This gives you:
- **Small/medium tasks** → run on your Mac's ANE for free, instantly, privately
- **Heavy tasks** → automatically escalate to Claude Opus via API
- **No GPU required** — ANE runs cool and quiet while you work
- **Tailscale-accessible** — remote access via [Hermit Purple Studio](https://github.com/DeadByDawn101/hermit-purple-studio#-remote-access-via-tailscale)

---

## 📊 Current Benchmark Results

| Model | Params | ms/step | Pipeline |
|-------|--------|---------|---------|
| Stories110M (12L, dim=768, MHA 12/12) | 109M | 91 ms | Dynamic (no recompile) |
| Qwen3-0.6B (28L, dim=1024, GQA 16/8) | 596M | 412 ms | Dynamic (no recompile) |

**INT8 W8A8 quantization — 1.88x throughput (M4, H16G):**

| Config | FP16 | INT8 W8A8 | Speedup |
|--------|------|-----------|---------|
| 128x conv 512ch 64x64 | 18.6 TOPS, 14.8ms | 35.1 TOPS, 7.8ms | **1.88x** |
| 64x conv 512ch 64x64 | 18.4 TOPS, 7.5ms | 34.1 TOPS, 4.0ms | **1.85x** |

---

## 🏗 Architecture

### What runs where

```
┌─────────────────────────────────────────────────────┐
│                   Apple Silicon Mac                  │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌────────────────┐ │
│  │   CPU    │    │   ANE    │    │   GPU (Metal)  │ │
│  │          │    │ 15.8T FP │    │   (optional)   │ │
│  │ RMSNorm  │    │  MatMul  │    │  GPU prefill   │ │
│  │ Residual │◄──►│  SDPA    │◄──►│  (IOSurface    │ │
│  │ Adam opt │    │  FFN     │    │   zero-copy)   │ │
│  │ cblas dW │    │  Attn    │    └────────────────┘ │
│  └──────────┘    └──────────┘                        │
│        │               │                             │
│        └───── bridge/ ─┘                             │
│            libane_bridge.dylib                       │
│            Python ctypes interface                   │
└─────────────────────────────────────────────────────┘
              │ (when local capacity exceeded)
              ▼
     Anthropic API (Claude Opus)
```

### Kernel layout (MHA — Stories110M, 6 kernels/layer)

| Kernel | Function |
|--------|---------|
| `sdpafwd` | QKV projection + SDPA + output projection |
| `ffnFused` | SwiGLU FFN (W1, W3, SiLU, W2) |
| `ffnBwdW2t` / `ffnBwdW13t` | FFN backward (split for memory) |
| `sdpaBwd1` / `sdpaBwd2` | SDPA backward |

### Key optimizations

- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates transpose overhead
- **vDSP vectorized RMSNorm** — 10x faster than naïve (6.7ms → 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals
- **Deferred cblas wait** — wait pushed into next step's forward pass for maximum overlap
- **ANE RMSNorm fusion** — RMSNorm folded into forward kernels as MIL ops
- **Wo^T fusion** — output projection backward merged into SDPA backward kernel
- **Forward taps** — Q, K, V, attention scores, hidden states exposed via concat outputs
- **exec() restart** — bypasses ~119 ANE compile limit per process

---

## 📁 File Structure

```
ane/
├── api_exploration.m          # Initial ANE API discovery
├── inmem_basic.m              # In-memory MIL compilation proof-of-concept
├── inmem_bench.m              # Core ANE benchmark harness
├── inmem_peak.m               # Peak throughput measurement
├── sram_bench.m               # SRAM bandwidth benchmarks
├── sram_probe.m               # SRAM capacity probing
├── ane_int8_bench.m           # INT8 W8A8 quantization benchmarks
│
├── bridge/
│   ├── ane_bridge.h           # C header — Python ctypes interface
│   ├── ane_bridge.m           # Objective-C bridge implementation
│   ├── libane_bridge.dylib    # Pre-built dylib (M-series Mac required)
│   └── Makefile               # Build the dylib yourself
│
├── training/                  # Full transformer training on ANE
│   └── ...                    # Forward + backward pass, Adam optimizer
│
└── benchmarks/                # Benchmark results and reports
```

---

## 🚀 Getting Started

### Requirements

- Apple Silicon Mac (M1 or newer; M4 recommended for best throughput)
- macOS 13+ (Ventura or newer)
- Xcode Command Line Tools
- Python 3.10+ (for the bridge / Python interface)

### 1. Clone

```bash
git clone https://github.com/DeadByDawn101/ANE.git
cd ANE
```

### 2. Build the bridge dylib

```bash
cd bridge
make
# Produces: libane_bridge.dylib
```

### 3. Run a benchmark

```bash
# Basic in-memory benchmark
clang -framework Foundation -framework CoreML -o inmem_bench inmem_bench.m
./inmem_bench

# INT8 W8A8 benchmark
clang -framework Foundation -framework CoreML -o ane_int8_bench ane_int8_bench.m
./ane_int8_bench
```

### 4. Use the Python bridge

```python
import ctypes

lib = ctypes.CDLL("bridge/libane_bridge.dylib")

# Initialize ANE session
lib.ane_init()

# Run a model
lib.ane_run_model(...)
```

---

## 🔗 Integration with Hermit Purple Studio

This repo is designed to integrate with [Hermit Purple Studio](https://github.com/DeadByDawn101/hermit-purple-studio) as a local inference backend. The planned architecture:

```
Hermit Purple Studio (React UI)
        │
        ▼
    ComfyUI API   ←── image generation (GPU)
        │
    ANE Bridge    ←── text / LLM inference (ANE)
        │
        ├── Small models (≤1B params): run on ANE directly
        ├── Medium models (1–7B): ANE + Metal hybrid
        └── Large / complex: Anthropic API → Claude Opus
```

The `bridge/libane_bridge.dylib` + Python ctypes layer means any Python-based inference stack (llama.cpp Python bindings, MLX, custom) can dispatch to ANE without Objective-C knowledge.

---

## 🌐 Remote Access via Tailscale

Since the ANE is local to your Mac, remote access works the same way as Hermit Purple Studio — via Tailscale:

```bash
# On your ANE Mac:
sudo tailscale up
python server.py --host 0.0.0.0 --port 8189

# From anywhere:
# http://100.xx.xx.xx:8189
```

See the [Hermit Purple Studio Tailscale guide](https://github.com/DeadByDawn101/hermit-purple-studio#-remote-access-via-tailscale) for full setup instructions.

---

## ⚠️ Honest Limitations

This is a **research project**, not a production inference stack. Be aware:

- ANE utilization is currently 5–9% of theoretical peak
- Many element-wise operations still fall back to CPU
- Not a replacement for GPU inference for models >1B today
- Private APIs may break with macOS updates
- No guarantee of stability across macOS versions

The benchmarks and all limitations are documented honestly in the accompanying articles:
- [Part 1: Reverse Engineering](https://github.com/maderix/ANE)
- [Part 2: Benchmarks](https://github.com/maderix/ANE)
- [Part 3: Training](https://github.com/maderix/ANE)

---

## 🎴 The Stand Connection

> *Hermit Purple's vines reach anywhere. The ANE is the silicon those vines run through.*

This project powers the local compute layer of the Hermit Purple ecosystem — fast, quiet, always-on inference from the neural engine built into your Apple Silicon chip. When the task exceeds local capacity, the Stand reaches further — out to Claude Opus via the Anthropic API — and brings back what you need.

---

## 📄 License

MIT — fork it, build on it, do something cool.

Original project by [maderix](https://github.com/maderix/ANE).
