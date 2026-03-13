"""
ane_inference.py — Tiered inference: ANE local → Claude Opus fallback.

Small/fast tasks run on the Apple Neural Engine (free, private, ~1W).
When the task exceeds local capacity, routes to Claude Opus via Anthropic API.
"""
import os
import time
from dataclasses import dataclass
from typing import Optional

from ane_device import ANEDevice
from ane_mem import MemoryRegion, TensorBuffer
from ane_queue import ANECompletionQueue, ANEQueuePair, ANEWorkRequest


# --- Model registry -----------------------------------------------------------
# Maps model name → (param_count_B, dim, n_layers, min_sram_mb)
_MODEL_REGISTRY = {
    "stories110m":  (0.11,  768, 12, 256),
    "qwen3-0.6b":   (0.6,  1024, 28, 512),
    "qwen3-1.5b":   (1.5,  2048, 28, 1024),
    "qwen3-3b":     (3.0,  2560, 36, 2048),
    "qwen3-7b":     (7.0,  3584, 32, 4096),
}

# Models we can attempt on ANE (≤1B today; expand as ANE throughput improves)
_ANE_CAPABLE_MODELS = {"stories110m", "qwen3-0.6b"}


@dataclass
class InferenceResult:
    text: str
    backend: str          # "ane" | "anthropic"
    model: str
    elapsed_ms: float
    tokens: int = 0
    error: Optional[str] = None


class ANEInference:
    """
    Tiered inference engine for Apple M-series Macs.

    Routing logic:
        1. If model is ANE-capable and fits in unified memory → run on ANE
        2. Otherwise → call Anthropic API (Claude Opus)

    Usage:
        engine = ANEInference(anthropic_api_key="sk-ant-...")
        result = engine.generate("Explain attention", model="qwen3-0.6b")
        print(result.backend, result.text)
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        fallback_model: str = "claude-opus-4-5",
        force_backend: Optional[str] = None,  # "ane" | "anthropic" | None
    ):
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.fallback_model = fallback_model
        self.force_backend = force_backend

        # Discover ANE device
        try:
            self.device = ANEDevice.default()
        except RuntimeError as e:
            print(f"[ANEInference] Warning: {e}")
            self.device = None

        # Shared completion queue (mirrors ibv_create_cq)
        self._cq = ANECompletionQueue(depth=64)
        self._qp = ANEQueuePair(cq=self._cq, lib=self._load_bridge())

    def _load_bridge(self):
        import platform
        if platform.system() != "Darwin":
            return None  # dylib is macOS/ARM64 only
        bridge_path = os.path.join(
            os.path.dirname(__file__), "..", "bridge", "libane_bridge.dylib"
        )
        bridge_path = os.path.abspath(bridge_path)
        if os.path.exists(bridge_path):
            try:
                import ctypes
                return ctypes.CDLL(bridge_path)
            except OSError:
                return None
        return None

    def _can_run_on_ane(self, model: str) -> bool:
        if self.force_backend == "anthropic":
            return False
        if self.force_backend == "ane":
            return True
        if self.device is None:
            return False
        if model not in _ANE_CAPABLE_MODELS:
            return False
        # Check if model fits in unified memory
        params_b, dim, n_layers, min_sram = _MODEL_REGISTRY.get(model, (99, 0, 0, 99999))
        if params_b > (self.device.unified_memory_gb * 0.5):  # use ≤50% of unified memory
            return False
        return True

    def generate(
        self,
        prompt: str,
        model: str = "qwen3-0.6b",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> InferenceResult:
        """
        Generate text. Routes to ANE or Anthropic API automatically.
        """
        if self._can_run_on_ane(model):
            return self._run_ane(prompt, model, max_tokens, temperature)
        else:
            return self._run_anthropic(prompt, max_tokens, temperature)

    def _run_ane(self, prompt: str, model: str, max_tokens: int, temperature: float) -> InferenceResult:
        """Run inference on the Apple Neural Engine."""
        t0 = time.monotonic()
        params_b, dim, n_layers, _ = _MODEL_REGISTRY[model]
        seq_len = min(len(prompt.split()) + max_tokens, 2048)

        try:
            # Allocate zero-copy tensor buffers (RDMA-style pinned memory)
            with TensorBuffer.for_model(dim, seq_len, n_layers) as bufs:

                # Submit forward pass kernels via the ANE queue pair
                # (mirrors ibv_post_send for each layer)
                wr_ids = []
                for layer_idx in range(n_layers):
                    wr = ANEWorkRequest(
                        kernel_name="sdpafwd",
                        input_ptr=bufs["hidden"].as_ctypes_ptr().value,
                        output_ptr=bufs["attn_out"].as_ctypes_ptr().value,
                        input_size=bufs["hidden"].size,
                        output_size=bufs["attn_out"].size,
                    )
                    wr_id = self._qp.post_send(wr)
                    wr_ids.append(wr_id)

                # Poll completions (mirrors ibv_poll_cq)
                self._qp.drain(timeout=30.0)
                completions = self._cq.poll(num_entries=len(wr_ids), timeout=1.0)

                elapsed = (time.monotonic() - t0) * 1000
                errors = [c.error for c in completions if not c.ok]

                if errors:
                    raise RuntimeError(f"ANE kernel errors: {errors}")

                # In a real implementation, decode token ids from output buffer.
                # For now return a placeholder that indicates ANE ran successfully.
                return InferenceResult(
                    text=f"[ANE output — model={model}, layers={n_layers}, dim={dim}]",
                    backend="ane",
                    model=model,
                    elapsed_ms=elapsed,
                    tokens=max_tokens,
                )

        except Exception as e:
            # ANE failure → fall back to Anthropic
            print(f"[ANEInference] ANE failed ({e}), falling back to Anthropic API")
            return self._run_anthropic(prompt, max_tokens, temperature, ane_error=str(e))

    def _run_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        ane_error: Optional[str] = None,
    ) -> InferenceResult:
        """Call Claude Opus via Anthropic API."""
        if not self.api_key:
            raise RuntimeError(
                "No Anthropic API key. Set ANTHROPIC_API_KEY or pass anthropic_api_key= to ANEInference."
            )

        t0 = time.monotonic()
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model=self.fallback_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
            elapsed = (time.monotonic() - t0) * 1000
            return InferenceResult(
                text=text,
                backend="anthropic",
                model=self.fallback_model,
                elapsed_ms=elapsed,
                tokens=msg.usage.output_tokens,
            )
        except ImportError:
            raise RuntimeError("pip install anthropic to enable Opus fallback")
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return InferenceResult(
                text="",
                backend="anthropic",
                model=self.fallback_model,
                elapsed_ms=elapsed,
                error=str(e),
            )


# --- CLI / server mode --------------------------------------------------------

def _serve(host: str, port: int, engine: ANEInference):
    """Simple HTTP server for remote access via Tailscale."""
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # suppress default logging

        def _send_json(self, data: dict, code: int = 200):
            resp = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(resp))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp)

        def do_OPTIONS(self):
            """CORS preflight — allow the Vite dev server at any port."""
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if self.path == "/generate":
                result = engine.generate(
                    prompt=body.get("prompt", ""),
                    model=body.get("model", "qwen3-0.6b"),
                    max_tokens=body.get("max_tokens", 256),
                    temperature=body.get("temperature", 0.7),
                )
                self._send_json({
                    "text": result.text,
                    "backend": result.backend,
                    "model": result.model,
                    "elapsed_ms": result.elapsed_ms,
                    "tokens": result.tokens,
                    "error": result.error,
                })

            elif self.path == "/prompt-enhance":
                # Use ANE (or Anthropic fallback) to rewrite a ComfyUI prompt.
                # Returns an enriched prompt with lighting, style, and detail cues.
                raw_prompt = body.get("prompt", "")
                style = body.get("style", "photorealistic")
                enhance_prompt = (
                    f"Rewrite the following image generation prompt to be more detailed and vivid "
                    f"for a {style} style. Add specific lighting, mood, composition, and detail cues. "
                    f"Keep it under 120 words. Return ONLY the improved prompt, no explanation.\n\n"
                    f"Original: {raw_prompt}"
                )
                result = engine.generate(
                    prompt=enhance_prompt,
                    model="qwen3-0.6b",
                    max_tokens=160,
                    temperature=0.6,
                )
                self._send_json({
                    "enhanced": result.text.strip(),
                    "original": raw_prompt,
                    "backend": result.backend,
                    "elapsed_ms": result.elapsed_ms,
                    "error": result.error,
                })

            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self._send_json({"status": "ok"})

            elif self.path == "/status":
                # Device info — consumed by useANE.js on connect
                dev = engine.device
                info = {
                    "status": "ok",
                    "device": str(dev) if dev else "simulation",
                    "chip": getattr(dev, "chip", "unknown"),
                    "ne_cores": getattr(dev, "ne_cores", 16),
                    "ne_tops": getattr(dev, "ne_tops", 38),
                    "ram_gb": getattr(dev, "ram_gb", None),
                    "macos": getattr(dev, "macos_version", None),
                    "ane_capable_models": list(_ANE_CAPABLE_MODELS),
                }
                self._send_json(info)

            elif self.path == "/models":
                self._send_json({
                    "models": list(_MODEL_REGISTRY.keys()),
                    "ane_capable": list(_ANE_CAPABLE_MODELS),
                })

            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer((host, port), Handler)
    print(f"[ANEInference] Server running on http://{host}:{port}")
    print(f"[ANEInference] Device: {engine.device}")
    server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ANE Inference — local or server mode")
    parser.add_argument("--server", action="store_true", help="Run as HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8189)
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--prompt", default="Explain gradient checkpointing in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--force-backend", choices=["ane", "anthropic"], default=None)
    args = parser.parse_args()

    engine = ANEInference(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        force_backend=args.force_backend,
    )

    if args.server:
        _serve(args.host, args.port, engine)
    else:
        result = engine.generate(args.prompt, model=args.model, max_tokens=args.max_tokens)
        print(f"\nBackend : {result.backend}")
        print(f"Model   : {result.model}")
        print(f"Time    : {result.elapsed_ms:.1f}ms")
        print(f"Output  :\n{result.text}")
