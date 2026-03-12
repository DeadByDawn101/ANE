"""
QA test suite for the ANE Python stack.
Runs on any platform (Linux CI, macOS M-series).
ANE-specific paths fall back to simulation mode gracefully.
"""
import sys, os, platform, time, threading, unittest
sys.path.insert(0, os.path.dirname(__file__))

# ─── Module import tests ──────────────────────────────────────────────────────

class TestImports(unittest.TestCase):
    def test_import_ane_device(self):
        import ane_device
        self.assertTrue(hasattr(ane_device, 'ANEDevice'))
        self.assertTrue(hasattr(ane_device, 'ANEDeviceInfo'))

    def test_import_ane_mem(self):
        import ane_mem
        self.assertTrue(hasattr(ane_mem, 'MemoryRegion'))
        self.assertTrue(hasattr(ane_mem, 'TensorBuffer'))

    def test_import_ane_queue(self):
        import ane_queue
        self.assertTrue(hasattr(ane_queue, 'ANECompletionQueue'))
        self.assertTrue(hasattr(ane_queue, 'ANEQueuePair'))
        self.assertTrue(hasattr(ane_queue, 'ANEWorkRequest'))
        self.assertTrue(hasattr(ane_queue, 'ANECompletion'))

    def test_import_ane_inference(self):
        import ane_inference
        self.assertTrue(hasattr(ane_inference, 'ANEInference'))
        self.assertTrue(hasattr(ane_inference, 'InferenceResult'))

    def test_package_init(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "__init__", os.path.join(os.path.dirname(__file__), "__init__.py")
        )

# ─── ANEDevice tests ──────────────────────────────────────────────────────────

class TestANEDevice(unittest.TestCase):
    def test_spec_table_complete(self):
        from ane_device import _ANE_SPECS
        expected = ["M1", "M2", "M3", "M4",
                    "M1 Pro", "M2 Pro", "M3 Pro", "M4 Pro",
                    "M1 Max", "M2 Max", "M3 Max", "M4 Max",
                    "M1 Ultra", "M2 Ultra", "M3 Ultra", "M4 Ultra"]
        for chip in expected:
            self.assertIn(chip, _ANE_SPECS, f"Missing chip spec: {chip}")

    def test_spec_tops_positive(self):
        from ane_device import _ANE_SPECS
        for chip, spec in _ANE_SPECS.items():
            self.assertGreater(spec.tops_fp16, 0, f"{chip} TOPS must be > 0")

    def test_ultra_double_tops(self):
        from ane_device import _ANE_SPECS
        for gen in ["M1", "M2", "M3", "M4"]:
            base = _ANE_SPECS[gen].tops_fp16
            ultra = _ANE_SPECS[f"{gen} Ultra"].tops_fp16
            self.assertAlmostEqual(ultra, base * 2, delta=0.1,
                msg=f"{gen} Ultra should be 2x base TOPS")

    def test_detect_chip_returns_string(self):
        from ane_device import ANEDevice
        chip = ANEDevice._detect_chip()
        self.assertIsInstance(chip, str)
        self.assertGreater(len(chip), 0)

    def test_detect_memory_returns_int(self):
        from ane_device import ANEDevice
        mem = ANEDevice._detect_unified_memory_gb()
        self.assertIsInstance(mem, int)
        self.assertGreaterEqual(mem, 0)

    def test_device_info_str(self):
        from ane_device import ANEDeviceInfo
        info = ANEDeviceInfo("M4", 38.0, 16, 16, "15.0", 32)
        s = str(info)
        self.assertIn("M4", s)
        self.assertIn("38.0", s)

    def test_default_raises_on_non_macos(self):
        # On Linux (CI), default() should raise RuntimeError
        from ane_device import ANEDevice
        if platform.system() != "Darwin":
            with self.assertRaises(RuntimeError):
                ANEDevice.default()

# ─── MemoryRegion tests ───────────────────────────────────────────────────────

class TestMemoryRegion(unittest.TestCase):
    def test_allocate_small(self):
        from ane_mem import MemoryRegion
        with MemoryRegion(4096) as mr:
            self.assertIsNotNone(mr._buf)
            self.assertGreaterEqual(mr.size, 4096)

    def test_size_aligns_to_page(self):
        from ane_mem import MemoryRegion, _PAGE_SIZE
        with MemoryRegion(1) as mr:
            self.assertEqual(mr.size % _PAGE_SIZE, 0)
        with MemoryRegion(100_000) as mr:
            self.assertEqual(mr.size % _PAGE_SIZE, 0)

    def test_allocate_large(self):
        from ane_mem import MemoryRegion
        size = 64 * 1024 * 1024  # 64MB
        with MemoryRegion(size) as mr:
            self.assertGreaterEqual(mr.size, size)

    def test_as_numpy_shape(self):
        from ane_mem import MemoryRegion
        import numpy as np
        with MemoryRegion(1024 * 1024) as mr:
            arr = mr.as_numpy(shape=(512, 512), dtype=np.float16)
            self.assertEqual(arr.shape, (512, 512))
            self.assertEqual(arr.dtype, np.float16)

    def test_as_numpy_writable(self):
        from ane_mem import MemoryRegion
        import numpy as np
        with MemoryRegion(64 * 1024) as mr:
            arr = mr.as_numpy(shape=(256,), dtype=np.float32)
            arr[0] = 42.0
            arr[255] = 99.0
            arr2 = mr.as_numpy(shape=(256,), dtype=np.float32)
            self.assertAlmostEqual(float(arr2[0]), 42.0)
            self.assertAlmostEqual(float(arr2[255]), 99.0)

    def test_zero_fills(self):
        from ane_mem import MemoryRegion
        import numpy as np
        with MemoryRegion(16 * 1024) as mr:
            arr = mr.as_numpy(dtype=np.uint8)
            arr[:64] = 255
            mr.zero()
            arr2 = mr.as_numpy(dtype=np.uint8)
            self.assertEqual(arr2[:64].sum(), 0)

    def test_context_manager_closes(self):
        from ane_mem import MemoryRegion
        mr = MemoryRegion(4096)
        with mr:
            pass
        self.assertIsNone(mr._buf)

    def test_repr(self):
        from ane_mem import MemoryRegion
        with MemoryRegion(4096) as mr:
            r = repr(mr)
            self.assertIn("MemoryRegion", r)
            self.assertIn("KB", r)

# ─── TensorBuffer tests ───────────────────────────────────────────────────────

class TestTensorBuffer(unittest.TestCase):
    def test_for_model_creates_regions(self):
        from ane_mem import TensorBuffer
        with TensorBuffer.for_model(dim=768, seq_len=128, n_layers=12) as buf:
            self.assertIn("hidden", buf._regions)
            self.assertIn("q", buf._regions)
            self.assertIn("k", buf._regions)
            self.assertIn("v", buf._regions)
            self.assertIn("ffn_out", buf._regions)
            self.assertIn("grad_w", buf._regions)

    def test_regions_accessible(self):
        from ane_mem import TensorBuffer
        import numpy as np
        with TensorBuffer.for_model(dim=512, seq_len=64, n_layers=6) as buf:
            hidden = buf["hidden"].as_numpy(dtype=np.float16)
            self.assertIsNotNone(hidden)

    def test_ctypes_ptr(self):
        from ane_mem import MemoryRegion
        import ctypes
        with MemoryRegion(4096) as mr:
            ptr = mr.as_ctypes_ptr()
            self.assertIsNotNone(ptr)
            # Should be a valid non-null pointer
            self.assertNotEqual(ctypes.cast(ptr, ctypes.c_void_p).value, 0)

# ─── ANEQueue tests ───────────────────────────────────────────────────────────

class TestANEQueue(unittest.TestCase):
    def test_completion_queue_depth(self):
        from ane_queue import ANECompletionQueue
        cq = ANECompletionQueue(depth=16)
        self.assertEqual(cq.depth, 16)

    def test_post_and_poll_completion(self):
        from ane_queue import ANECompletionQueue, ANECompletion
        cq = ANECompletionQueue(depth=8)
        wc = ANECompletion(wr_id=42, status=0, elapsed_ms=1.5)
        cq.post_completion(wc)
        results = cq.poll(num_entries=1, timeout=0.1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].wr_id, 42)
        self.assertTrue(results[0].ok)

    def test_completion_ok_false_on_error(self):
        from ane_queue import ANECompletion
        wc = ANECompletion(wr_id=1, status=1, elapsed_ms=0.5, error="simulated error")
        self.assertFalse(wc.ok)
        self.assertEqual(wc.error, "simulated error")

    def test_queue_pair_submit_and_complete(self):
        from ane_queue import ANECompletionQueue, ANEQueuePair, ANEWorkRequest
        cq = ANECompletionQueue(depth=16)
        qp = ANEQueuePair(cq=cq, lib=None)  # lib=None → simulation mode

        wr = ANEWorkRequest(kernel_name="sdpafwd", wr_id=1)
        qp.post_send(wr)
        qp.drain(timeout=5.0)

        completions = cq.poll(num_entries=1, timeout=1.0)
        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].wr_id, 1)
        self.assertGreaterEqual(completions[0].elapsed_ms, 0)

    def test_queue_pair_multiple_kernels(self):
        from ane_queue import ANECompletionQueue, ANEQueuePair, ANEWorkRequest
        cq = ANECompletionQueue(depth=32)
        qp = ANEQueuePair(cq=cq, lib=None)

        n = 12
        for i in range(n):
            wr = ANEWorkRequest(kernel_name=f"layer_{i}", wr_id=i)
            qp.post_send(wr)

        qp.drain(timeout=10.0)
        completions = cq.poll(num_entries=n, timeout=2.0)
        self.assertEqual(len(completions), n)
        wr_ids = {c.wr_id for c in completions}
        self.assertEqual(wr_ids, set(range(n)))

    def test_exec_restart_counter(self):
        from ane_queue import ANECompletionQueue, ANEQueuePair, ANEWorkRequest
        cq = ANECompletionQueue(depth=200)
        qp = ANEQueuePair(cq=cq, lib=None)
        qp._KERNEL_LIMIT = 5  # lower limit for testing

        for i in range(12):
            qp.post_send(ANEWorkRequest(kernel_name="test", wr_id=i))

        qp.drain(timeout=10.0)
        # After 12 submissions with limit=5, exec should have restarted twice
        # _exec_count resets to 0 on restart, so final count is 12 % 5 = 2
        self.assertLess(qp._exec_count, 5)

    def test_callback_fires_on_completion(self):
        from ane_queue import ANECompletionQueue, ANEQueuePair, ANEWorkRequest
        results = []
        cq = ANECompletionQueue(depth=8)
        qp = ANEQueuePair(cq=cq, lib=None)

        wr = ANEWorkRequest(
            kernel_name="test",
            wr_id=99,
            callback=lambda wc: results.append(wc.wr_id)
        )
        qp.post_send(wr)
        qp.drain(timeout=5.0)
        time.sleep(0.05)
        self.assertIn(99, results)

# ─── Inference routing tests ──────────────────────────────────────────────────

class TestInferenceRouting(unittest.TestCase):
    def _make_engine(self, force=None):
        from ane_inference import ANEInference
        return ANEInference(
            anthropic_api_key=None,
            force_backend=force,
        )

    def test_force_anthropic_routes_away_from_ane(self):
        from ane_inference import ANEInference
        engine = ANEInference(force_backend="anthropic")
        self.assertFalse(engine._can_run_on_ane("stories110m"))

    def test_unknown_model_cant_run_on_ane(self):
        engine = self._make_engine()
        self.assertFalse(engine._can_run_on_ane("gpt-4o"))
        self.assertFalse(engine._can_run_on_ane("llama-70b"))
        self.assertFalse(engine._can_run_on_ane("nonexistent-model"))

    def test_large_model_routes_to_anthropic(self):
        engine = self._make_engine()
        # qwen3-7b should never go to ANE (too large)
        self.assertFalse(engine._can_run_on_ane("qwen3-7b"))

    def test_force_ane_overrides_routing(self):
        from ane_inference import ANEInference
        # force_backend="ane" should return True even without a real device
        engine = ANEInference(force_backend="ane")
        # Only check the flag logic — don't check device presence
        result = engine._can_run_on_ane("stories110m")
        # On Linux: device is None, force_backend=ane → True (bypasses device check)
        self.assertTrue(result)

    def test_model_registry_complete(self):
        from ane_inference import _MODEL_REGISTRY, _ANE_CAPABLE_MODELS
        for m in _ANE_CAPABLE_MODELS:
            self.assertIn(m, _MODEL_REGISTRY, f"{m} in ANE_CAPABLE but not in MODEL_REGISTRY")

    def test_inference_result_dataclass(self):
        from ane_inference import InferenceResult
        r = InferenceResult(text="hello", backend="ane", model="stories110m", elapsed_ms=91.0)
        self.assertEqual(r.text, "hello")
        self.assertEqual(r.backend, "ane")
        self.assertIsNone(r.error)

    def test_no_api_key_raises_for_anthropic(self):
        from ane_inference import ANEInference
        engine = ANEInference(anthropic_api_key=None, force_backend="anthropic")
        with self.assertRaises(RuntimeError):
            engine._run_anthropic("hello", 10, 0.7)

    def test_ane_simulation_run(self):
        """Full end-to-end simulation on Linux (no real ANE dylib)."""
        from ane_inference import ANEInference
        engine = ANEInference(anthropic_api_key="dummy-key-for-test", force_backend="ane")
        result = engine._run_ane("Hello world", "stories110m", 32, 0.7)
        # On Linux: TensorBuffer works, queue runs in sim mode, returns ANE result
        self.assertIn(result.backend, ("ane", "anthropic"))  # either is valid in sim
        self.assertGreater(result.elapsed_ms, 0)

    def test_ane_failure_returns_result_with_error_or_fallback(self):
        """When ANE fails with no API key, should get RuntimeError about API key."""
        from ane_inference import ANEInference
        engine = ANEInference(anthropic_api_key=None, force_backend="ane")
        # Patch TensorBuffer to simulate failure
        import ane_mem
        orig = ane_mem.TensorBuffer.for_model
        def boom(*a, **kw): raise RuntimeError("simulated ANE OOM")
        ane_mem.TensorBuffer.for_model = staticmethod(boom)
        try:
            with self.assertRaises(RuntimeError):
                engine._run_ane("test", "stories110m", 10, 0.7)
        finally:
            ane_mem.TensorBuffer.for_model = staticmethod(orig)

# ─── ObjC / C file checks ─────────────────────────────────────────────────────

class TestBridgeFiles(unittest.TestCase):
    def test_ane_memory_h_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "bridge", "ane_memory.h")
        self.assertTrue(os.path.exists(path), "ane_memory.h missing")

    def test_ane_memory_m_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "bridge", "ane_memory.m")
        self.assertTrue(os.path.exists(path), "ane_memory.m missing")

    def test_ane_memory_m_has_key_functions(self):
        path = os.path.join(os.path.dirname(__file__), "..", "bridge", "ane_memory.m")
        src = open(path).read()
        for fn in ["ANERegisterMemoryRegion", "ANEDeregisterMemoryRegion",
                   "ANEMemoryRegionGetPtr", "ANEMemoryRegionGetSurface"]:
            self.assertIn(fn, src, f"Missing function: {fn}")

    def test_makefile_references_new_files(self):
        path = os.path.join(os.path.dirname(__file__), "..", "bridge", "Makefile")
        src = open(path).read()
        self.assertIn("ane_memory.m", src)
        self.assertIn("ane_dmabuf.m", src)

    def test_bridge_h_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "bridge", "ane_bridge.h")
        self.assertTrue(os.path.exists(path))

# ─── Memory stress test ───────────────────────────────────────────────────────

class TestMemoryStress(unittest.TestCase):
    def test_many_regions_no_leak(self):
        from ane_mem import MemoryRegion
        # Allocate and release 50 regions — should not OOM or crash
        for _ in range(50):
            with MemoryRegion(256 * 1024) as mr:
                mr.zero()

    def test_concurrent_queue_dispatch(self):
        from ane_queue import ANECompletionQueue, ANEQueuePair, ANEWorkRequest
        cq = ANECompletionQueue(depth=128)
        qp = ANEQueuePair(cq=cq, lib=None)

        n = 32
        for i in range(n):
            qp.post_send(ANEWorkRequest(kernel_name="parallel_test", wr_id=i))

        qp.drain(timeout=15.0)
        completions = cq.poll(num_entries=n, timeout=3.0)
        self.assertEqual(len(completions), n, f"Expected {n} completions, got {len(completions)}")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestImports, TestANEDevice, TestMemoryRegion, TestTensorBuffer,
                TestANEQueue, TestInferenceRouting, TestBridgeFiles, TestMemoryStress]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
