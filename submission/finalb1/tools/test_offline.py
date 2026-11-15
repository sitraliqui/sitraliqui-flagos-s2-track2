"""CPU-only contract/lifecycle checks. These do not establish GPU correctness."""
import ast
import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "vllm-plugin-FL/vllm_fl/compilation/kv_decode"
pkg = types.ModuleType("kv_test")
pkg.__path__ = [str(PACKAGE)]
sys.modules["kv_test"] = pkg
contract = importlib.import_module("kv_test.contract")
runtime = importlib.import_module("kv_test.runtime")
controller_mod = importlib.import_module("kv_test.controller")
NS = types.SimpleNamespace


def host_rows(active=63, cap=64, length=4097):
    width = contract.TABLE_COLUMNS
    pages = (length + 15) // 16
    rows = [list(range(i * width, (i + 1) * width)) for i in range(active)]
    slots = [r[pages - 1] * 16 + (length - 1) % 16 for r in rows]
    return dict(query_start=list(range(active + 1)) + [active] * (cap - active),
                lengths=[length] * active + [0] * (cap - active),
                table=rows + [[0] * width for _ in range(cap - active)],
                slots=slots + [-1] * (cap - active))


class MetadataTests(unittest.TestCase):
    def test_all_admitted_boundaries(self):
        for cap, counts in contract.BATCHES.items():
            for active in counts:
                for n in contract.PROOF_LENGTHS:
                    with self.subTest(cap=cap, active=active, length=n):
                        self.assertIsNone(contract.route_reason(
                            cap, [1] * active, [n] * active,
                            [n - 1] * active, [4096] * active, active))
                        self.assertIsNone(contract.metadata_reason(
                            host_rows(active, cap, n), cap * contract.TABLE_COLUMNS, active))

    def test_page_alias_and_wrong_latest_slot_are_rejected(self):
        h = host_rows()
        h["table"][1][0] = h["table"][0][0]
        self.assertEqual(contract.metadata_reason(h, 20480, 63), "shared_readable_page")
        h = host_rows()
        h["slots"][0] -= 1
        self.assertEqual(contract.metadata_reason(h, 20480, 63), "write_not_at_latest_token")

    def test_inactive_rows_are_never_writable(self):
        for key, value in (("slots", 0), ("lengths", 1)):
            h = host_rows()
            h[key][-1] = value
            self.assertEqual(contract.metadata_reason(h, 20480, 63), "invalid_padding")
        h = host_rows()
        h["table"][-1][0] = 1
        self.assertEqual(contract.metadata_reason(h, 20480, 63), "invalid_padding_table")

    def test_outside_lengths_prefill_and_mixed_fall_back(self):
        for n in (1, 4096, 5120, 16384):
            self.assertEqual(contract.route_reason(32, [1] * 32, [n] * 32,
                             [n - 1] * 32, [0] * 32, 32), "context_bound")
        self.assertEqual(contract.route_reason(32, [2] + [1] * 31, [4097] * 32,
                         [4096] * 32, [4096] * 32, 32), "mixed_or_prefill")
        self.assertEqual(contract.route_reason(32, [1] * 32, [4097] * 32,
                         [4096] * 32, [5000] * 32, 32), "mixed_or_prefill")


class LifecycleTests(unittest.TestCase):
    def test_nested_runner_exception_restores_all_scopes(self):
        c = NS(step=None, capture_owner=None, capture_runner=None)
        r = object()

        @runtime.runner_scope("dummy")
        def dummy(runner, n, is_graph_capturing=False):
            self.assertIs(runtime.active_controller(), c)
            self.assertIs(c.capture_runner, r)
            raise RuntimeError("capture_failure")

        @runtime.runner_scope("capture")
        def capture(runner):
            self.assertIs(c.capture_owner, r)
            dummy(runner, 32, True)

        with patch.object(runtime, "controller_for", return_value=c):
            with self.assertRaisesRegex(RuntimeError, "capture_failure"):
                capture(r)
        self.assertIsNone(runtime.active_controller())
        self.assertEqual((c.step, c.capture_owner, c.capture_runner), (None, None, None))

    def test_pool_and_entries_restore_on_capture_error(self):
        original, pool = {}, object()
        w = NS(concrete_graph_entries=original, graph_pool=pool)
        calls = []
        with self.assertRaises(ValueError):
            with controller_mod.graph_entries(w, {"candidate": 1}, object(),
                                              lambda wrapper: calls.append(wrapper.graph_pool)):
                raise ValueError("owned failure")
        self.assertIs(w.concrete_graph_entries, original)
        self.assertIs(w.graph_pool, pool)
        self.assertEqual(calls, [pool])

    def make_replay(self, reject=None, stage_fail=False):
        c = controller_mod.KVDecodeController.__new__(controller_mod.KVDecodeController)
        desc = types.SimpleNamespace(num_tokens=32)
        # Batch descriptors in the real API are immutable/hashable.
        class Descriptor:
            num_tokens = 32
        desc = Descriptor()
        mode = NS(name="FULL")
        ctx = NS(batch_descriptor=desc, cudagraph_runtime_mode=mode)
        native_entry, candidate_entry = NS(graph=object()), NS(graph=object())
        staged_calls, native_calls = [], []
        class Destination:
            def copy_(self, value):
                staged_calls.append(value)
                if stage_fail:
                    raise RuntimeError("staging_failure")
        bundle = {"staged": [Destination() for _ in range(4)]}
        w = NS(concrete_graph_entries={desc: native_entry}, graph_pool=object(),
               runtime_mode=mode, kv_decode_variants={})
        entries = {desc: candidate_entry}
        w.kv_decode_variants["4k-s16"] = {desc: (entries, object(), bundle, native_entry)}
        w._call_native = lambda *a, **k: native_calls.append(
            "candidate" if w.concrete_graph_entries is entries else "native") or "out"
        class Lengths:
            def __getitem__(self, value):
                return self
            def tolist(self):
                return [4097] * 32
        runner = NS(model=w, input_batch=NS(num_reqs=32), optimistic_seq_lens_cpu=Lengths())
        c.profiles = (controller_mod.FOUR_K,)
        c.context_available = lambda: True
        c.context = lambda: ctx
        c.capture_runner = None
        c.step = runner, (1,) * 32
        c.event = lambda *a: None
        c.admission = lambda *a: reject if reject else (
            dict(zip(controller_mod.META, range(4))), {})
        return c, w, staged_calls, native_calls

    def test_rejected_route_never_stages_or_runs_candidate(self):
        c, w, stage, calls = self.make_replay("context_bound")
        self.assertEqual(c.call(w), "out")
        self.assertEqual((stage, calls), ([], ["native"]))

    def test_staging_failure_cannot_fall_back_and_write_twice(self):
        c, w, stage, calls = self.make_replay(stage_fail=True)
        with self.assertRaisesRegex(RuntimeError, "staging_failure"):
            c.call(w)
        self.assertEqual((stage, calls), ([0], []))

    def test_admitted_route_stages_once_then_replays_candidate(self):
        c, w, stage, calls = self.make_replay()
        self.assertEqual(c.call(w), "out")
        self.assertEqual((stage, calls), ([0, 1, 2, 3], ["candidate"]))

    def test_clear_releases_both_native_and_candidate_entries(self):
        tree = ast.parse((PACKAGE.parent / "graph.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GraphWrapper")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "clear_graphs")
        ns = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "graph.py", "exec"), ns)
        w = NS(kv_decode_variants={1: object()}, concrete_graph_entries={2: object()})
        ns["clear_graphs"](w)
        self.assertEqual((w.kv_decode_variants, w.concrete_graph_entries), ({}, {}))

    def test_unified_capture_keeps_profile_scopes_separate_on_exception(self):
        from kv_test.profiles import FOUR_K, SIXTEEN_K
        c, w, _, _ = self.make_replay()
        c.profiles = (FOUR_K, SIXTEEN_K)
        c.capture_runner = c.step[0]
        calls = []
        def invoke(*args, **kwargs):
            calls.append(c.policy.name)
            if c.policy is SIXTEEN_K:
                raise RuntimeError("second capture failed")
        c._call_profile = invoke
        with self.assertRaisesRegex(RuntimeError, "second capture failed"):
            c.call(w)
        self.assertEqual(calls, ["4k-s16", "16k-s32"])
        self.assertIs(c.policy, FOUR_K)

    def test_profile_selection_disjoint_and_mixed_context_falls_back(self):
        from kv_test.profiles import FOUR_K, SIXTEEN_K, select_profile
        both = (FOUR_K, SIXTEEN_K)
        for active in range(17, 33):
            for length in (16384, 16385, 16399, 16400, 16401, 17406, 17407):
                self.assertIs(select_profile(both, 32, active, [length] * active), SIXTEEN_K)
                self.assertIsNone(SIXTEEN_K.route_reason(32, [1]*active, [length]*active,
                                                       [length-1]*active, [100]*active, active))
        self.assertIs(select_profile(both, 1, 1, [16384]), SIXTEEN_K)
        self.assertIs(select_profile(both, 32, 32, [4097]*32), FOUR_K)
        for length in (16383, 17408):
            self.assertIsNone(select_profile(both, 32, 32, [length]*32))
        self.assertIsNone(select_profile(both, 32, 32, [4097]*16 + [16384]*16))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT / "offline-tests.json").write_text(json.dumps({
        "tests": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "passed": result.wasSuccessful(),
        "scope": "CPU contract and lifecycle only; no installed GPU-stack validation",
    }, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
