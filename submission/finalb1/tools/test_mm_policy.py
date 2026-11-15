"""CPU exclusions for bounded numerical fix; not a performance claim."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/mm_stability.py"
spec = importlib.util.spec_from_file_location("mm_stability_cpu", p)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

class PolicyTest(unittest.TestCase):
    def test_only_exact_dtype_layout_and_enabled_shapes(self):
        for k, n in policy.FAMILIES:
            for m in (1, 2, 3, 4, 5, 8, 9, 32, 64, 65):
                for enabled in (False, True):
                    self.assertEqual(policy.stable_shape(m, n, k, "torch.bfloat16", "torch.bfloat16",
                                                        (k, 1, 1, k), enabled),
                                     enabled and m in (3, 4, 8))
        for dtype in ("torch.float16", "torch.float32", "torch.float64"):
            self.assertFalse(policy.stable_shape(4, 2048, 2048, dtype, dtype, (2048, 1, 1, 2048), True))
        self.assertFalse(policy.stable_shape(4, 2048, 2048, "torch.bfloat16", "torch.bfloat16", (2048, 1, 2048, 1), True))
        self.assertFalse(policy.stable_shape(4, 4096, 2048, "torch.bfloat16", "torch.bfloat16", (2048, 1, 1, 2048), True))

    def test_reject_pipelined_or_split_reductions(self):
        configs = [NS(num_stages=s, kwargs={"SPLIT_K": k}) for s in (1, 2, 3) for k in (1, 2, 4)]
        self.assertEqual(policy.stable_configs(configs), [configs[0]])

if __name__ == "__main__":
    unittest.main(verbosity=2)
