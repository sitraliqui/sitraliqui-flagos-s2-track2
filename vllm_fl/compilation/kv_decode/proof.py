# SPDX-License-Identifier: Apache-2.0
"""Explicit untimed all-layer observer; never enabled in performance runs."""

import json
import os
from pathlib import Path

from .bindings import LAYERS, META, require
from .contract import GRAPH_HEADROOM, SHADOW_CAP
from .controller import KVDecodeController


class ProofController(KVDecodeController):
    def __init__(self):
        super().__init__()
        self.output = Path(os.environ["VLLM_FL_KV_DECODE_PROOF_DIR"])
        require(self.output.is_absolute(), "Proof output must be absolute")
        self.output.mkdir(parents=True, exist_ok=True)
        self.shadow = self.flags = None
        self.proof_steps = []
        self.captures = []

    def save(self, name, value):
        target = self.output / name
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(target)

    def capture_states(self):
        t = self.torch
        first = self.context().no_compile_layers[LAYERS[0]].kv_cache
        if self.shadow is None:
            needed = first.numel() * 3
            free, _ = t.cuda.mem_get_info()
            require(
                needed <= SHADOW_CAP and free >= needed + GRAPH_HEADROOM,
                "Full physical KV observer memory budget",
            )
            self.shadow = t.empty_like(first)
            self.flags = t.empty_like(first, dtype=t.bool)
        require(first.shape == self.shadow.shape, "Observer cache binding changed")
        bundle = super().capture_states()
        if bundle is not None:
            for state in bundle["layers"].values():
                state["checks"] = t.empty(4, device=first.device, dtype=t.bool)
                state["reference"] = t.empty(
                    (state["capacity"], 16, 128), device=first.device, dtype=first.dtype
                )
        return bundle

    def capture_forward(
        self,
        state,
        impl,
        query,
        key,
        value,
        cache,
        metadata,
        output,
        output_scale,
        output_block_scale,
    ):
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

        t = self.torch
        layer = next(
            self.context().no_compile_layers[name]
            for name, item in self.variant_capture["layers"].items()
            if item is state
        )
        reference = state["reference"]
        reference.fill_(float("nan"))
        self.shadow.copy_(cache)
        TritonAttentionImpl.do_kv_cache_update(
            impl,
            layer,
            key,
            value,
            self.shadow,
            self.context().slot_mapping[layer.layer_name],
        )
        TritonAttentionImpl.forward(
            impl, layer, query, key, value, self.shadow, metadata, reference, None, None
        )
        active = state["staged"][1] > 0
        output.copy_(t.where(active[:, None, None], float("nan"), 123.0))
        for scratch in state["scratch"]:
            scratch.fill_(float("nan"))
        result = super().capture_forward(
            state,
            impl,
            query,
            key,
            value,
            cache,
            metadata,
            output,
            output_scale,
            output_block_scale,
        )
        checks = state["checks"]
        t.ne(cache.view(t.int16), self.shadow.view(t.int16), out=self.flags)
        t.any(self.flags, out=checks[0])
        got, want = output.float(), reference.float()
        bad = (
            (~t.isfinite(got))
            | (~t.isfinite(want))
            | ((got - want).abs() > 0.002 + 0.02 * want.abs())
        )
        t.any(bad & active[:, None, None], out=checks[1])
        t.any((output != 123) & ~active[:, None, None], out=checks[2])
        checks[3].zero_()
        for scratch in state["scratch"]:
            mask = (~active).view(-1, *([1] * (scratch.ndim - 1)))
            checks[3].logical_or_(t.any((~t.isnan(scratch)) & mask))
        return result

    def verify_mirrors(self, live, host):
        return all(live[key].detach().cpu().tolist() == host[key] for key in META)

    def observe_capture(self, bundle):
        super().observe_capture(bundle)
        self.captures.append(
            dict(
                profile=self.policy.name,
                capacity=bundle["capacity"],
                segments=self.segments,
                layers=len(bundle["layers"]),
            )
        )
        self.save("captures.json", self.captures)

    def observe_replay(self, bundle, host, step, descriptor):
        checks = (
            self.torch.stack([state["checks"] for state in bundle["layers"].values()])
            .cpu()
            .tolist()
        )
        bad = [name for name, flags in zip(LAYERS, checks) if any(flags)]
        row = dict(
            profile=self.policy.name,
            capacity=bundle["capacity"],
            active=len(step[1]),
            lengths=host["lengths"][: len(step[1])],
            segments=self.segments,
            layers=list(LAYERS),
            checks=checks,
            passed=not bad,
        )
        require(len(self.proof_steps) < 4096, "Proof record budget exhausted")
        self.proof_steps.append(row)
        self.save(
            "layer-proof.json",
            dict(
                scope="untimed installed model all-layer Graph diagnostic",
                steps=self.proof_steps,
                rtol=0.02,
                atol=0.002,
                full_physical_KV=True,
                active_finite_coverage=True,
            ),
        )
        require(not bad, "Native/full-KV/padding comparison failed: " + ",".join(bad))
