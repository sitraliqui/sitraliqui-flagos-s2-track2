# SPDX-License-Identifier: Apache-2.0
"""Normal OOT layer registration with an opaque, public FlagGems mm boundary.

Weight loading, TP communication, bias, quantization and batch-invariant fallback
remain in the upstream layers/methods. No installed class method is reassigned.
"""

import torch
from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)


@torch.library.custom_op(
    "vllm_fl::iluvatar_unquantized_mm", mutates_args=(), device_types="cuda"
)
def opaque_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    import flag_gems

    # Public selected backend, including its normal libentry/libtuner wrappers.
    return flag_gems.mm(x, weight.T)


@opaque_mm.register_fake
def _fake_mm(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0]))


def eligible(layer, x, bias):
    from vllm import envs

    w = layer.weight
    return (
        not envs.VLLM_BATCH_INVARIANT
        and bias is None
        and x.ndim == w.ndim == 2
        and x.is_cuda
        and x.dtype == w.dtype == torch.bfloat16
        and x.device == w.device
        and x.shape[1] == w.shape[1]
        and x.numel() > 0
        and w.numel() > 0
    )


class LinearMethod(UnquantizedLinearMethod):
    def apply(self, layer, x, bias=None):
        if eligible(layer, x, bias):
            return opaque_mm(x, layer.weight)
        return super().apply(layer, x, bias)


class EmbeddingMethod(UnquantizedEmbeddingMethod):
    def apply(self, layer, x, bias=None):
        if eligible(layer, x, bias):
            return opaque_mm(x, layer.weight)
        return super().apply(layer, x, bias)


class BoundaryLayer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vllm_fl.compilation.kv_decode.runtime import target_device
        from vllm_fl.utils import use_flaggems_op

        if not (target_device() and use_flaggems_op("mm")):
            return
        # Keep quantized/subclass methods intact. The original constructor and
        # weight creation have run unchanged before selecting our derived method.
        if type(self.quant_method) is UnquantizedLinearMethod:
            self.quant_method = LinearMethod()
        elif type(self.quant_method) is UnquantizedEmbeddingMethod:
            self.quant_method = EmbeddingMethod()


class ReplicatedLinearFL(BoundaryLayer, ReplicatedLinear):
    pass


class ColumnParallelLinearFL(BoundaryLayer, ColumnParallelLinear):
    pass


class MergedColumnParallelLinearFL(BoundaryLayer, MergedColumnParallelLinear):
    pass


class QKVParallelLinearFL(BoundaryLayer, QKVParallelLinear):
    pass


class RowParallelLinearFL(BoundaryLayer, RowParallelLinear):
    pass


class VocabParallelEmbeddingFL(BoundaryLayer, VocabParallelEmbedding):
    pass


class ParallelLMHeadFL(BoundaryLayer, ParallelLMHead):
    pass


def register_layers():
    pairs = (
        (ReplicatedLinear, ReplicatedLinearFL),
        (ColumnParallelLinear, ColumnParallelLinearFL),
        (MergedColumnParallelLinear, MergedColumnParallelLinearFL),
        (QKVParallelLinear, QKVParallelLinearFL),
        (RowParallelLinear, RowParallelLinearFL),
        (VocabParallelEmbedding, VocabParallelEmbeddingFL),
        (ParallelLMHead, ParallelLMHeadFL),
    )
    # Preflight the entire registration to avoid partially replacing another
    # plugin's classes. Identical repeated general-plugin calls are harmless.
    for original, replacement in pairs:
        prior = op_registry_oot.get(original.__name__)
        if prior is not None and prior is not replacement:
            raise RuntimeError(f"Existing OOT owner for {original.__name__}")
    for original, replacement in pairs:
        if original.__name__ not in op_registry_oot:
            PluggableLayer.register_oot(name=original.__name__)(replacement)
