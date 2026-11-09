# SPDX-License-Identifier: Apache-2.0
"""Optional extension of the normally selected Triton attention backend."""

from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from .runtime import active_controller


class TritonKVDecodeBackend(TritonAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return TritonKVDecodeImpl


class TritonKVDecodeImpl(TritonAttentionImpl):
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        controller = active_controller()
        state = controller.layer_state(layer) if controller else None
        if state is None:
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        controller.capture_write(state, key, value, kv_cache, slot_mapping)

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        controller = active_controller()
        state = controller.layer_state(layer) if controller else None
        if state is None:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        return controller.capture_forward(
            state,
            self,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
