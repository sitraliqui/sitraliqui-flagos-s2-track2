# SPDX-License-Identifier: Apache-2.0
"""Narrow numerical-validation policy, independent of CUDA imports.

This constrains the normal autotuner; it is not a direct fixed-kernel launcher.
The numerical validity of all retained configurations needs target validation.
"""

FAMILIES = frozenset(
    ((2048, 2560), (2048, 2048), (2048, 12288), (6144, 2048), (2048, 130560))
)


def stable_shape(m, n, k, dtype_a, dtype_b, strides, enabled):
    return (
        enabled
        and m in (3, 4, 8)
        and (k, n) in FAMILIES
        and dtype_a == dtype_b == "torch.bfloat16"
        and tuple(strides) == (k, 1, 1, k)
    )


def stable_configs(configs):
    return [c for c in configs if c.num_stages == 1 and c.kwargs.get("SPLIT_K", 1) == 1]


def added_safe_config(config):
    return (
        config.kwargs == {"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}
        and config.num_warps == 8
        and config.num_stages == 1
    )
