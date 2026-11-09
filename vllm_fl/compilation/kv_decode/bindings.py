# SPDX-License-Identifier: Apache-2.0

META = ("query_start", "lengths", "table", "slots")
LAYERS = tuple(f"model.layers.{i}.self_attn.attn" for i in range(42))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def view_key(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.device),
        str(tensor.dtype),
    )


def views(metadata, slot):
    return dict(
        zip(
            META,
            (metadata.query_start_loc, metadata.seq_lens, metadata.block_table, slot),
        )
    )


def strong_view_key(tensor):
    return (
        *view_key(tensor),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
    )


def shared_binding(runner, ctx, cap):
    """Read-only identity checks; no GPU content transfer or GPU operations."""
    groups = runner.kv_cache_config.kv_cache_groups
    group_ids = []
    for name in LAYERS:
        matching = [i for i, g in enumerate(groups) if name in g.layer_names]
        require(len(matching) == 1, "layer has ambiguous KV group")
        group_ids.append(matching[0])
    require(len(set(group_ids)) == 1, "target layers span different KV groups")
    gid = group_ids[0]
    table = runner.input_batch.block_table[gid]
    canonical = (
        runner.query_start_loc.gpu[: cap + 1],
        runner.seq_lens[:cap],
        table.get_device_tensor(cap),
        table.slot_mapping.gpu[:cap],
    )
    signature = tuple(strong_view_key(x) for x in canonical)
    capacities = set()
    for name in LAYERS:
        layer = ctx.no_compile_layers[name]
        capacities.add(layer.kv_cache.shape[0])
        live = views(ctx.attn_metadata[name], ctx.slot_mapping[name])
        require(
            tuple(strong_view_key(live[k]) for k in META) == signature,
            "layer metadata is not exact shared live view",
        )
    require(len(capacities) == 1, "layer cache capacities differ")
    objects = (
        runner,
        runner.input_batch,
        table,
        table.get_cpu_tensor(),
        table.num_blocks_per_row,
        runner.query_start_loc.np,
        runner.optimistic_seq_lens_cpu,
        runner.num_scheduled_tokens.np,
        runner.input_batch.num_computed_tokens_cpu,
        runner.input_batch.num_prompt_tokens,
    )
    return dict(group=gid, signature=signature, objects=objects)


def validate_independence(states, require_outputs):
    require(list(states) == list(LAYERS), "incomplete independent layer states")
    owned = []
    for state in states.values():
        items = [state["cache"], *state["scratch"]]
        if require_outputs:
            items.append(state["output"])
        for tensor in items:
            pointer = tensor.untyped_storage().data_ptr()
            require(pointer not in owned, "layer cache/scratch/output storage aliases")
            owned.append(pointer)


def plain(impl, metadata):
    require(
        impl.kv_cache_dtype in ("auto", "bfloat16")
        and impl.sliding_window == (-1, -1)
        and impl.alibi_slopes is None
        and impl.sinks is None
        and not impl.logits_soft_cap
        and not impl.use_td
        and impl.chunk_lookback in (None, -1)
        and not metadata.use_cascade
        and metadata.mm_prefix_range_tensor is None
        and metadata.causal is True
        and str(getattr(impl.attn_type, "value", impl.attn_type)).lower() == "decoder",
        "unsupported attention semantics",
    )
