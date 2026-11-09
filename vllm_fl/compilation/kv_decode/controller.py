# SPDX-License-Identifier: Apache-2.0
"""Bounded source-owned fusion. Unsupported replay routes use native graphs."""

from contextlib import contextmanager
from .bindings import (
    LAYERS,
    META,
    plain,
    require,
    shared_binding,
    validate_independence,
    view_key,
    views,
)
from .contract import (
    BATCHES,
    MAX_CONTEXT,
    MIB,
    SEGMENTS,
    TABLE_COLUMNS,
    has_headroom,
    metadata_reason,
    route_reason,
)


@contextmanager
def graph_entries(wrapper, entries, pool, restore_pool=None):
    original_entries, original_pool = wrapper.concrete_graph_entries, wrapper.graph_pool
    wrapper.concrete_graph_entries, wrapper.graph_pool = entries, pool
    try:
        yield
    finally:
        wrapper.concrete_graph_entries, wrapper.graph_pool = (
            original_entries,
            original_pool,
        )
        if restore_pool is not None:
            restore_pool(wrapper)


class KVDecodeController:
    capacities = tuple(BATCHES)
    plain = staticmethod(plain)

    def __init__(self):
        import torch
        from flag_gems.runtime.backend._iluvatar.fused import kv_decode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        self.torch, self.fused = torch, kv_decode
        self.context, self.context_available = (
            get_forward_context,
            is_forward_context_available,
        )
        self.segments = SEGMENTS
        self.step = self.capture_owner = self.capture_runner = None
        self.variant_capture = None
        self.first_hits = set()

    def event(self, reason, step=None, descriptor=None):
        if reason.startswith("fused_active_") and reason not in self.first_hits:
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "FL KV-decode fusion entered %s, S%d", reason, self.segments
            )
            self.first_hits.add(reason)

    def observe_replay(self, bundle, host, step, descriptor):
        pass

    def verify_mirrors(self, live, host):
        # CPU/device source binding is checked on every replay. Full GPU mirror
        # comparison is a separate untimed acceptance test, not hidden timing work.
        return True

    def layer_state(self, layer):
        bundle = self.variant_capture
        return bundle["layers"].get(layer.layer_name) if bundle else None

    def supports_capture(self, runner, ctx, cap):
        # Check model/backend compatibility before allocating or suppressing any
        # native KV write. Other models and explicit attention backends stay native.
        from .backend import TritonKVDecodeImpl

        try:
            p = runner.vllm_config.parallel_config
            if (
                p.decode_context_parallel_size != 1
                or p.prefill_context_parallel_size != 1
                or runner.vllm_config.kv_transfer_config is not None
                or runner.speculative_config is not None
                or runner.use_async_spec_decode
            ):
                return False
            for name in LAYERS:
                layer, meta = ctx.no_compile_layers[name], ctx.attn_metadata[name]
                if not isinstance(layer.impl, TritonKVDecodeImpl):
                    return False
                self.plain(layer.impl, meta)
                if (
                    meta.max_query_len != 1
                    or meta.num_par_softmax_segments != self.segments
                    or meta.seq_threshold_3D < cap
                    or layer.kv_sharing_target_layer_name is not None
                    or layer.kv_cache.dtype != self.torch.bfloat16
                    or tuple(layer.kv_cache.shape[1:]) != (2, 16, 2, 128)
                ):
                    return False
            return True
        except (
            AttributeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            RuntimeError,
        ):
            return False

    def capture_write(self, state, key, value, cache, slot):
        require(
            state["write"] is None
            and view_key(cache) == view_key(state["cache"])
            and view_key(slot) == state["live_signature"][-1],
            "capture writer binding",
        )
        state["write"] = view_key(key), view_key(value)

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
        self.plain(impl, metadata)
        cap = state["capacity"]
        tensors = (
            query,
            key,
            value,
            output,
            cache,
            *state["staged"],
            *state["scratch"],
        )
        pointers = [tensor.untyped_storage().data_ptr() for tensor in tensors]
        require(
            len(set(pointers[3:])) == len(pointers[3:])
            and not set(pointers[:3]).intersection(pointers[3:]),
            "capture output/cache alias",
        )
        require(
            state["write"] == (view_key(key), view_key(value))
            and not state["used"]
            and view_key(cache) == view_key(state["cache"])
            and output_scale is None
            and output_block_scale is None
            and tuple(query.shape) == tuple(output.shape) == (cap, 16, 128)
            and tuple(key.shape) == tuple(value.shape) == (cap, 2, 128)
            and tuple(value.stride()) == (2560, 128, 1)
            and all(
                x.dtype == self.torch.bfloat16 and x.device == cache.device
                for x in (query, key, value, output)
            )
            and all(x.is_contiguous() for x in (query, key, output)),
            "capture tensor contract",
        )
        self.launch(state, query, key, value, cache, output, impl.scale)
        state["output"], state["used"] = output, True
        return output

    def admission(self, runner, schedule, ctx, bundle, args, kwargs):
        active = runner.input_batch.num_reqs
        cap = bundle["capacity"]
        inputs = runner.input_batch
        lengths = runner.optimistic_seq_lens_cpu[:cap].tolist()
        reason = route_reason(
            cap,
            schedule,
            lengths[:active],
            inputs.num_computed_tokens_cpu[:active],
            inputs.num_prompt_tokens[:active],
            active,
        )
        if reason is not None:
            return reason
        parallel = runner.vllm_config.parallel_config
        if (
            parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or runner.vllm_config.kv_transfer_config is not None
            or (runner.speculative_config is not None)
            or runner.use_async_spec_decode
            or (self.inputs(args, kwargs) != bundle["inputs"])
        ):
            return "unsupported_runner_or_inputs"
        if runner.num_scheduled_tokens.np[:active].tolist() != [1] * active:
            return "runner_schedule_changed"
        bound = shared_binding(runner, ctx, cap)
        previous = bundle["binding"]
        if (
            bound["group"] != previous["group"]
            or bound["signature"] != previous["signature"]
            or len(bound["objects"]) != len(previous["objects"])
            or any((a is not b for a, b in zip(bound["objects"], previous["objects"])))
        ):
            return "metadata_rebound"
        for name, state in bundle["layers"].items():
            layer = ctx.no_compile_layers[name]
            meta = ctx.attn_metadata[name]
            self.plain(layer.impl, meta)
            require(
                meta.max_query_len == 1
                and meta.num_par_softmax_segments == self.segments
                and (meta.seq_threshold_3D >= cap),
                "Matched native S16 decode path changed",
            )
            live = views(meta, ctx.slot_mapping[name])
            if (
                view_key(layer.kv_cache) != view_key(state["cache"])
                or tuple((view_key(live[key]) for key in META))
                != state["live_signature"]
            ):
                return "layer_rebound"
        table = inputs.block_table[bound["group"]]
        if (
            table.block_size != 16
            or table.use_hybrid_blocks
            or table.pcp_world_size != 1
            or (table.dcp_world_size != 1)
        ):
            return "unsupported_block_table"
        raw = table.get_cpu_tensor()
        if str(raw.device) != "cpu" or raw.shape[1] < TABLE_COLUMNS:
            return "unbound_cpu_table"
        rows = raw[:active, :TABLE_COLUMNS].tolist() + [
            [0] * TABLE_COLUMNS for _ in range(cap - active)
        ]
        pages = [(length + 15) // 16 for length in lengths[:active]]
        if any(
            (int(n) < used for n, used in zip(table.num_blocks_per_row[:active], pages))
        ):
            return "unallocated_page"
        slots = [
            row[p - 1] * 16 + (n - 1) % 16
            for n, row, p in zip(lengths[:active], rows, pages)
        ]
        host = dict(
            query_start=runner.query_start_loc.np[: cap + 1].tolist(),
            lengths=lengths,
            table=rows,
            slots=slots + [-1] * (cap - active),
        )
        reason = metadata_reason(
            host, bundle["layers"][LAYERS[0]]["cache"].shape[0], active
        )
        if reason:
            return reason
        return ({**live, "table": live["table"][:, :TABLE_COLUMNS]}, host)

    def launch(self, state, q, k, v, cache, out, scale):
        kc, vc = cache.unbind(1)
        cu, lengths, table, slots = state["staged"]
        self.fused._fused_attention_internal(
            q,
            kc,
            vc,
            out,
            cu,
            1,
            lengths,
            MAX_CONTEXT,
            float(scale),
            True,
            (-1, -1),
            table,
            0.0,
            1.0,
            1.0,
            1.0,
            seq_threshold_3D=state["capacity"],
            num_par_softmax_segments=self.segments,
            softmax_segm_output=state["scratch"][0],
            softmax_segm_max=state["scratch"][1],
            softmax_segm_expsum=state["scratch"][2],
            _new_k=k,
            _new_v=v,
            _slot_mapping=slots,
            _latest_tile_only=True,
            _fixed_query_capacity=True,
        )

    def capture_states(self):
        t, ctx = (self.torch, self.context())
        cap = ctx.batch_descriptor.num_tokens
        if cap not in BATCHES:
            return None
        runner = self.capture_runner
        binding = shared_binding(runner, ctx, cap)
        first = ctx.no_compile_layers[LAYERS[0]].kv_cache
        # Re-check each capture, including recapture after clear_graphs().
        # An unsupported memory budget falls back before touching real KV.
        free, _ = t.cuda.mem_get_info()
        scratch_bytes = cap * len(LAYERS) * 16 * self.segments * (128 + 2) * 4
        if not has_headroom(free, scratch_bytes):
            return None
        states, staged = ({}, None)
        for name in LAYERS:
            layer = ctx.no_compile_layers[name]
            cache = layer.kv_cache
            meta = ctx.attn_metadata[name]
            live = views(meta, ctx.slot_mapping[name])
            self.plain(layer.impl, meta)
            require(
                meta.num_par_softmax_segments == self.segments
                and meta.seq_threshold_3D >= cap,
                "Capture no longer matches native S16 3D attention",
            )
            require(
                cache.ndim == 5
                and tuple(cache.shape[1:]) == (2, 16, 2, 128)
                and (cache.shape == first.shape)
                and cache.is_contiguous()
                and (cache.shape[0] >= cap * TABLE_COLUMNS)
                and (cache.device == first.device)
                and (cache.device.type != "cpu")
                and (cache.dtype == t.bfloat16)
                and (layer.kv_sharing_target_layer_name is None)
                and (meta.max_query_len == 1)
                and (not self.fused.is_batch_invariant),
                "Long Graph cache or attention contract",
            )
            require(
                tuple(live["query_start"].shape) == (cap + 1,)
                and tuple(live["lengths"].shape) == (cap,)
                and (tuple(live["slots"].shape) == (cap,))
                and (live["table"].ndim == 2)
                and (live["table"].shape[0] == cap)
                and (live["table"].shape[1] >= TABLE_COLUMNS)
                and all(
                    (
                        x.is_contiguous()
                        and x.device == cache.device
                        and (x.dtype in (t.int32, t.int64))
                        for x in live.values()
                    )
                ),
                "Long Graph metadata contract",
            )
            if staged is None:
                dummy = (
                    list(range(cap + 1)),
                    [1] * cap,
                    [[i] + [0] * (TABLE_COLUMNS - 1) for i in range(cap)],
                    [i * 16 for i in range(cap)],
                )
                staged = tuple(
                    (
                        t.tensor(values, dtype=live[key].dtype, device=cache.device)
                        for key, values in zip(META, dummy)
                    )
                )
            scratch = tuple(
                (
                    t.empty(shape, device=cache.device, dtype=t.float32)
                    for shape in (
                        (cap, 16, self.segments, 128),
                        (cap, 16, self.segments),
                        (cap, 16, self.segments),
                    )
                )
            )
            state = dict(
                cache=cache,
                capacity=cap,
                staged=staged,
                scratch=scratch,
                write=None,
                used=False,
                live_signature=tuple((view_key(live[key]) for key in META)),
            )
            states[name] = state
        validate_independence(states, require_outputs=False)
        cache_pointers = {
            state["cache"].untyped_storage().data_ptr() for state in states.values()
        }
        require(
            all(
                (
                    name in states
                    or not hasattr(layer, "kv_cache")
                    or layer.kv_cache.untyped_storage().data_ptr() not in cache_pointers
                    for name, layer in ctx.no_compile_layers.items()
                )
            ),
            "Unknown layer aliases a target cache",
        )
        first_state = states[LAYERS[0]]
        private = t.empty_like(first[:cap])
        q = t.zeros((cap, 16, 128), device=first.device, dtype=first.dtype)
        k = t.zeros((cap, 2, 128), device=first.device, dtype=first.dtype)
        v = t.empty_strided(
            k.shape, (2560, 128, 1), device=first.device, dtype=first.dtype
        ).zero_()
        self.launch(
            first_state,
            q,
            k,
            v,
            private,
            t.empty_like(q),
            ctx.no_compile_layers[LAYERS[0]].impl.scale,
        )
        t.cuda.synchronize()
        staged[0].zero_()
        staged[1].zero_()
        staged[2].zero_()
        staged[3].fill_(-1)
        t.cuda.synchronize()
        return dict(capacity=cap, layers=states, staged=staged, binding=binding)

    def observe_capture(self, bundle):
        free, _ = self.torch.cuda.mem_get_info()
        require(free >= 256 * MIB, "Post-capture VRAM guard reached")

    def inputs(self, args, kwargs):
        def describe(value):
            if isinstance(value, self.torch.Tensor):
                return view_key(value)
            if value is None:
                return None
            raise ValueError("Unexpected model input")

        return tuple(map(describe, args)), tuple(
            (k, describe(v)) for k, v in sorted(kwargs.items())
        )

    def restore_capture_pool(self, wrapper):
        # The pinned wrapper changes a process-global collective allocator pool
        # during capture as well as its own graph_pool attribute.
        from vllm.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )
        from vllm.platforms import current_platform

        set_graph_pool_id(
            wrapper.graph_pool
            if wrapper.graph_pool is not None
            else current_platform.graph_pool_handle()
        )

    def call(self, wrapper, *args, **kwargs):
        if not self.context_available():
            return wrapper._call_native(*args, **kwargs)
        ctx = self.context()
        descriptor = ctx.batch_descriptor
        runner = self.capture_runner
        step = self.step
        model_step = step is not None and wrapper is step[0].model and bool(step[1])
        if (
            getattr(ctx.cudagraph_runtime_mode, "name", "") != "FULL"
            or wrapper.runtime_mode != ctx.cudagraph_runtime_mode
        ) or getattr(descriptor, "num_tokens", None) not in self.capacities:
            if model_step:
                self.event("native_graph_mode_or_bucket", step, descriptor)
            return wrapper._call_native(*args, **kwargs)
        native = wrapper.concrete_graph_entries
        variants = wrapper.kv_decode_variants
        if (
            runner is not None
            and wrapper is runner.model
            and (descriptor not in variants)
        ):
            result = wrapper._call_native(*args, **kwargs)
            entry = native.get(descriptor)
            if entry is None or entry.graph is None:
                return result
            if not self.supports_capture(runner, ctx, descriptor.num_tokens):
                return result
            bundle = self.capture_states()
            if bundle is None:
                return result
            bundle["inputs"] = self.inputs(args, kwargs)
            entries, pool = ({}, self.torch.cuda.graph_pool_handle())
            self.variant_capture = bundle
            try:
                with graph_entries(wrapper, entries, pool, self.restore_capture_pool):
                    wrapper._call_native(*args, **kwargs)
                require(
                    all((state["used"] for state in bundle["layers"].values())),
                    "incomplete fusion capture",
                )
                validate_independence(bundle["layers"], require_outputs=True)
                require(
                    entries[descriptor].graph is not entry.graph,
                    "native and fused graph alias",
                )
                self.observe_capture(bundle)
                wrapper.kv_decode_variants[descriptor] = (entries, pool, bundle, entry)
                self.event(f"captured_{bundle['capacity']}")
            finally:
                self.variant_capture = None
            return result
        variant = variants.get(descriptor)
        if not model_step:
            return wrapper._call_native(*args, **kwargs)
        if variant is None:
            self.event("native_no_candidate_graph", step, descriptor)
            return wrapper._call_native(*args, **kwargs)
        entries, pool, bundle, native_entry = variant
        if native.get(descriptor) is not native_entry or native_entry.graph is None:
            self.event("native_entry_replaced", step, descriptor)
            return wrapper._call_native(*args, **kwargs)
        candidate_entry = entries.get(descriptor)
        if (
            candidate_entry is None
            or candidate_entry.graph is None
            or candidate_entry.graph is native_entry.graph
        ):
            self.event("candidate_entry_invalid", step, descriptor)
            return wrapper._call_native(*args, **kwargs)
        try:
            admission = self.admission(*step, ctx, bundle, args, kwargs)
        except (
            RuntimeError,
            AttributeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ):
            admission = "admission_rejected"
        if isinstance(admission, str):
            self.event(admission, step, descriptor)
            return wrapper._call_native(*args, **kwargs)
        live, host = admission
        if not self.verify_mirrors(live, host):
            self.event("mirror_mismatch", step, descriptor)
            return wrapper._call_native(*args, **kwargs)
        for destination, key in zip(bundle["staged"], META):
            destination.copy_(live[key])
        with graph_entries(wrapper, entries, pool):
            result = wrapper._call_native(*args, **kwargs)
        self.observe_replay(bundle, host, step, descriptor)
        self.event(f"fused_active_{len(step[1])}", step, descriptor)
        return result
