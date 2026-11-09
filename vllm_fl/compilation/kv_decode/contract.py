# SPDX-License-Identifier: Apache-2.0
def check_metadata(query_start, lengths, table, slots, cache_blocks, active):
    """Check the read/write contract, using at most one interval per live page."""
    if not (
        type(active) is int
        and type(cache_blocks) is int
        and cache_blocks > 0
        and 0 < active <= len(lengths) == len(table)
        and len(query_start) == len(lengths) + 1
        and len(slots) >= active
        and all(type(x) is int for x in (*query_start, *lengths, *slots))
        and table
        and table[0]
        and all(len(row) == len(table[0]) for row in table)
    ):
        return "metadata_shape"
    if query_start != list(range(active + 1)) + [active] * (len(lengths) - active):
        return "not_packed_single_query"
    if any(n <= 0 or n > PAGE * len(table[0]) for n in lengths[:active]):
        return "invalid_length"
    if any(lengths[active:]) or any(slot != -1 for slot in slots[active:]):
        return "invalid_padding"

    # Each readable page starts at offset zero, even when partially used, so
    # any repeated physical page necessarily aliases a readable token.
    used_pages = set()
    for n, row, slot in zip(lengths[:active], table[:active], slots[:active]):
        page_count = (n + PAGE - 1) // PAGE
        for page in row[:page_count]:
            if type(page) is not int or not 0 <= page < cache_blocks:
                return "invalid_physical_page"
            if page in used_pages:
                return "shared_readable_page"
            used_pages.add(page)
        if slot != row[page_count - 1] * PAGE + (n - 1) % PAGE:
            return "write_not_at_latest_token"
    return None


"""Bounded 4K admission policy shared by live execution and CPU checks."""
PAGE = 16

MIN_CONTEXT = 4097
MAX_CONTEXT = 5119
TABLE_COLUMNS = (MAX_CONTEXT + PAGE - 1) // PAGE
BATCHES = {32: (32,), 64: (63, 64)}
SEGMENTS = 16
PROOF_LENGTHS = (4097, 4098, 4111, 4112, 4113, 5118, 5119)
LAYERS = 42
MIB = 1024**2
GRAPH_HEADROOM = 768 * MIB
SHADOW_CAP = 1024 * MIB


def planned_bytes(segments=SEGMENTS):
    return sum(BATCHES) * LAYERS * 16 * segments * (128 + 2) * 4


def route_reason(cap, scheduled, lengths, computed, prompts, active):
    if cap not in BATCHES or active not in BATCHES[cap] or len(scheduled) != active:
        return "unsupported_batch"
    if any(token != 1 for token in scheduled):
        return "mixed_or_prefill"
    if not len(lengths) == len(computed) == len(prompts) == active:
        return "unbound_rows"
    if any(
        done < prompt or length != done + 1
        for length, done, prompt in zip(lengths, computed, prompts)
    ):
        return "mixed_or_prefill"
    if any(length < MIN_CONTEXT or length > MAX_CONTEXT for length in lengths):
        return "context_bound"
    return None


def metadata_reason(host, cache_blocks, active):
    if len(host["slots"]) != len(host["lengths"]) or any(
        len(row) != TABLE_COLUMNS for row in host["table"]
    ):
        return "metadata_shape"
    if any(any(row) for row in host["table"][active:]):
        return "invalid_padding_table"
    return check_metadata(
        host["query_start"],
        host["lengths"],
        host["table"],
        host["slots"],
        cache_blocks,
        active,
    )


def has_headroom(free, scratch, shadow=0):
    return 0 <= shadow <= SHADOW_CAP and free >= scratch + shadow + GRAPH_HEADROOM


def proof_targets(cap, active, lengths):
    """A boundary is proved for every active row, not just min/max of a batch."""
    if (
        len(lengths) != active
        or route_reason(
            cap,
            [1] * active,
            lengths,
            [n - 1 for n in lengths],
            [4096] * active,
            active,
        )
        is not None
    ):
        return set()
    if len(set(lengths)) > 1:
        return {(cap, active, "ragged")}
    length = lengths[0]
    return {(cap, active, str(length))} if length in PROOF_LENGTHS else set()


def required_proofs():
    return {
        (cap, active, str(n))
        for cap, batches in BATCHES.items()
        for active in batches
        for n in (*PROOF_LENGTHS, "ragged")
    }
