# SPDX-License-Identifier: Apache-2.0
"""Disjoint candidate domains; the long profile needs installed validation."""

from dataclasses import dataclass
import os
from .contract import check_metadata


@dataclass(frozen=True)
class Profile:
    name: str
    minimum: int
    maximum: int
    segments: int
    batches: tuple

    @property
    def BATCHES(self):
        return dict(self.batches)

    @property
    def TABLE_COLUMNS(self):
        return (self.maximum + 15) // 16

    def route_reason(self, cap, scheduled, lengths, computed, prompts, active):
        if (
            cap not in self.BATCHES
            or active not in self.BATCHES[cap]
            or len(scheduled) != active
        ):
            return "unsupported_batch"
        if any(n != 1 for n in scheduled):
            return "mixed_or_prefill"
        if not len(lengths) == len(computed) == len(prompts) == active:
            return "unbound_rows"
        if any(
            done < prompt or length != done + 1
            for length, done, prompt in zip(lengths, computed, prompts)
        ):
            return "mixed_or_prefill"
        if any(n < self.minimum or n > self.maximum for n in lengths):
            return "context_bound"
        return None

    def metadata_reason(self, host, blocks, active):
        if len(host["slots"]) != len(host["lengths"]) or any(
            len(row) != self.TABLE_COLUMNS for row in host["table"]
        ):
            return "metadata_shape"
        if any(any(row) for row in host["table"][active:]):
            return "invalid_padding_table"
        return check_metadata(
            host["query_start"],
            host["lengths"],
            host["table"],
            host["slots"],
            blocks,
            active,
        )


FOUR_K = Profile("4k-s16", 4097, 5119, 16, ((32, (32,)), (64, (63, 64))))
# KV16384 development point and official-shaped decode neighbourhood.
# Padded B32 rows need per-row guards; this is not yet GPU-qualified.
SIXTEEN_K = Profile(
    "16k-s32", 16384, 17407, 32, ((1, (1,)), (32, tuple(range(17, 33))))
)


def configured_profiles():
    mode = os.environ.get("VLLM_FL_KV_DECODE_PROFILE", "4k")
    return {"4k": (FOUR_K,), "16k": (SIXTEEN_K,), "unified": (FOUR_K, SIXTEEN_K)}[mode]


def select_profile(profiles, capacity, active, lengths):
    if len(lengths) != active or active <= 0:
        return None
    matches = [
        p
        for p in profiles
        if active in p.BATCHES.get(capacity, ())
        and all(p.minimum <= n <= p.maximum for n in lengths)
    ]
    if len(matches) > 1:
        raise RuntimeError("Overlapping fusion profiles")
    return matches[0] if matches else None
