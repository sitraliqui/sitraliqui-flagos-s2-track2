# SPDX-License-Identifier: Apache-2.0
"""Explicit runner scopes. Importing this module never initializes CUDA."""

from contextvars import ContextVar
from functools import wraps
from inspect import signature
import os


_active = ContextVar("fl_kv_decode_controller", default=None)


def requested():
    return os.environ.get("VLLM_FL_KV_DECODE_FUSION", "0") == "1"


def target_device():
    import torch
    from vllm import envs

    return (
        torch.cuda.is_available()
        and not envs.VLLM_BATCH_INVARIANT
        and torch.cuda.get_device_name() == "Iluvatar BI-V150"
    )


def active_controller():
    return _active.get()


def controller_for(runner):
    if not requested():
        return None
    if not hasattr(runner, "_fl_kv_decode_controller"):
        if not target_device():
            runner._fl_kv_decode_controller = None
        else:
            if os.environ.get("VLLM_FL_KV_DECODE_PROOF_DIR"):
                from .proof import ProofController

                runner._fl_kv_decode_controller = ProofController()
            else:
                from .controller import KVDecodeController

                runner._fl_kv_decode_controller = KVDecodeController()
    return runner._fl_kv_decode_controller


def runner_scope(kind):
    """Source-level runner decorator; never replaces an installed class method."""

    def decorate(fn):
        sig = signature(fn)

        @wraps(fn)
        def scoped(runner, *args, **kwargs):
            controller = controller_for(runner)
            if controller is None:
                return fn(runner, *args, **kwargs)
            previous = (
                controller.step,
                controller.capture_owner,
                controller.capture_runner,
            )
            token = _active.set(controller)
            try:
                if kind == "execute":
                    schedule = args[0] if args else kwargs["scheduler_output"]
                    controller.step = (
                        runner,
                        tuple(map(int, schedule.num_scheduled_tokens.values())),
                    )
                elif kind == "capture":
                    controller.capture_owner = runner
                elif kind == "dummy":
                    bound = sig.bind(runner, *args, **kwargs)
                    if (
                        controller.capture_owner is runner
                        and bound.arguments.get("is_graph_capturing") is True
                    ):
                        controller.capture_runner = runner
                else:
                    raise ValueError("Unknown KV-decode runner scope")
                return fn(runner, *args, **kwargs)
            finally:
                (
                    controller.step,
                    controller.capture_owner,
                    controller.capture_runner,
                ) = previous
                _active.reset(token)

        return scoped

    return decorate
