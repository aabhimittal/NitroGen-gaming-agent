"""NitroGen — an educational, end-to-end vision-action foundation model for gaming agents.

This package is a compact, faithful, *runnable* re-implementation of the core ideas
behind NitroGen (NVIDIA, 2025): a single RGB game frame is mapped to a chunk of future
standardized gamepad actions using a flow-matching (diffusion-style) action head trained
with pure behavior cloning.

The public surface mirrors the conceptual pipeline described in ``docs/``:

    frame ──► VisionEncoder ──► image tokens
                                     │
    noise + t ──► ActionHead (DiT) ──┴──► velocity field ──► ODE integrate ──► action chunk

Typical usage::

    from nitrogen import NitroGen, NitroGenConfig
    model = NitroGen(NitroGenConfig())
    loss = model.compute_loss(frames, action_chunks)      # training (behavior cloning)
    actions = model.sample_actions(frames)                # inference (action chunking)
"""

from nitrogen.action_space import (
    GamepadAction,
    ACTION_DIM,
    BUTTON_NAMES,
    encode_action,
    decode_action,
)
from nitrogen.models.nitrogen import NitroGen, NitroGenConfig

__all__ = [
    "GamepadAction",
    "ACTION_DIM",
    "BUTTON_NAMES",
    "encode_action",
    "decode_action",
    "NitroGen",
    "NitroGenConfig",
]

__version__ = "0.1.0"
