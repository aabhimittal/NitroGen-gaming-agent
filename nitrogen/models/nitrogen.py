"""The full NitroGen policy: vision encoder + flow-matching action head.

This module wires the three pieces from ``docs/`` together into a single policy
with two public methods:

    * :meth:`compute_loss`  — the behavior-cloning training objective. Given a
      batch of (frame, expert-action-chunk) pairs it returns the flow-matching
      loss. **No reinforcement learning, no reward** — pure imitation.

    * :meth:`sample_actions` — inference. Given a frame it generates a chunk of
      ``chunk_size`` future actions by integrating the learned ODE from noise.

Everything downstream (the trainer, the benchmark, the demo) only touches these
two methods, so the generative internals stay encapsulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from nitrogen.action_space import ACTION_DIM, decode_action
from nitrogen.models.action_head import ActionHead, ActionHeadConfig
from nitrogen.models.flow_matching import FlowMatching
from nitrogen.models.vision_encoder import VisionConfig, VisionEncoder


@dataclass
class NitroGenConfig:
    """Configuration for the whole policy.

    The defaults describe a *small* model (~a few million params) that trains on
    CPU in minutes — enough to actually learn the toy games in this repo. Scaling
    to the paper's 500M is a matter of widening/deepening the two sub-configs.
    """

    action_dim: int = ACTION_DIM
    chunk_size: int = 16
    # NOTE on resolution: the *paper* encodes 256x256 frames (256 patch tokens).
    # The default here is 128x128 (64 tokens) purely so the whole pipeline trains
    # in minutes on a CPU — the encoder transparently resizes 256px env frames
    # down to this size. Set ``VisionConfig(image_size=256)`` to match the paper.
    vision: VisionConfig = field(
        default_factory=lambda: VisionConfig(image_size=128, width=256, depth=4, num_heads=4)
    )
    head: ActionHeadConfig | None = None      # filled in __post_init__ to match vision
    sample_steps: int = 10                    # ODE integration steps at inference

    def __post_init__(self):
        if self.head is None:
            self.head = ActionHeadConfig(
                action_dim=self.action_dim,
                chunk_size=self.chunk_size,
                width=self.vision.width,
                context_width=self.vision.width,
                depth=4,
                num_heads=self.vision.num_heads,
                time_embed_dim=self.vision.width,
            )


class NitroGen(nn.Module):
    def __init__(self, cfg: NitroGenConfig | None = None):
        super().__init__()
        self.cfg = cfg or NitroGenConfig()
        self.encoder = VisionEncoder(self.cfg.vision)
        self.action_head = ActionHead(self.cfg.head)
        self.flow = FlowMatching()

    # -- helpers ---------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, 3, H, W) in [0, 1] -> image tokens (B, N, width)."""
        return self.encoder(frames)

    # -- training --------------------------------------------------------
    def compute_loss(self, frames: torch.Tensor, action_chunks: torch.Tensor) -> torch.Tensor:
        """Behavior-cloning loss via flow matching.

        Args:
            frames: (B, 3, H, W) float in [0, 1] — the single conditioning frame.
            action_chunks: (B, T, action_dim) — expert actions in the *normalized*
                ``[-1, 1]`` space (see :func:`nitrogen.action_space.encode_action`).
        """
        image_tokens = self.encode(frames)

        def velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.action_head(x_t, t, image_tokens)

        return self.flow.loss(velocity_fn, action_chunks)

    # -- inference -------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self, frames: torch.Tensor, num_steps: int | None = None
    ) -> torch.Tensor:
        """Generate an action chunk per frame. Returns (B, T, action_dim) in [-1, 1]."""
        self.eval()
        image_tokens = self.encode(frames)
        B = frames.shape[0]
        steps = num_steps or self.cfg.sample_steps

        def velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.action_head(x_t, t, image_tokens)

        chunk = self.flow.sample(
            velocity_fn,
            shape=(B, self.cfg.chunk_size, self.cfg.action_dim),
            device=frames.device,
            num_steps=steps,
        )
        return chunk.clamp(-1.0, 1.0)

    @torch.no_grad()
    def act(self, frame: np.ndarray, num_steps: int | None = None) -> np.ndarray:
        """Convenience: single RGB frame -> a chunk of *raw* gamepad action vectors.

        Args:
            frame: (H, W, 3) uint8 or float in [0, 1].
        Returns:
            (T, action_dim) raw action vectors (sticks in [-1,1], triggers in
            [0,1], buttons in {0,1}) — ready to feed to an environment.
        """
        arr = np.asarray(frame)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        norm_chunk = self.sample_actions(tensor, num_steps=num_steps)[0].cpu().numpy()
        return decode_action(norm_chunk)
