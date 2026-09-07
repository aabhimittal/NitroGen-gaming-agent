"""Single frame -> action *chunk* dataset.

This is where NitroGen's core supervision shape appears::

    input :  frame_i                                       (ONE RGB image)
    target:  [action_i, action_{i+1}, ..., action_{i+T-1}]  (a T-step CHUNK)

No frame stack and no history — the policy must infer everything it needs from
one image, and it commits to a whole chunk of future actions at once.

``pad_last`` controls what happens near the end of an episode, where fewer than
``T`` real actions remain:

    * ``True``  — pad with the *null* action. Keeps every frame trainable.
    * ``False`` — drop those frames entirely (the trainer's default). Otherwise
      a short-episode corpus is dominated by padding and the policy learns to
      idle.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from nitrogen.action_space import ACTION_DIM, encode_action, null_action
from nitrogen.data.synthetic import Episode


class FrameActionChunkDataset(Dataset):
    """Flattens a list of :class:`Episode` into ``(frame, action_chunk)`` pairs."""

    def __init__(
        self,
        episodes: Sequence[Episode],
        chunk_size: int = 16,
        pad_last: bool = True,
    ):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.episodes = list(episodes)
        self.chunk_size = int(chunk_size)
        self.pad_last = bool(pad_last)
        self._null = null_action().astype(np.float32)

        # Precompute the (episode, start-frame) index map once, so __getitem__
        # stays O(1) and the length is exact.
        self.index: List[Tuple[int, int]] = []
        for e_idx, ep in enumerate(self.episodes):
            n = len(ep)
            last = n if pad_last else n - self.chunk_size + 1
            for f_idx in range(max(0, last)):
                self.index.append((e_idx, f_idx))

    def __len__(self) -> int:
        return len(self.index)

    def action_chunk(self, e_idx: int, f_idx: int) -> np.ndarray:
        """The normalized ``(chunk_size, ACTION_DIM)`` target, padded if needed."""
        ep = self.episodes[e_idx]
        raw = ep.actions[f_idx:f_idx + self.chunk_size]
        chunk = encode_action(np.asarray(raw, dtype=np.float32))
        missing = self.chunk_size - chunk.shape[0]
        if missing > 0:
            pad = np.repeat(self._null[None, :], missing, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk.astype(np.float32)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        e_idx, f_idx = self.index[i]
        frame = self.episodes[e_idx].frames[f_idx]
        # (H, W, 3) uint8 -> (3, H, W) float in [0, 1], the encoder's contract.
        pixels = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
        pixels = pixels.permute(2, 0, 1)
        chunk = torch.from_numpy(self.action_chunk(e_idx, f_idx))
        return pixels, chunk

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FrameActionChunkDataset(samples={len(self)}, episodes={len(self.episodes)}, "
            f"chunk_size={self.chunk_size}, action_dim={ACTION_DIM}, pad_last={self.pad_last})"
        )
