import numpy as np
import torch

from nitrogen import NitroGen, NitroGenConfig, ACTION_DIM
from nitrogen.models.vision_encoder import VisionConfig


def _tiny_config():
    vc = VisionConfig(image_size=64, patch_size=16, width=64, depth=2, num_heads=2)
    return NitroGenConfig(chunk_size=8, vision=vc)


def test_forward_shapes():
    model = NitroGen(_tiny_config())
    frames = torch.rand(3, 3, 64, 64)
    chunks = torch.rand(3, 8, ACTION_DIM) * 2 - 1
    loss = model.compute_loss(frames, chunks)
    assert torch.isfinite(loss)
    actions = model.sample_actions(frames, num_steps=4)
    assert actions.shape == (3, 8, ACTION_DIM)
    assert actions.min() >= -1.0 and actions.max() <= 1.0


def test_act_returns_raw_chunk():
    model = NitroGen(_tiny_config())
    frame = np.random.rand(64, 64, 3).astype(np.float32)
    raw = model.act(frame, num_steps=4)
    assert raw.shape == (8, ACTION_DIM)
    # Buttons decode to {0, 1}.
    buttons = raw[:, 6:]
    assert set(np.unique(buttons)).issubset({0.0, 1.0})


def test_loss_decreases_on_single_batch():
    """A quick optimization sanity check: the model can overfit one batch."""
    torch.manual_seed(0)
    model = NitroGen(_tiny_config())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    frames = torch.rand(4, 3, 64, 64)
    chunks = torch.rand(4, 8, ACTION_DIM) * 2 - 1
    first = None
    for _ in range(30):
        loss = model.compute_loss(frames, chunks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first
