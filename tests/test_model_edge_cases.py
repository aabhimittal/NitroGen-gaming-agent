"""Edge cases in the model itself — the ones a deployment actually hits.

Shapes and losses are covered in ``test_model.py``; this file is about the
boundaries: degenerate batch and chunk sizes, unusual input resolutions and
dtypes, sampler step counts, and the guarantee that a sampled action is always
inside the action space no matter what the ODE did on the way there.
"""

import numpy as np
import pytest
import torch

from nitrogen import NitroGen, NitroGenConfig
from nitrogen.action_space import (
    ACTION_DIM,
    BUTTON_SLICE,
    STICK_SLICE,
    TRIGGER_SLICE,
    decode_action,
    encode_action,
)
from nitrogen.models.vision_encoder import VisionConfig


def tiny_model(chunk_size=4, image_size=32) -> NitroGen:
    """The smallest configuration that still exercises every code path."""
    return NitroGen(NitroGenConfig(
        chunk_size=chunk_size,
        vision=VisionConfig(image_size=image_size, patch_size=16, width=32,
                            depth=1, num_heads=2),
    ))


# -- degenerate batch / chunk sizes ------------------------------------------

def test_single_sample_batch():
    model = tiny_model()
    out = model.sample_actions(torch.rand(1, 3, 32, 32))
    assert out.shape == (1, 4, ACTION_DIM)


def test_empty_batch_is_a_no_op_not_a_crash():
    # Batches can legitimately arrive empty from a filtered eval loader.
    model = tiny_model()
    out = model.sample_actions(torch.rand(0, 3, 32, 32))
    assert out.shape == (0, 4, ACTION_DIM)


def test_chunk_size_of_one_degenerates_gracefully():
    model = tiny_model(chunk_size=1)
    assert model.sample_actions(torch.rand(2, 3, 32, 32)).shape == (2, 1, ACTION_DIM)


def test_single_ode_step_still_produces_a_valid_action():
    # The cheapest possible sampler setting — one Euler step from noise.
    out = tiny_model().sample_actions(torch.rand(2, 3, 32, 32), num_steps=1)
    assert torch.isfinite(out).all() and out.abs().max() <= 1.0


def test_more_ode_steps_stay_in_range():
    out = tiny_model().sample_actions(torch.rand(1, 3, 32, 32), num_steps=64)
    assert torch.isfinite(out).all() and out.abs().max() <= 1.0


# -- input resolution and dtype ----------------------------------------------

@pytest.mark.parametrize("hw", [(32, 32), (64, 64), (17, 41), (256, 256)])
def test_encoder_accepts_any_resolution_and_aspect_ratio(hw):
    model = tiny_model()
    tokens = model.encode(torch.rand(1, 3, *hw))
    assert tokens.shape == (1, model.cfg.vision.num_patches, model.cfg.vision.width)


def test_act_accepts_uint8_and_float_frames_identically():
    model = tiny_model()
    frame_u8 = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    torch.manual_seed(0)
    a = model.act(frame_u8)
    torch.manual_seed(0)
    b = model.act(frame_u8.astype(np.float32) / 255.0)
    assert np.allclose(a, b, atol=1e-5)


def test_act_returns_raw_hardware_ranges():
    action = tiny_model().act(np.zeros((32, 32, 3), dtype=np.uint8))
    assert action.shape == (4, ACTION_DIM)
    assert action[:, STICK_SLICE].min() >= -1.0 and action[:, STICK_SLICE].max() <= 1.0
    assert action[:, TRIGGER_SLICE].min() >= 0.0 and action[:, TRIGGER_SLICE].max() <= 1.0
    assert set(np.unique(action[:, BUTTON_SLICE])) <= {0.0, 1.0}


# -- determinism --------------------------------------------------------------

def test_sampling_is_reproducible_under_a_fixed_seed():
    # Without this, a regression run against fixed frames can't distinguish a
    # real regression from resampling noise.
    model = tiny_model()
    frames = torch.rand(2, 3, 32, 32)
    torch.manual_seed(7)
    first = model.sample_actions(frames)
    torch.manual_seed(7)
    assert torch.allclose(first, model.sample_actions(frames))


def test_sampling_is_stochastic_without_a_seed():
    model = tiny_model()
    frames = torch.rand(1, 3, 32, 32)
    assert not torch.allclose(model.sample_actions(frames), model.sample_actions(frames))


def test_sample_actions_leaves_the_model_in_eval_mode_without_grads():
    model = tiny_model()
    model.train()
    out = model.sample_actions(torch.rand(1, 3, 32, 32))
    assert not out.requires_grad and not model.training


# -- training-side edges ------------------------------------------------------

def test_loss_is_finite_on_saturated_targets():
    # Real chunks sit at the corners of the cube: full deflection, buttons at
    # exactly +/-1. Nothing in the loss may blow up there.
    model = tiny_model()
    chunks = torch.sign(torch.randn(3, 4, ACTION_DIM))
    loss = model.compute_loss(torch.rand(3, 3, 32, 32), chunks)
    assert torch.isfinite(loss) and loss.item() >= 0.0


def test_loss_backward_reaches_both_towers():
    model = tiny_model()
    loss = model.compute_loss(torch.rand(2, 3, 32, 32), torch.zeros(2, 4, ACTION_DIM))
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert any(n.startswith("encoder") for n in grads)
    assert any(n.startswith("action_head") for n in grads)
    assert all(torch.isfinite(g).all() for g in grads.values())


def test_chunk_size_mismatch_between_config_and_batch_is_rejected():
    model = tiny_model(chunk_size=4)
    with pytest.raises(RuntimeError):
        model.compute_loss(torch.rand(2, 3, 32, 32), torch.zeros(2, 9, ACTION_DIM))


# -- action-space round trips -------------------------------------------------

def test_encode_decode_round_trip_is_exact_on_valid_actions():
    rng = np.random.default_rng(0)
    raw = np.concatenate([
        rng.uniform(-1, 1, 4), rng.uniform(0, 1, 2),
        (rng.random(12) < 0.5).astype(np.float64),
    ]).astype(np.float32)
    assert np.allclose(decode_action(encode_action(raw)), raw, atol=1e-6)


def test_button_threshold_sits_exactly_at_the_encoded_midpoint():
    # 0 is the image of 0.5 under encoding; a sample landing exactly there must
    # resolve deterministically rather than by float luck.
    norm = np.zeros(ACTION_DIM, dtype=np.float32)
    assert (decode_action(norm)[BUTTON_SLICE] == 0.0).all()
    norm[BUTTON_SLICE] = 1e-6
    assert (decode_action(norm)[BUTTON_SLICE] == 1.0).all()


def test_out_of_range_model_output_is_clamped_by_decode():
    norm = np.full(ACTION_DIM, 5.0, dtype=np.float32)
    out = decode_action(norm)
    assert out[STICK_SLICE].max() <= 1.0 and out[TRIGGER_SLICE].max() <= 1.0


def test_encode_handles_batched_and_single_vectors_alike():
    raw = np.zeros((3, 5, ACTION_DIM), dtype=np.float32)
    assert encode_action(raw).shape == (3, 5, ACTION_DIM)
    assert encode_action(raw[0, 0]).shape == (ACTION_DIM,)
