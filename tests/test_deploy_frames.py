"""Capture-pipeline edge cases: everything a real screen grabber can hand you."""

import numpy as np
import pytest

from nitrogen.deploy.frames import FrameSanitizer, letterbox, resize_nearest


def good_frame(h=64, w=64, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


def noisy_frame(seed, h=64, w=64):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# -- the happy path stays cheap and lossless --------------------------------

def test_uint8_frame_passes_through_as_unit_float():
    pixels, health = FrameSanitizer()(good_frame(value=255))
    assert health.ok and not health.repaired
    assert pixels.dtype == np.float32 and pixels.shape == (64, 64, 3)
    assert np.allclose(pixels, 1.0)


# -- layout coercion ---------------------------------------------------------

def test_channel_first_frame_is_transposed():
    chw = np.zeros((3, 32, 48), dtype=np.uint8)
    chw[0] = 255
    pixels, health = FrameSanitizer()(chw)
    assert pixels.shape == (32, 48, 3) and "chw_to_hwc" in health.repaired
    assert np.allclose(pixels[..., 0], 1.0) and np.allclose(pixels[..., 1], 0.0)


def test_grayscale_plane_is_broadcast_to_rgb():
    pixels, health = FrameSanitizer()(np.full((16, 16), 64, dtype=np.uint8))
    assert pixels.shape == (16, 16, 3) and "gray_to_rgb" in health.repaired
    assert np.allclose(pixels, 64 / 255)


def test_rgba_alpha_channel_is_dropped():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    pixels, health = FrameSanitizer()(rgba)
    assert pixels.shape == (8, 8, 3) and "dropped_alpha" in health.repaired


def test_bgr_capture_is_swapped_when_declared():
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[..., 0] = 255                       # blue in BGR
    pixels, _ = FrameSanitizer(bgr=True)(bgr)
    assert np.allclose(pixels[..., 2], 1.0) and np.allclose(pixels[..., 0], 0.0)


def test_ambiguous_small_frame_is_read_channel_last():
    # (3, 4, 3) could be CHW or HWC. Guessing CHW would silently transpose real
    # frames from tiny games, so the channel-last reading must win.
    pixels, health = FrameSanitizer()(np.zeros((3, 4, 3), dtype=np.uint8))
    assert pixels.shape == (3, 4, 3) and "chw_to_hwc" not in health.repaired


# -- dtype coercion ----------------------------------------------------------

def test_uint16_capture_is_scaled_by_its_own_range():
    pixels, health = FrameSanitizer()(np.full((8, 8, 3), 65535, dtype=np.uint16))
    assert "uint16_scaled" in health.repaired and np.allclose(pixels, 1.0)


def test_float_frame_left_in_0_255_is_rescaled():
    pixels, health = FrameSanitizer()(np.full((8, 8, 3), 255.0, dtype=np.float32))
    assert "rescaled_from_255" in health.repaired and np.allclose(pixels, 1.0)


def test_float_frame_already_in_unit_range_is_not_rescaled():
    pixels, health = FrameSanitizer()(np.full((8, 8, 3), 0.5, dtype=np.float64))
    assert "rescaled_from_255" not in health.repaired and np.allclose(pixels, 0.5)


# -- numerical corruption ----------------------------------------------------

def test_partial_nan_is_repaired_and_flagged():
    frame = np.full((8, 8, 3), 0.5, dtype=np.float32)
    frame[0, 0, 0] = np.nan
    frame[1, 1, 1] = np.inf
    pixels, health = FrameSanitizer()(frame)
    assert health.ok and np.isfinite(pixels).all()
    assert any(r.startswith("nonfinite") for r in health.repaired)


def test_all_nan_frame_is_rejected_not_zero_filled():
    pixels, health = FrameSanitizer()(np.full((8, 8, 3), np.nan, dtype=np.float32))
    assert pixels is None and health.reason == "all_nonfinite"


def test_out_of_range_hdr_values_are_clipped():
    frame = np.full((8, 8, 3), 0.5, dtype=np.float32)
    frame[0, 0] = -3.0
    pixels, health = FrameSanitizer()(frame)
    assert "clipped" in health.repaired and pixels.min() >= 0.0 and pixels.max() <= 1.0


# -- structurally unusable input --------------------------------------------

@pytest.mark.parametrize("bad,expected", [
    (None, "frame_is_none"),
    (np.zeros((0, 8, 3), dtype=np.uint8), "empty_frame"),
    (np.zeros((8,), dtype=np.uint8), "bad_shape"),
    (np.zeros((2, 2, 2, 3), dtype=np.uint8), "bad_shape"),
    (np.zeros((8, 8, 5), dtype=np.uint8), "bad_shape"),
    (np.zeros((1, 8, 3), dtype=np.uint8), "too_small"),
    (np.zeros((4, 4, 3), dtype=np.complex64), "bad_dtype"),
])
def test_unusable_frames_are_reported_never_raised(bad, expected):
    pixels, health = FrameSanitizer()(bad)
    assert pixels is None and not health.ok and health.reason.startswith(expected)


# -- resolution --------------------------------------------------------------

def test_resize_to_target_size():
    pixels, _ = FrameSanitizer(target_size=(32, 32))(good_frame(100, 250))
    assert pixels.shape == (32, 32, 3)


def test_letterbox_preserves_aspect_ratio_of_ultrawide_capture():
    # A 21:9 frame squashed to a square distorts every on-screen distance the
    # policy was trained on; letterboxing pads instead.
    frame = np.full((90, 210, 3), 200, dtype=np.uint8)
    pixels, _ = FrameSanitizer(target_size=(64, 64), keep_aspect=True)(frame)
    assert pixels.shape == (64, 64, 3)
    content_rows = np.where(pixels.mean(axis=(1, 2)) > 0.1)[0]
    assert len(content_rows) < 64                    # padding bars exist
    # ...and they are centered, not stacked on one edge.
    assert abs(content_rows.min() - (63 - content_rows.max())) <= 1


def test_resize_nearest_is_exact_on_identity():
    frame = noisy_frame(3, 20, 20)
    assert np.array_equal(resize_nearest(frame, 20, 20), frame)


def test_letterbox_of_square_input_fills_the_target():
    out = letterbox(np.full((10, 10, 3), 7, dtype=np.uint8), 20, 20)
    assert out.shape == (20, 20, 3) and (out == 7).all()


# -- black / frozen / saturated detection ------------------------------------

def test_black_loading_screen_is_flagged_but_usable():
    pixels, health = FrameSanitizer()(np.zeros((16, 16, 3), dtype=np.uint8))
    assert health.ok and health.black and health.degraded and pixels is not None


def test_blown_out_white_frame_is_flagged_saturated():
    _, health = FrameSanitizer()(good_frame(value=255))
    assert health.saturated


def test_frozen_capture_needs_patience_consecutive_repeats():
    san = FrameSanitizer(freeze_patience=3)
    frame = noisy_frame(0)
    healths = [san(frame)[1] for _ in range(4)]
    assert [h.frozen for h in healths] == [False, False, False, True]


def test_motion_resets_the_freeze_counter():
    san = FrameSanitizer(freeze_patience=2)
    still = noisy_frame(1)
    san(still), san(still)
    assert san(still)[1].frozen
    assert not san(noisy_frame(2))[1].frozen          # capture recovered
    assert san.repeats == 0


def test_freeze_tolerance_absorbs_lossy_encoder_noise():
    # A lossy capture never repeats a frame bit-for-bit, so a zero tolerance
    # would never detect a genuine stall on that pipeline.
    base = np.full((16, 16, 3), 0.5, dtype=np.float32)
    san = FrameSanitizer(freeze_patience=2, freeze_tolerance=0.01)
    rng = np.random.default_rng(0)
    frozen = [san(base + rng.normal(0, 0.001, base.shape).astype(np.float32))[1].frozen
              for _ in range(3)]
    assert frozen[-1]


def test_reset_clears_freeze_history():
    san = FrameSanitizer(freeze_patience=2)
    frame = noisy_frame(5)
    for _ in range(3):
        san(frame)
    san.reset()
    assert not san(frame)[1].frozen and san.last_good is not None


def test_rejected_frame_does_not_disturb_freeze_tracking():
    san = FrameSanitizer(freeze_patience=2)
    frame = noisy_frame(6)
    san(frame)
    san(None)                                        # dropped capture
    san(frame)
    assert san(frame)[1].frozen                      # 3rd identical usable frame


def test_invalid_frame_preserves_previous_good_pixels():
    san = FrameSanitizer()
    ok, _ = san(good_frame())
    san(None)
    assert san.last_good is not None and np.array_equal(san.last_good, ok)


def test_freeze_patience_must_be_positive():
    with pytest.raises(ValueError):
        FrameSanitizer(freeze_patience=0)
