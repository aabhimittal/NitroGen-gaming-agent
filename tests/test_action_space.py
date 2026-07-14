import numpy as np

from nitrogen.action_space import (
    ACTION_DIM,
    BUTTON_NAMES,
    GamepadAction,
    decode_action,
    encode_action,
    null_action,
)


def test_action_dim():
    assert ACTION_DIM == 4 + 2 + len(BUTTON_NAMES)


def test_encode_decode_roundtrip():
    a = GamepadAction(
        left_x=0.5, left_y=-0.3, right_x=-1.0, right_y=0.2,
        left_trigger=0.7, right_trigger=0.0,
        buttons={"A": True, "X": True, "start": True},
    )
    raw = a.to_vector()
    norm = encode_action(raw)
    assert norm.min() >= -1.0 and norm.max() <= 1.0
    back = decode_action(norm)
    # Sticks/triggers recovered closely; buttons exactly.
    np.testing.assert_allclose(back, raw, atol=1e-6)


def test_button_thresholding():
    norm = null_action()
    # Nudge one button channel slightly positive -> should decode to pressed.
    b_index = 4 + 2  # first button
    norm[b_index] = 0.2
    raw = decode_action(norm)
    assert raw[b_index] == 1.0


def test_batched_encode():
    raw = np.stack([GamepadAction().to_vector() for _ in range(5)])
    norm = encode_action(raw)
    assert norm.shape == (5, ACTION_DIM)
