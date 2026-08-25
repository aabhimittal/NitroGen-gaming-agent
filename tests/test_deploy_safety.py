"""Hardware-legality edge cases for actions coming out of a generative head."""

import numpy as np
import pytest

from nitrogen.action_space import (
    ACTION_DIM,
    BUTTON_NAMES,
    BUTTON_SLICE,
    GamepadAction,
    STICK_SLICE,
    TRIGGER_SLICE,
)
from nitrogen.deploy.safety import ActionGuard, GamepadLimits


def raw(**kw) -> np.ndarray:
    """A raw action vector built from named gamepad fields."""
    buttons = kw.pop("buttons", {})
    return GamepadAction(buttons=buttons, **kw).to_vector()


def loose(**kw) -> ActionGuard:
    """A guard with the temporal limits relaxed, to isolate one behaviour."""
    defaults = dict(stick_max_rate=10.0, trigger_max_rate=10.0, min_hold_ticks=1,
                    blocked_buttons=())
    defaults.update(kw)
    return ActionGuard(GamepadLimits(**defaults))


# -- shape and numerics ------------------------------------------------------

def test_wrong_width_is_rejected_loudly():
    with pytest.raises(ValueError, match=str(ACTION_DIM)):
        ActionGuard()(np.zeros(ACTION_DIM - 1, dtype=np.float32))


def test_nan_channels_become_neutral_never_reach_the_pad():
    vec = raw(left_x=np.nan, right_y=np.inf)
    out = loose()(vec)
    assert np.isfinite(out).all() and out[0] == 0.0 and out[3] == 0.0
    assert loose().stats["calls"] == 0                # fresh guard, separate stats


def test_output_is_always_within_hardware_ranges():
    rng = np.random.default_rng(0)
    guard = ActionGuard()
    for _ in range(50):
        out = guard(rng.uniform(-5, 5, ACTION_DIM).astype(np.float32))
        assert out[STICK_SLICE].min() >= -1.0 and out[STICK_SLICE].max() <= 1.0
        assert out[TRIGGER_SLICE].min() >= 0.0 and out[TRIGGER_SLICE].max() <= 1.0
        assert set(np.unique(out[BUTTON_SLICE])) <= {0.0, 1.0}


def test_guard_does_not_mutate_the_caller_array():
    vec = raw(left_x=0.02)
    copy = vec.copy()
    loose()(vec)
    assert np.array_equal(vec, copy)


# -- stick geometry ----------------------------------------------------------

def test_diagonal_beyond_the_unit_circle_is_clamped_preserving_direction():
    out = loose()(raw(left_x=0.98, right_x=0.0, left_y=0.98))
    mag = np.hypot(out[0], out[1])
    assert mag == pytest.approx(1.0, abs=1e-5)
    assert out[0] == pytest.approx(out[1], abs=1e-6)   # 45 degrees preserved


def test_radial_deadzone_kills_idle_drift():
    out = loose(stick_deadzone=0.1)(raw(left_x=0.05, left_y=0.05))
    assert out[0] == 0.0 and out[1] == 0.0


def test_deadzone_is_radial_not_axiswise():
    # Axis-wise deadzones snap a slow diagonal to a cardinal direction. A vector
    # of magnitude 0.14 must survive intact under a 0.1 deadzone.
    out = loose(stick_deadzone=0.1)(raw(left_x=0.1, left_y=0.1))
    assert out[0] == pytest.approx(0.1) and out[1] == pytest.approx(0.1)


def test_each_stick_is_gated_independently():
    out = loose(stick_deadzone=0.2)(raw(left_x=0.01, right_x=0.9))
    assert out[0] == 0.0 and out[2] == pytest.approx(0.9)


def test_triggers_are_clamped_and_deadzoned():
    out = loose(trigger_deadzone=0.1)(raw(left_trigger=0.02, right_trigger=5.0))
    assert out[4] == 0.0 and out[5] == 1.0


# -- slew limiting -----------------------------------------------------------

def test_full_deflection_reversal_is_rate_limited():
    guard = ActionGuard(GamepadLimits(stick_max_rate=0.25, min_hold_ticks=1))
    guard(raw(left_x=1.0))                       # ramps up over several ticks
    for _ in range(8):
        guard(raw(left_x=1.0))
    assert guard(raw(left_x=-1.0))[0] == pytest.approx(0.75)
    assert guard.stats["rate_limited"] > 0


def test_rate_limiting_converges_to_the_commanded_value():
    guard = ActionGuard(GamepadLimits(stick_max_rate=0.3, min_hold_ticks=1))
    for _ in range(10):
        out = guard(raw(left_x=1.0))
    assert out[0] == pytest.approx(1.0)


def test_reset_clears_slew_state_between_episodes():
    guard = ActionGuard(GamepadLimits(stick_max_rate=0.2, min_hold_ticks=1))
    for _ in range(6):
        guard(raw(left_x=1.0))
    guard.reset()
    assert guard(raw(left_x=1.0))[0] == pytest.approx(0.2)


# -- buttons -----------------------------------------------------------------

def test_menu_buttons_are_blocked_by_default():
    guard = ActionGuard()
    out = guard(raw(buttons={"start": True, "back": True, "A": True}))
    got = GamepadAction.from_vector(out)
    assert not got.button("start") and not got.button("back") and got.button("A")
    assert guard.stats["buttons_blocked"] == 2


def test_opposing_dpad_directions_cannot_both_be_emitted():
    guard = loose()
    out = guard(raw(buttons={"dpad_up": True, "dpad_down": True}))
    got = GamepadAction.from_vector(out)
    assert not (got.button("dpad_up") and got.button("dpad_down"))
    assert guard.stats["exclusive_resolved"] == 1


def test_exclusive_tie_releases_both_rather_than_guessing():
    vec = np.zeros(ACTION_DIM, dtype=np.float32)
    vec[BUTTON_SLICE][BUTTON_NAMES.index("dpad_left")] = 1.0
    vec[BUTTON_SLICE][BUTTON_NAMES.index("dpad_right")] = 1.0
    got = GamepadAction.from_vector(loose()(vec))
    assert not got.button("dpad_left") and not got.button("dpad_right")


def test_stronger_direction_wins_a_continuous_conflict():
    vec = np.zeros(ACTION_DIM, dtype=np.float32)
    vec[BUTTON_SLICE][BUTTON_NAMES.index("dpad_up")] = 0.95
    vec[BUTTON_SLICE][BUTTON_NAMES.index("dpad_down")] = 0.65
    got = GamepadAction.from_vector(loose()(vec))
    assert got.button("dpad_up") and not got.button("dpad_down")


def test_single_tick_flicker_is_debounced_to_a_real_press():
    guard = loose(min_hold_ticks=3)
    pressed = raw(buttons={"A": True})
    released = raw()
    assert GamepadAction.from_vector(guard(pressed)).button("A")
    for _ in range(2):                                # held open by debounce
        assert GamepadAction.from_vector(guard(released)).button("A")
    assert not GamepadAction.from_vector(guard(released)).button("A")
    assert guard.stats["debounced"] == 2


def test_hysteresis_keeps_a_marginal_button_stable():
    # A continuous channel sitting at 0.5 between thresholds must not chatter.
    guard = loose(press_threshold=0.7, release_threshold=0.3)
    idx = BUTTON_NAMES.index("A")
    marginal = np.zeros(ACTION_DIM, dtype=np.float32)
    marginal[BUTTON_SLICE][idx] = 0.5
    assert not GamepadAction.from_vector(guard(marginal)).button("A")   # never rose
    high = marginal.copy()
    high[BUTTON_SLICE][idx] = 0.8
    assert GamepadAction.from_vector(guard(high)).button("A")
    assert GamepadAction.from_vector(guard(marginal)).button("A")       # stays held


def test_unknown_blocked_button_is_a_configuration_error():
    with pytest.raises(ValueError, match="unknown blocked buttons"):
        GamepadLimits(blocked_buttons=("turbo",))


def test_inverted_thresholds_are_a_configuration_error():
    with pytest.raises(ValueError):
        GamepadLimits(press_threshold=0.2, release_threshold=0.8)


# -- chunk-level API ---------------------------------------------------------

def test_sanitize_chunk_applies_temporal_limits_in_order():
    guard = ActionGuard(GamepadLimits(stick_max_rate=0.25, min_hold_ticks=1))
    chunk = np.stack([raw(left_x=1.0) for _ in range(4)])
    out = guard.sanitize_chunk(chunk)
    assert out.shape == (4, ACTION_DIM)
    assert list(np.round(out[:, 0], 2)) == [0.25, 0.5, 0.75, 1.0]


def test_sanitize_chunk_rejects_a_non_2d_array():
    with pytest.raises(ValueError):
        ActionGuard().sanitize_chunk(np.zeros(ACTION_DIM, dtype=np.float32))


def test_empty_chunk_is_a_no_op():
    out = ActionGuard().sanitize_chunk(np.zeros((0, ACTION_DIM), dtype=np.float32))
    assert out.shape == (0, ACTION_DIM)


def test_as_gamepad_returns_the_ergonomic_view():
    got = loose().as_gamepad(raw(left_x=0.5, buttons={"B": True}))
    assert isinstance(got, GamepadAction) and got.left_x == pytest.approx(0.5)
    assert got.button("B")
