"""Runtime supervision: the loop must never raise, never stall, never latch."""

import numpy as np
import pytest

from nitrogen.action_space import ACTION_DIM, BUTTON_SLICE
from nitrogen.deploy.controller import ChunkExecutor
from nitrogen.deploy.frames import FrameSanitizer
from nitrogen.deploy.runtime import PolicyRuntime, RuntimeMetrics, StepReport
from nitrogen.deploy.safety import ActionGuard, GamepadLimits


def frame(seed=0, h=32, w=32):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def constant_policy(value=0.5, length=8):
    def policy(pixels):
        chunk = np.zeros((length, ACTION_DIM), dtype=np.float32)
        chunk[:, 0] = value
        return chunk
    return policy


def counting_policy(length=8):
    """Records how many times it was actually invoked."""
    calls = {"n": 0}

    def policy(pixels):
        calls["n"] += 1
        return np.zeros((length, ACTION_DIM), dtype=np.float32)
    return policy, calls


def permissive_runtime(policy, **kwargs):
    """No deadzone/slew shaping, so tests read the executor's output directly."""
    kwargs.setdefault("guard", ActionGuard(GamepadLimits(
        stick_deadzone=0.0, stick_max_rate=10.0, trigger_max_rate=10.0,
        min_hold_ticks=1, blocked_buttons=())))
    return PolicyRuntime(policy, **kwargs)


# -- the happy path ----------------------------------------------------------

def test_step_returns_a_legal_action_and_a_report():
    rt = permissive_runtime(constant_policy(0.5))
    report = rt.step(frame())
    assert isinstance(report, StepReport) and report.ok and report.replanned
    assert report.action.shape == (ACTION_DIM,)
    assert float(report.action[0]) == pytest.approx(0.5)


def test_policy_is_called_only_on_replan_ticks():
    policy, calls = counting_policy()
    rt = permissive_runtime(policy, executor=ChunkExecutor(replan_every=4))
    for i in range(8):
        rt.step(frame(i))
    assert calls["n"] == 2
    assert rt.metrics.summary()["replans"] == 2


def test_tick_advances_once_per_step_whatever_happens():
    rt = permissive_runtime(constant_policy())
    for i, f in enumerate([frame(1), None, np.zeros((16, 16, 3), np.uint8), frame(2)]):
        assert rt.step(f).tick == i
    assert rt.executor.tick == 4


# -- degrading instead of failing -------------------------------------------

def test_a_broken_frame_keeps_executing_the_buffered_chunk():
    # Losing one frame must not stop the agent: the buffered plan is still the
    # best available estimate of what to do next.
    rt = permissive_runtime(constant_policy(0.6), executor=ChunkExecutor(replan_every=8))
    rt.step(frame())
    report = rt.step(None)
    assert report.reason.startswith("bad_frame") and report.degraded
    assert float(report.action[0]) == pytest.approx(0.6)
    assert rt.metrics.summary()["bad_frames"] == 1


def test_a_throwing_policy_never_propagates():
    def exploding(pixels):
        raise RuntimeError("CUDA out of memory")

    rt = permissive_runtime(exploding)
    report = rt.step(frame())
    assert report.reason == "policy_error:RuntimeError" and report.degraded
    assert np.allclose(report.action, 0.0)


def test_nan_chunk_is_rejected_before_it_reaches_the_pad():
    def diverged(pixels):
        return np.full((8, ACTION_DIM), np.nan, dtype=np.float32)

    rt = permissive_runtime(diverged)
    report = rt.step(frame())
    assert report.reason == "nonfinite_chunk"
    assert np.isfinite(report.action).all() and np.allclose(report.action, 0.0)
    assert rt.metrics.summary()["nonfinite_chunks"] == 1


@pytest.mark.parametrize("bad", [
    np.zeros((8, ACTION_DIM - 2), dtype=np.float32),
    np.zeros(ACTION_DIM, dtype=np.float32),
    np.zeros((0, ACTION_DIM), dtype=np.float32),
])
def test_wrongly_shaped_chunks_are_rejected(bad):
    rt = permissive_runtime(lambda pixels: bad)
    assert rt.step(frame()).reason.startswith("bad_chunk_shape")


def test_frozen_capture_idles_instead_of_acting_on_stale_pixels():
    rt = permissive_runtime(constant_policy(1.0),
                            sanitizer=FrameSanitizer(freeze_patience=2))
    still = frame(7)
    for _ in range(3):
        report = rt.step(still)
    assert report.reason == "frozen_capture" and np.allclose(report.action, 0.0)
    assert rt.metrics.summary()["frozen_frames"] >= 1


def test_black_loading_screen_idles_and_can_be_opted_out():
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    assert permissive_runtime(constant_policy(1.0)).step(black).reason == "black_frame"
    permissive = permissive_runtime(constant_policy(1.0), hold_on_black=False)
    assert permissive.step(black).ok


def test_idling_does_not_latch_a_held_button():
    # The one failure mode a control loop can never have: a button stuck down
    # because the loop stopped updating.
    def presser(pixels):
        chunk = np.zeros((8, ACTION_DIM), dtype=np.float32)
        chunk[:, BUTTON_SLICE] = 1.0
        return chunk

    rt = permissive_runtime(presser, sanitizer=FrameSanitizer(freeze_patience=2))
    still = frame(9)
    rt.step(still)
    assert rt.step(still).action[BUTTON_SLICE].sum() > 0
    assert rt.step(still).action[BUTTON_SLICE].sum() == 0     # frozen -> released


# -- circuit breaker ---------------------------------------------------------

def test_breaker_opens_after_repeated_failures_and_stops_calling_the_policy():
    calls = {"n": 0}

    def flaky(pixels):
        calls["n"] += 1
        raise ValueError("boom")

    rt = permissive_runtime(flaky, max_consecutive_failures=2, recover_after=20,
                            executor=ChunkExecutor(replan_every=1))
    reports = [rt.step(frame(i)) for i in range(6)]
    assert calls["n"] == 2                                   # then the breaker opened
    assert reports[-1].reason == "breaker_open"
    assert rt.breaker_open and rt.metrics.summary()["breaker_trips"] == 1


def test_breaker_retries_after_the_cooldown_and_recovers():
    state = {"fail": True}

    def flaky(pixels):
        if state["fail"]:
            raise ValueError("boom")
        chunk = np.zeros((4, ACTION_DIM), dtype=np.float32)
        chunk[:, 0] = 0.9
        return chunk

    rt = permissive_runtime(flaky, max_consecutive_failures=1, recover_after=3,
                            executor=ChunkExecutor(replan_every=1))
    rt.step(frame(0))                                        # trips immediately
    assert rt.breaker_open
    state["fail"] = False
    for _ in range(3):
        report = rt.step(frame(1))
    assert report.ok and float(report.action[0]) == pytest.approx(0.9)
    assert not rt.breaker_open


def test_an_isolated_failure_does_not_trip_the_breaker():
    state = {"n": 0}

    def occasionally(pixels):
        state["n"] += 1
        if state["n"] == 2:
            raise ValueError("transient")
        return np.zeros((4, ACTION_DIM), dtype=np.float32)

    rt = permissive_runtime(occasionally, max_consecutive_failures=3,
                            executor=ChunkExecutor(replan_every=1))
    for i in range(5):
        rt.step(frame(i))
    assert not rt.breaker_open
    assert rt.metrics.summary()["policy_errors"] == 1


# -- metrics -----------------------------------------------------------------

def test_metrics_track_latency_percentiles_and_degradation():
    rt = permissive_runtime(constant_policy(), executor=ChunkExecutor(replan_every=1))
    for i in range(10):
        rt.step(frame(i))
    rt.step(None)
    s = rt.metrics.summary()
    assert s["ticks"] == 11 and s["bad_frames"] == 1
    assert 0.0 < s["degraded_rate"] < 1.0
    assert s["latency_p95_ms"] >= s["latency_p50_ms"] >= 0.0


def test_latency_budget_misses_are_counted_not_enforced():
    import time

    def slow(pixels):
        time.sleep(0.01)
        return np.zeros((4, ACTION_DIM), dtype=np.float32)

    rt = permissive_runtime(slow, latency_budget_ms=1.0,
                            executor=ChunkExecutor(replan_every=1))
    report = rt.step(frame())
    assert report.ok                                  # late, but still delivered
    assert rt.metrics.summary()["budget_misses"] == 1


def test_metrics_summary_is_safe_before_any_tick():
    s = RuntimeMetrics().summary()
    assert s["ticks"] == 0 and s["latency_p50_ms"] == 0.0 and s["degraded_rate"] == 0.0


def test_reset_clears_runtime_state():
    rt = permissive_runtime(constant_policy())
    rt.step(frame())
    rt.reset()
    assert rt.executor.tick == 0 and rt.metrics.summary()["ticks"] == 0
    assert not rt.breaker_open


def test_invalid_supervision_settings_are_rejected():
    for kwargs in (dict(max_consecutive_failures=0), dict(recover_after=0)):
        with pytest.raises(ValueError):
            PolicyRuntime(constant_policy(), **kwargs)


# -- end to end against the toy game ----------------------------------------

def test_runtime_survives_a_hostile_capture_stream_end_to_end():
    """A full episode where the capture layer misbehaves in a different way
    every few ticks. The loop must produce a legal action for every single tick."""
    from nitrogen.envs.toy_game import make_game

    env = make_game("reacher")
    obs = env.reset(seed=3)
    rt = PolicyRuntime(constant_policy(0.4), executor=ChunkExecutor(replan_every=3))

    corruptions = [
        lambda f: f,
        lambda f: None,
        lambda f: f.astype(np.float32) / 255.0,
        lambda f: np.transpose(f, (2, 0, 1)),
        lambda f: np.concatenate([f, np.full(f.shape[:2] + (1,), 255, np.uint8)], axis=2),
        lambda f: np.zeros_like(f),
        lambda f: f.astype(np.float32) * np.nan,
        lambda f: f[:, :, 0],
        lambda f: f.astype(np.uint16) * 257,
    ]
    done, t = False, 0
    while not done:
        report = rt.step(corruptions[t % len(corruptions)](obs))
        assert np.isfinite(report.action).all()
        assert report.action.shape == (ACTION_DIM,)
        obs, done, info = env.step(report.action)
        t += 1
    assert rt.metrics.summary()["ticks"] == t
