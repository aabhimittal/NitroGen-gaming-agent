"""Chunk-execution edge cases: seams, latency, starvation, and clock discipline."""

import numpy as np
import pytest

from nitrogen.action_space import ACTION_DIM, BUTTON_NAMES, BUTTON_SLICE
from nitrogen.deploy.controller import ChunkExecutor


def ramp_chunk(start: float, length: int = 8, step: float = 0.1) -> np.ndarray:
    """A chunk whose left_x walks linearly — easy to read back per tick."""
    chunk = np.zeros((length, ACTION_DIM), dtype=np.float32)
    chunk[:, 0] = start + step * np.arange(length)
    return chunk


def const_chunk(value: float, length: int = 8) -> np.ndarray:
    chunk = np.zeros((length, ACTION_DIM), dtype=np.float32)
    chunk[:, 0] = value
    return chunk


def button_chunk(pressed: bool, length: int = 8) -> np.ndarray:
    chunk = np.zeros((length, ACTION_DIM), dtype=np.float32)
    chunk[:, BUTTON_SLICE.start + BUTTON_NAMES.index("A")] = float(pressed)
    return chunk


# -- basic contract ----------------------------------------------------------

def test_chunk_is_replayed_one_action_per_tick():
    ex = ChunkExecutor(replan_every=8, ensemble=False)
    ex.submit(ramp_chunk(0.0))
    assert [round(float(ex.step()[0]), 2) for _ in range(4)] == [0.0, 0.1, 0.2, 0.3]
    assert ex.tick == 4


def test_configuration_is_validated():
    for kwargs in (dict(replan_every=0), dict(latency_steps=-1),
                   dict(ensemble_decay=0.0), dict(fallback="panic"),
                   dict(max_pending=0)):
        with pytest.raises(ValueError):
            ChunkExecutor(**kwargs)


@pytest.mark.parametrize("bad", [
    np.zeros((4, ACTION_DIM - 1), dtype=np.float32),
    np.zeros(ACTION_DIM, dtype=np.float32),
    np.zeros((0, ACTION_DIM), dtype=np.float32),
])
def test_malformed_chunks_are_refused(bad):
    with pytest.raises(ValueError):
        ChunkExecutor().submit(bad)


def test_nonfinite_chunk_is_refused_at_the_boundary():
    chunk = const_chunk(0.5)
    chunk[2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ChunkExecutor().submit(chunk)


# -- replanning cadence ------------------------------------------------------

def test_replan_is_requested_before_the_first_chunk():
    assert ChunkExecutor().should_replan


def test_replan_cadence_is_independent_of_chunk_length():
    ex = ChunkExecutor(replan_every=3)
    ex.submit(const_chunk(1.0, length=8))
    flags = []
    for _ in range(4):
        flags.append(ex.should_replan)
        ex.step()
    assert flags == [False, False, False, True]


def test_early_exhaustion_forces_an_immediate_replan():
    # A chunk shorter than the replan interval must not leave the agent
    # uncontrolled until the interval elapses.
    ex = ChunkExecutor(replan_every=10)
    ex.submit(const_chunk(1.0, length=2))
    ex.step(), ex.step()
    assert ex.should_replan and ex.horizon_remaining == 0


# -- temporal ensembling -----------------------------------------------------

def test_overlapping_chunks_are_blended_at_the_seam():
    ex = ChunkExecutor(replan_every=2, ensemble=True, ensemble_decay=0.5)
    ex.submit(const_chunk(0.0))
    ex.step(), ex.step()
    ex.submit(const_chunk(1.0))
    # newest weighted 1, older 0.5 -> (1*1 + 0*0.5) / 1.5
    assert float(ex.step()[0]) == pytest.approx(2 / 3, abs=1e-6)
    assert ex.stats["ensembled"] == 1


def test_ensembling_can_be_disabled_for_strict_newest_wins():
    ex = ChunkExecutor(replan_every=2, ensemble=False)
    ex.submit(const_chunk(0.0))
    ex.step(), ex.step()
    ex.submit(const_chunk(1.0))
    assert float(ex.step()[0]) == pytest.approx(1.0)


def test_buttons_are_voted_never_half_pressed():
    ex = ChunkExecutor(replan_every=2, ensemble=True, ensemble_decay=0.5)
    ex.submit(button_chunk(False))
    ex.step(), ex.step()
    ex.submit(button_chunk(True))
    action = ex.step()
    assert set(np.unique(action[BUTTON_SLICE])) <= {0.0, 1.0}


def test_decay_of_one_is_a_plain_mean():
    ex = ChunkExecutor(replan_every=2, ensemble=True, ensemble_decay=1.0)
    ex.submit(const_chunk(0.0))
    ex.step(), ex.step()
    ex.submit(const_chunk(1.0))
    assert float(ex.step()[0]) == pytest.approx(0.5)


# -- latency compensation ----------------------------------------------------

def test_latency_steps_skip_the_already_stale_head_of_a_chunk():
    ex = ChunkExecutor(replan_every=8, ensemble=False, latency_steps=2)
    ex.submit(ramp_chunk(0.0))
    assert float(ex.step()[0]) == pytest.approx(0.2)


def test_chunk_shorter_than_the_latency_is_dropped_entirely():
    ex = ChunkExecutor(latency_steps=4, fallback="neutral")
    ex.submit(const_chunk(1.0, length=3))
    assert ex.stats["dropped_stale"] == 1 and ex.pending_chunks == 0
    assert float(ex.step()[0]) == 0.0


# -- starvation --------------------------------------------------------------

def test_neutral_fallback_releases_everything_when_starved():
    ex = ChunkExecutor(fallback="neutral")
    ex.submit(const_chunk(1.0, length=1))
    ex.step()
    assert np.allclose(ex.step(), 0.0) and ex.stats["starved"] == 1


def test_hold_fallback_repeats_the_last_action():
    ex = ChunkExecutor(fallback="hold")
    ex.submit(const_chunk(0.7, length=1))
    ex.step()
    assert float(ex.step()[0]) == pytest.approx(0.7)


def test_decay_fallback_relaxes_sticks_and_drops_buttons_immediately():
    ex = ChunkExecutor(fallback="decay", decay_rate=0.5)
    chunk = button_chunk(True, length=1)
    chunk[:, 0] = 1.0
    ex.submit(chunk)
    ex.step()
    first = ex.step()
    assert float(first[0]) == pytest.approx(0.5)
    assert first[BUTTON_SLICE].sum() == 0.0          # a latched button is worse
    assert float(ex.step()[0]) == pytest.approx(0.25)


def test_starved_ticks_still_advance_the_clock():
    ex = ChunkExecutor()
    for _ in range(3):
        ex.step()
    assert ex.tick == 3 and ex.stats["starved"] == 3


# -- memory and lifecycle ----------------------------------------------------

def test_pending_chunks_are_bounded_under_rapid_replanning():
    ex = ChunkExecutor(replan_every=1, max_pending=3)
    for _ in range(50):
        ex.submit(const_chunk(1.0, length=16))
        ex.step()
    assert ex.pending_chunks <= 3


def test_exhausted_chunks_are_evicted():
    ex = ChunkExecutor(replan_every=8)
    ex.submit(const_chunk(1.0, length=2))
    ex.step(), ex.step()
    ex.step()
    assert ex.pending_chunks == 0


def test_reset_restores_a_pristine_executor():
    ex = ChunkExecutor()
    ex.submit(const_chunk(1.0))
    ex.step()
    ex.reset()
    assert ex.tick == 0 and ex.pending_chunks == 0 and ex.should_replan
    assert ex.stats["ticks"] == 0


def test_returned_actions_are_copies():
    ex = ChunkExecutor()
    ex.submit(const_chunk(0.5))
    action = ex.step()
    action[0] = 99.0
    assert float(ex.step()[0]) == pytest.approx(0.5)
