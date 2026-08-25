"""The robustness harness itself: faults must be honest and rollouts must finish."""

import numpy as np
import pytest

from nitrogen import NitroGen, NitroGenConfig
from nitrogen.benchmark.robustness import (
    DEFAULT_PERTURBATIONS,
    format_report,
    robustness_report,
    run_guarded_episode,
    run_raw_episode,
)
from nitrogen.models.vision_encoder import VisionConfig


@pytest.fixture(scope="module")
def tiny():
    return NitroGen(NitroGenConfig(
        chunk_size=4,
        vision=VisionConfig(image_size=32, patch_size=16, width=32, depth=1, num_heads=2),
    ))


def frame():
    return np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)


# -- the faults are what they claim to be ------------------------------------

def test_every_perturbation_changes_or_drops_the_frame():
    rng = np.random.default_rng(0)
    # Frames must differ per tick, or a stalling fault is indistinguishable from
    # a working capture.
    stream = [np.full((16, 16, 3), t * 7, dtype=np.uint8) for t in range(20)]
    for name, fault in DEFAULT_PERTURBATIONS.items():
        if name == "clean":
            continue
        seen = [fault(stream[t], t, rng) for t in range(20)]
        assert any(
            s is None or s.shape != stream[t].shape or not np.array_equal(s, stream[t])
            for t, s in enumerate(seen)
        ), f"{name} was a no-op"


def test_clean_is_the_identity():
    base = frame()
    assert DEFAULT_PERTURBATIONS["clean"](base, 0, np.random.default_rng(0)) is base


def test_dropped_frames_actually_drops_some_but_not_all():
    from nitrogen.benchmark.robustness import dropped_frames

    fault = dropped_frames(0.5)
    rng = np.random.default_rng(1)
    seen = [fault(frame(), t, rng) for t in range(60)]
    assert 0 < sum(s is None for s in seen) < 60


def test_frozen_capture_repeats_the_same_pixels():
    from nitrogen.benchmark.robustness import frozen_capture

    fault = frozen_capture(stall_every=4, stall_for=3)
    rng = np.random.default_rng(0)
    seen = [fault(np.full((8, 8, 3), t, dtype=np.uint8), t, rng) for t in range(10)]
    assert any(np.array_equal(seen[i], seen[i + 1]) for i in range(len(seen) - 1))


def test_hdr_float_injects_nonfinite_pixels_the_runtime_must_survive():
    out = DEFAULT_PERTURBATIONS["hdr_float"](frame(), 0, np.random.default_rng(0))
    assert out.dtype == np.float32 and not np.isfinite(out).all()


def test_wrong_layout_is_channel_first_and_channel_swapped():
    base = frame()
    out = DEFAULT_PERTURBATIONS["wrong_layout"](base, 0, np.random.default_rng(0))
    assert out.shape == (3, 64, 64)
    assert np.array_equal(out[0], base[:, :, 2])


# -- rollouts terminate and return a verdict under every fault ---------------

@pytest.mark.parametrize("name", list(DEFAULT_PERTURBATIONS))
def test_guarded_rollout_completes_under_every_fault(tiny, name):
    result = run_guarded_episode(tiny, "reacher", seed=1, perturb=DEFAULT_PERTURBATIONS[name],
                                 sample_steps=2)
    assert isinstance(result, bool)


@pytest.mark.parametrize("name", ["clean", "dropped_frames", "hdr_float", "wrong_layout"])
def test_raw_rollout_completes_under_faults_it_can_survive(tiny, name):
    result = run_raw_episode(tiny, "reacher", seed=1, perturb=DEFAULT_PERTURBATIONS[name],
                             sample_steps=2)
    assert isinstance(result, bool)


def test_latency_compensation_does_not_stall_either_loop(tiny):
    kwargs = dict(game="reacher", seed=2, perturb=DEFAULT_PERTURBATIONS["clean"],
                  sample_steps=2, latency_steps=2)
    assert isinstance(run_guarded_episode(tiny, **kwargs), bool)
    assert isinstance(run_raw_episode(tiny, **kwargs), bool)


def test_latency_longer_than_the_chunk_is_survivable(tiny):
    # Every sampled action is stale on arrival; the loop must idle, not hang.
    assert isinstance(
        run_guarded_episode(tiny, game="reacher", seed=2,
                            perturb=DEFAULT_PERTURBATIONS["clean"],
                            sample_steps=2, latency_steps=99),
        bool,
    )


# -- the report --------------------------------------------------------------

def test_report_covers_both_arms_of_every_requested_fault(tiny):
    faults = {k: DEFAULT_PERTURBATIONS[k] for k in ("clean", "dropped_frames")}
    report = robustness_report(tiny, games=["reacher"], episodes=1,
                               perturbations=faults, sample_steps=2)
    assert set(report) == set(faults)
    for row in report.values():
        assert 0.0 <= row["raw"] <= 1.0 and 0.0 <= row["guarded"] <= 1.0
        assert row["delta"] == pytest.approx(row["guarded"] - row["raw"])


def test_format_report_renders_every_row(tiny):
    text = format_report({"clean": {"raw": 0.5, "guarded": 0.6, "delta": 0.1}})
    assert "clean" in text and "50.0%" in text and "+10.0%" in text
