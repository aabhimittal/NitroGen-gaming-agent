"""Dataset edge cases: chunk padding, persistence, and lossy overlay footage."""

import numpy as np
import pytest

from nitrogen.action_space import ACTION_DIM, BUTTON_NAMES, GamepadAction, null_action
from nitrogen.data.dataset import FrameActionChunkDataset
from nitrogen.data.overlay_extraction import (
    OverlayExtractor,
    evaluate_extraction,
    render_overlay,
)
from nitrogen.data.synthetic import (
    Episode,
    collect_dataset,
    collect_episode,
    dataset_stats,
    load_episodes,
    save_episodes,
)


def fake_episode(length=5, game="reacher", size=8) -> Episode:
    rng = np.random.default_rng(length)
    return Episode(
        game=game,
        frames=rng.integers(0, 256, (length, size, size, 3), dtype=np.uint8),
        actions=rng.uniform(-1, 1, (length, ACTION_DIM)).astype(np.float32),
        success=True,
    )


# -- chunking near the end of an episode -------------------------------------

def test_pad_last_true_keeps_every_frame_and_pads_with_the_null_action():
    ds = FrameActionChunkDataset([fake_episode(5)], chunk_size=4, pad_last=True)
    assert len(ds) == 5
    _, chunk = ds[4]                                   # only one real action left
    assert chunk.shape == (4, ACTION_DIM)
    assert np.allclose(chunk[1:].numpy(), null_action(), atol=1e-6)


def test_pad_last_false_drops_frames_without_a_full_chunk():
    ds = FrameActionChunkDataset([fake_episode(5)], chunk_size=4, pad_last=False)
    assert len(ds) == 2                                # frames 0 and 1 only


def test_episode_shorter_than_the_chunk_yields_nothing_when_unpadded():
    # Otherwise a corpus of short episodes trains the policy almost entirely on
    # end-of-episode padding, i.e. on idling.
    ds = FrameActionChunkDataset([fake_episode(3)], chunk_size=8, pad_last=False)
    assert len(ds) == 0


def test_chunk_size_of_one_is_plain_single_step_cloning():
    ds = FrameActionChunkDataset([fake_episode(3)], chunk_size=1)
    assert len(ds) == 3 and ds[0][1].shape == (1, ACTION_DIM)


def test_empty_corpus_is_an_empty_dataset_not_an_error():
    assert len(FrameActionChunkDataset([], chunk_size=8)) == 0


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        FrameActionChunkDataset([fake_episode()], chunk_size=0)


def test_samples_are_normalized_and_channel_first():
    frame, chunk = FrameActionChunkDataset([fake_episode(4, size=16)], chunk_size=2)[0]
    assert frame.shape == (3, 16, 16)
    assert 0.0 <= float(frame.min()) and float(frame.max()) <= 1.0
    assert float(chunk.min()) >= -1.0 and float(chunk.max()) <= 1.0


def test_index_map_spans_multiple_episodes_without_crossing_them():
    eps = [fake_episode(4), fake_episode(6)]
    ds = FrameActionChunkDataset(eps, chunk_size=3, pad_last=False)
    assert len(ds) == 2 + 4
    assert {e for e, _ in ds.index} == {0, 1}


# -- persistence --------------------------------------------------------------

def test_save_load_round_trip_preserves_ragged_episodes(tmp_path):
    eps = [fake_episode(3), fake_episode(7), fake_episode(5)]
    path = str(tmp_path / "corpus.npz")
    save_episodes(eps, path)
    back = load_episodes(path)
    assert [len(e) for e in back] == [3, 7, 5]
    assert back[1].game == eps[1].game and back[1].success == eps[1].success
    assert np.array_equal(back[1].frames, eps[1].frames)
    assert np.allclose(back[1].actions, eps[1].actions)


def test_saving_an_empty_corpus_is_refused(tmp_path):
    with pytest.raises(ValueError):
        save_episodes([], str(tmp_path / "empty.npz"))


# -- collection ---------------------------------------------------------------

def test_collect_episode_aligns_frames_with_the_actions_taken_from_them():
    ep = collect_episode("reacher", seed=0)
    assert len(ep.frames) == len(ep.actions) > 0
    assert ep.frames.dtype == np.uint8 and ep.actions.dtype == np.float32


def test_collection_is_deterministic_for_a_given_seed():
    a, b = collect_episode("reacher", seed=5), collect_episode("reacher", seed=5)
    assert np.array_equal(a.frames, b.frames) and np.allclose(a.actions, b.actions)


def test_games_do_not_share_rollout_seeds():
    # Sharing seeds would correlate object spawns across games and quietly
    # shrink the effective diversity of the corpus.
    eps = collect_dataset(["reacher", "avoider"], episodes_per_game=2, noise=0.0)
    assert len(eps) == 4
    first = [e for e in eps if e.game == "reacher"][0]
    second = [e for e in eps if e.game == "avoider"][0]
    assert not np.allclose(first.actions[:3], second.actions[:3])


def test_zero_episodes_per_game_yields_an_empty_corpus():
    assert collect_dataset(["reacher"], episodes_per_game=0) == []


def test_dataset_stats_summarize_a_corpus():
    stats = dataset_stats([fake_episode(4), fake_episode(6)])
    assert stats["episodes"] == 2 and stats["frames"] == 10
    assert dataset_stats([])["episodes"] == 0


# -- overlay extraction under degraded footage --------------------------------

def test_extraction_returns_none_when_no_overlay_is_present():
    # Cutscenes and menus hide the widget; those frames must be *filtered*, not
    # labeled with a fabricated action.
    assert OverlayExtractor().extract(np.zeros((64, 64, 3), dtype=np.uint8)) is None


@pytest.mark.parametrize("bad", [
    np.zeros((64, 64), dtype=np.uint8),
    np.zeros((64, 64, 1), dtype=np.uint8),
])
def test_extraction_refuses_non_rgb_frames(bad):
    assert OverlayExtractor().extract(bad) is None


def test_extreme_stick_deflections_round_trip():
    action = GamepadAction(left_x=-1.0, left_y=1.0, right_x=1.0, right_y=-1.0)
    got = OverlayExtractor().extract(render_overlay(np.zeros((256, 256, 3), np.uint8), action))
    assert got is not None
    for a, b in ((got.left_x, -1.0), (got.left_y, 1.0), (got.right_x, 1.0), (got.right_y, -1.0)):
        assert a == pytest.approx(b, abs=0.1)


def test_out_of_range_stick_values_are_clamped_by_the_renderer():
    action = GamepadAction(left_x=5.0, left_y=-5.0)
    got = OverlayExtractor().extract(render_overlay(np.zeros((256, 256, 3), np.uint8), action))
    assert -1.0 <= got.left_x <= 1.0 and -1.0 <= got.left_y <= 1.0


def test_triggers_round_trip_within_quantization_error():
    action = GamepadAction(left_trigger=0.0, right_trigger=1.0)
    got = OverlayExtractor().extract(render_overlay(np.zeros((256, 256, 3), np.uint8), action))
    assert got.left_trigger == pytest.approx(0.0, abs=0.05)
    assert got.right_trigger == pytest.approx(1.0, abs=0.05)


def test_every_button_is_read_back_independently():
    for name in BUTTON_NAMES:
        action = GamepadAction(buttons={name: True})
        got = OverlayExtractor().extract(
            render_overlay(np.full((256, 256, 3), 30, np.uint8), action))
        pressed = [n for n in BUTTON_NAMES if got.button(n)]
        assert pressed == [name]


def test_render_overlay_does_not_mutate_the_source_frame():
    frame = np.full((128, 128, 3), 40, dtype=np.uint8)
    before = frame.copy()
    render_overlay(frame, GamepadAction(left_x=1.0))
    assert np.array_equal(frame, before)


def test_render_overlay_rejects_a_non_rgb_frame():
    with pytest.raises(ValueError):
        render_overlay(np.zeros((32, 32), dtype=np.uint8), GamepadAction())


@pytest.mark.parametrize("size", [128, 200, 256, 320])
def test_extraction_quality_holds_across_capture_resolutions(size):
    rng = np.random.default_rng(2)
    actions = [
        GamepadAction(
            left_x=float(rng.uniform(-1, 1)), left_y=float(rng.uniform(-1, 1)),
            buttons={n: bool(rng.random() < 0.3) for n in BUTTON_NAMES},
        )
        for _ in range(25)
    ]
    m = evaluate_extraction(actions, size=size)
    assert m["localization_rate"] == 1.0
    assert m["joystick_r2"] > 0.9 and m["button_accuracy"] > 0.95


def test_metrics_are_defined_when_every_frame_is_unlocalizable():
    class Blind(OverlayExtractor):
        def locate(self, frame):
            return None

    import nitrogen.data.overlay_extraction as oe
    original, oe.OverlayExtractor = oe.OverlayExtractor, Blind
    try:
        m = evaluate_extraction([GamepadAction()])
    finally:
        oe.OverlayExtractor = original
    assert m["n"] == 0 and m["localization_rate"] == 0.0 and m["joystick_r2"] == 0.0


def test_constant_ground_truth_does_not_divide_by_zero():
    # R2 is undefined when the target has no variance; a perfect reconstruction
    # must report 1.0 rather than nan.
    m = evaluate_extraction([GamepadAction() for _ in range(5)])
    assert m["joystick_r2"] == pytest.approx(1.0) and np.isfinite(m["joystick_r2"])


def test_evaluate_extraction_needs_at_least_one_action():
    with pytest.raises(ValueError):
        evaluate_extraction([])
