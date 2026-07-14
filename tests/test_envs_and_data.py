import numpy as np

from nitrogen.action_space import BUTTON_NAMES, GamepadAction
from nitrogen.data.dataset import FrameActionChunkDataset
from nitrogen.data.overlay_extraction import evaluate_extraction
from nitrogen.data.synthetic import collect_dataset
from nitrogen.envs.toy_game import GAME_REGISTRY, make_game


def test_all_games_render_and_step():
    for name in GAME_REGISTRY:
        env = make_game(name)
        frame = env.reset(seed=0)
        assert frame.shape == (256, 256, 3) and frame.dtype == np.uint8
        action = env.expert_action()
        frame, done, info = env.step(action)
        assert "success" in info


def test_experts_are_competent():
    # Scripted experts should solve their own games most of the time.
    for name in GAME_REGISTRY:
        wins = 0
        for s in range(10):
            env = make_game(name)
            env.reset(seed=s)
            done, info = False, {}
            while not done:
                _, done, info = env.step(env.expert_action())
            wins += info["success"]
        assert wins >= 8, f"{name} expert only won {wins}/10"


def test_dataset_chunking():
    eps = collect_dataset(["reacher"], episodes_per_game=2, noise=0.1)
    ds = FrameActionChunkDataset(eps, chunk_size=16)
    frame, chunk = ds[0]
    assert frame.shape == (3, 256, 256)
    assert chunk.shape == (16, 18)
    assert float(chunk.min()) >= -1.0 and float(chunk.max()) <= 1.0


def test_overlay_extraction_quality():
    rng = np.random.default_rng(1)
    actions = [
        GamepadAction(
            left_x=float(rng.uniform(-1, 1)), left_y=float(rng.uniform(-1, 1)),
            right_x=float(rng.uniform(-1, 1)), right_y=float(rng.uniform(-1, 1)),
            buttons={n: bool(rng.random() < 0.3) for n in BUTTON_NAMES},
        )
        for _ in range(80)
    ]
    metrics = evaluate_extraction(actions)
    assert metrics["joystick_r2"] > 0.9
    assert metrics["button_accuracy"] > 0.95
