# 2 · The standardized gamepad action space

**Code:** [`nitrogen/action_space.py`](../nitrogen/action_space.py)

## The problem it solves

Every game has a different control scheme. Some use keyboard + mouse, some a
gamepad, some touch. Even among gamepad games, "the A button" does different
things and analog sticks are mapped differently. If your policy's output space
were game-specific, you could never train one model across many games, and you
could never transfer to a new one — the output head wouldn't even have the right
shape.

NitroGen's fix is to define **one universal action interface** and make every
game speak it: a **virtual Xbox-style gamepad**. The model always emits the same
kind of thing — stick positions, trigger values, button states — regardless of
which game is on screen. This is exactly analogous to how a language model always
emits tokens from one vocabulary no matter the task.

## The interface

```
2 analog sticks   : left (x, y), right (x, y)   ∈ [-1, 1]
2 analog triggers : left, right                 ∈ [ 0, 1]
12 digital buttons: A B X Y LB RB, d-pad ×4, start, back  ∈ {0, 1}
────────────────────────────────────────────────────────────────
ACTION_DIM = 4 + 2 + 12 = 18
```

`GamepadAction` is the human-readable form used by environments and the
overlay-extraction pipeline:

```python
from nitrogen.action_space import GamepadAction
a = GamepadAction(left_x=0.8, left_y=-0.2, buttons={"A": True})
```

## Packing into a continuous vector for the generative head

The action head generates actions with **flow matching**, a *continuous*
diffusion-style process ([ch 5](05_flow_matching.md)). It needs a real-valued
target vector, but our controls are a mix of continuous and *discrete* channels.
The trick is to embed everything into one continuous cube `[-1, 1]^18`:

| channel | raw range | normalized range | mapping |
|---------|-----------|------------------|---------|
| sticks | `[-1, 1]` | `[-1, 1]` | identity |
| triggers | `[0, 1]` | `[-1, 1]` | `2x − 1` |
| buttons | `{0, 1}` | `{−1, +1}` | `2x − 1` |

```python
from nitrogen.action_space import encode_action, decode_action
norm = encode_action(a.to_vector())   # -> model space, all channels in [-1, 1]
raw  = decode_action(norm)            # buttons thresholded back to {0, 1}
```

Flow matching then models the *whole 18-D vector* as continuous noise→data
transport. At decode time the button channels are thresholded at 0 (the image of
0.5 under the encoding) to recover crisp presses. This "**buttons as soft
continuous values that get rounded**" trick is what lets a single generative head
emit both smooth analog control *and* discrete button presses — no separate
categorical output head required.

## Why standardization enables transfer

Because the action space is identical across games, the *skill* the model learns
— "when the objective is up-and-to-the-right, push the left stick up-and-to-the-
right" — is expressed in the same coordinates in every game. A policy trained on
Reacher and Dodger can be dropped into an unseen Chaser game and its learned
stick behavior still means the same thing. That is precisely the cross-game
transfer we measure in [ch 10](10_transfer.md).

Continue to [**3 · Building a video-action dataset from overlays →**](03_data_pipeline.md)
