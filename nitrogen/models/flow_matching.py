"""Flow matching — the generative engine behind NitroGen's action head.

NitroGen does **not** regress actions directly (a single MLP that outputs one
action). Instead it *generates* a chunk of future actions with a diffusion-style
process. The specific formulation is **(rectified) flow matching**, which is
simpler and faster to sample than classic score-based diffusion.

The idea in one paragraph
-------------------------
Pick a straight-line path between pure Gaussian noise ``x0 ~ N(0, I)`` and a real
action chunk ``x1`` (the expert's actions):

    x_t = (1 - t) * x0 + t * x1,      t in [0, 1]

Differentiate w.r.t. ``t`` and the "velocity" that moves a point along this path
is *constant*:

    dx_t/dt = x1 - x0                 (the flow-matching target)

We train a network ``v_theta(x_t, t, condition)`` to predict that velocity from a
*noised* action chunk, the flow time ``t``, and the image conditioning. At
inference we start from noise ``x0`` and integrate the learned ODE forward from
``t=0`` to ``t=1`` — the endpoint is a freshly generated action chunk.

Why this instead of plain regression?
    * Multi-modality: at a given frame several action chunks may be valid
      (dodge left *or* right). An MSE regressor averages them into a mushy,
      often invalid action; a generative model can commit to one mode.
    * Temporal coherence: generating the *whole chunk* jointly (16 steps) keeps
      the actions self-consistent instead of independently noisy.

This module is deliberately model-agnostic: it knows nothing about images or
gamepads. It just provides the path, the training target, and the ODE sampler.
"""

from __future__ import annotations

from typing import Callable

import torch


class FlowMatching:
    """Rectified-flow paths, training target, and an Euler ODE sampler.

    Convention: ``t = 0`` is noise, ``t = 1`` is data.
    """

    def __init__(self, sigma_min: float = 0.0):
        # sigma_min > 0 gives the "conditional OT" path a little noise floor at
        # t=1; 0 recovers the plain straight-line (rectified flow) path.
        self.sigma_min = sigma_min

    # -- training --------------------------------------------------------
    def sample_time(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        """Sample flow times ``t ~ U(0, 1)`` shaped ``(batch, 1, 1)`` for broadcasting."""
        return torch.rand(batch, 1, 1, device=device, dtype=dtype)

    def interpolate(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Point on the noise->data path: ``x_t = (1 - (1 - sigma_min) t) x0 + t x1``."""
        return (1.0 - (1.0 - self.sigma_min) * t) * x0 + t * x1

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """The constant velocity field for the path above: ``x1 - (1 - sigma_min) x0``."""
        return x1 - (1.0 - self.sigma_min) * x0

    def loss(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x1: torch.Tensor,
        noise: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Conditional flow-matching loss for a batch of data samples ``x1``.

        ``velocity_fn(x_t, t)`` must return the network's velocity prediction
        (the image/condition is expected to be closed over by the caller). The
        loss is a plain MSE between predicted and target velocity.
        """
        if noise is None:
            noise = torch.randn_like(x1)
        if t is None:
            t = self.sample_time(x1.shape[0], x1.device, x1.dtype)
        x_t = self.interpolate(noise, x1, t)
        target = self.target_velocity(noise, x1)
        pred = velocity_fn(x_t, t)
        return torch.mean((pred - target) ** 2)

    # -- sampling --------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        device,
        num_steps: int = 10,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """Integrate the ODE ``dx/dt = v_theta(x, t)`` from noise (t=0) to data (t=1).

        Uses a fixed-step forward Euler integrator. NitroGen samples action chunks
        with only a handful of steps (flow matching is cheap to sample) — the
        default of 10 is plenty for the toy tasks here.
        """
        x = torch.randn(*shape, device=device, dtype=dtype)  # x0 ~ N(0, I)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_scalar = i * dt
            t = torch.full((shape[0], 1, 1), t_scalar, device=device, dtype=dtype)
            v = velocity_fn(x, t)
            x = x + dt * v
        return x
