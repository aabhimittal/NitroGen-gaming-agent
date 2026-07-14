import torch

from nitrogen.models.flow_matching import FlowMatching


def test_interpolation_endpoints():
    fm = FlowMatching()
    x0 = torch.randn(4, 16, 8)
    x1 = torch.randn(4, 16, 8)
    t0 = torch.zeros(4, 1, 1)
    t1 = torch.ones(4, 1, 1)
    torch.testing.assert_close(fm.interpolate(x0, x1, t0), x0)
    torch.testing.assert_close(fm.interpolate(x0, x1, t1), x1)


def test_target_velocity_is_displacement():
    fm = FlowMatching()
    x0 = torch.randn(2, 4, 3)
    x1 = torch.randn(2, 4, 3)
    torch.testing.assert_close(fm.target_velocity(x0, x1), x1 - x0)


def test_sampler_recovers_constant_field():
    """If the velocity field is the *true* constant displacement, the ODE sampler
    should map the sampled noise exactly onto the data point."""
    fm = FlowMatching()
    target = torch.tensor([[[2.0, -1.0]]])  # (1, 1, 2)

    # v(x, t) = x1 - x0 = target - x0 requires knowing x0; instead test the
    # simpler invariant: integrating a field equal to (target - x)/(1 - t)
    # is unstable, so we check the constant-field case directly.
    captured = {}

    def velocity_fn(x, t):
        if "x0" not in captured:
            captured["x0"] = x.clone()
        return target - captured["x0"]

    out = fm.sample(velocity_fn, shape=(1, 1, 2), device="cpu", num_steps=50)
    torch.testing.assert_close(out, target, atol=1e-4, rtol=1e-4)


def test_loss_is_scalar_and_finite():
    fm = FlowMatching()
    x1 = torch.randn(8, 16, 18)
    loss = fm.loss(lambda x_t, t: torch.zeros_like(x_t), x1)
    assert loss.ndim == 0 and torch.isfinite(loss)
