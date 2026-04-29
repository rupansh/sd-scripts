"""Unit tests for the EGGROLL optimizer.

These tests intentionally avoid importing `library.train_util` so they can run
without the heavy training-stack deps (cv2, transformers, etc.). The
`get_optimizer` factory wiring is exercised in `test_eggroll_factory.py`.
"""

import pytest
import torch
from torch.nn import Parameter

from library.eggroll_optimizer import EGGROLL, rank_shape


# ---------------------------------------------------------------- shape utils


def test_rank_shape_bounds_and_centered():
    f = torch.tensor([10.0, -3.0, 5.0, 0.0, 7.0])
    shaped = rank_shape(f)
    # All values lie in [-0.5, 0.5]
    assert shaped.min().item() == pytest.approx(-0.5)
    assert shaped.max().item() == pytest.approx(0.5)
    # Sum is approximately zero (ranks are symmetric)
    assert abs(shaped.sum().item()) < 1e-5
    # Order is preserved
    order = f.argsort()
    assert torch.equal(shaped[order], shaped.sort().values)


def test_rank_shape_singleton():
    assert rank_shape(torch.tensor([1.0])).item() == 0.0


# -------------------------------------------------------------- optimizer API


def _make_optimizer(params, **overrides):
    kwargs = dict(lr=0.05, sigma=0.02, population_size=8, seed=123)
    kwargs.update(overrides)
    return EGGROLL(params, **kwargs)


def test_step_requires_closure():
    p = Parameter(torch.zeros(4, 4))
    opt = _make_optimizer([p])
    with pytest.raises(RuntimeError, match="closure"):
        opt.step()


def test_step_modifies_params():
    torch.manual_seed(0)
    p = Parameter(torch.zeros(8, 8))
    opt = _make_optimizer([p])

    # closure: prefer parameters near (1,1,...): minimize MSE distance to ones.
    target = torch.ones(8, 8)

    def closure():
        return ((p - target) ** 2).mean()

    before = p.detach().clone()
    opt.step(closure)
    assert not torch.allclose(p.detach(), before), "step did not change params"


def test_population_size_must_be_even():
    p = Parameter(torch.zeros(4, 4))
    with pytest.raises(ValueError, match="even"):
        EGGROLL([p], lr=0.05, sigma=0.02, population_size=7)


def test_handles_1d_and_2d_and_4d():
    """Mix DoRA-magnitude-style 1D + LoRA 2D + conv 4D parameters."""
    torch.manual_seed(0)
    mag = Parameter(torch.zeros(8))           # 1D (DoRA magnitude)
    w2d = Parameter(torch.zeros(8, 16))       # 2D (LoRA up/down)
    w4d = Parameter(torch.zeros(8, 16, 3, 3)) # 4D (conv)
    targets = (torch.full_like(mag, 0.5), torch.full_like(w2d, 0.5), torch.full_like(w4d, 0.5))

    opt = _make_optimizer([mag, w2d, w4d], lr=0.5, sigma=0.05)

    def closure():
        return ((mag - targets[0]) ** 2).mean() + ((w2d - targets[1]) ** 2).mean() + ((w4d - targets[2]) ** 2).mean()

    initial = closure().item()
    for _ in range(10):
        opt.step(closure)
    final = closure().item()
    assert final < initial, f"loss did not decrease: {initial} -> {final}"


def test_clip_norm_is_respected():
    """A wild fitness landscape should still produce a bounded delta_W."""
    torch.manual_seed(7)
    p = Parameter(torch.zeros(16, 32))

    # Use lr=1 and clip_norm=0.1 so that the post-step displacement is the
    # delta_W itself; we verify its Frobenius norm is bounded by clip_norm.
    opt = EGGROLL([p], lr=1.0, sigma=0.02, population_size=16, clip_norm=0.1, seed=0)

    # Closure with huge dynamic range — fitness shaping caps it but we double
    # check clip_norm anyway.
    def closure():
        return p.sum() * 1e6  # arbitrary; rank-shaping normalises

    opt.step(closure)
    assert p.detach().norm().item() <= 0.1 + 1e-5


def test_convex_convergence():
    """EGGROLL should reduce a quadratic loss substantially after many steps."""
    torch.manual_seed(42)
    p = Parameter(torch.zeros(8, 8))
    target = torch.full_like(p, 0.3)
    opt = _make_optimizer([p], lr=0.5, sigma=0.05, population_size=32)

    def closure():
        return ((p - target) ** 2).mean()

    initial = closure().item()
    for _ in range(40):
        opt.step(closure)
    final = closure().item()
    # Big improvement; not asking for arbitrary precision.
    assert final < initial * 0.5, f"failed to converge: {initial} -> {final}"


def test_deterministic_with_seed():
    torch.manual_seed(0)
    p1 = Parameter(torch.zeros(6, 6))
    p2 = Parameter(torch.zeros(6, 6))
    opt1 = EGGROLL([p1], lr=0.05, sigma=0.02, population_size=8, seed=42)
    opt2 = EGGROLL([p2], lr=0.05, sigma=0.02, population_size=8, seed=42)

    def make_closure(p):
        def closure():
            return (p ** 2).mean()
        return closure

    opt1.step(make_closure(p1))
    opt2.step(make_closure(p2))
    assert torch.allclose(p1.detach(), p2.detach(), atol=1e-7)


def test_dora_like_setup_decreases_loss():
    """Mimic a DoRA layer: magnitude vector m + direction matrix V; target is
    a known (m*, V*) pair. EGGROLL should pull both toward the optimum."""
    torch.manual_seed(1)
    m = Parameter(torch.ones(4))
    V = Parameter(torch.zeros(4, 8))
    m_star = torch.full_like(m, 1.5)
    V_star = torch.full_like(V, 0.25)

    opt = _make_optimizer([m, V], lr=0.5, sigma=0.05, population_size=32)

    def closure():
        return ((m - m_star) ** 2).mean() + ((V - V_star) ** 2).mean()

    losses = [closure().item()]
    for _ in range(30):
        opt.step(closure)
        losses.append(closure().item())
    assert losses[-1] < losses[0] * 0.5
