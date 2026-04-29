"""
EGGROLL — Evolution Guided GeneRal Optimisation via Low-rank Learning.

Gradient-free Evolution Strategies optimizer that perturbs each weight matrix
with rank-r outer products (E = a ⊗ b^T) and uses antithetic sampling. The
update is

    W <- W + lr * (1/M) * Σ_j Δf_j * (a_j ⊗ b_j^T)

with Δf_j = (f(W + σ E_j) - f(W - σ E_j)) / 2 after rank-shaping the fitness.

This optimizer requires a closure passed to step(); it does NOT use .grad.
The closure must (re)compute the training loss given the current parameter
values without calling .backward(). The optimizer is responsible for swapping
parameter values to evaluate fitness for each perturbation.

Reference:
    paper:  https://arxiv.org/abs/2511.16652
    repo:   https://github.com/sigridjineth/eggroll-embedding-trainer
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.optim.optimizer import Optimizer


def _flatten_2d(t: torch.Tensor) -> Tuple[torch.Tensor, Sequence[int]]:
    """Flatten an ND tensor to 2D [first_dim, prod(rest)] and remember shape."""
    if t.dim() == 0:
        return t.view(1, 1), t.shape
    if t.dim() == 1:
        return t.view(t.shape[0], 1), t.shape
    if t.dim() == 2:
        return t, t.shape
    return t.reshape(t.shape[0], -1), t.shape


def _unflatten(flat: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    if len(shape) == 0:
        return flat.view(())
    if len(shape) == 1:
        return flat.view(shape[0])
    return flat.reshape(*shape)


def rank_shape(values: torch.Tensor) -> torch.Tensor:
    """Map a 1D fitness vector to ranks in [-0.5, 0.5]."""
    n = values.shape[0]
    if n <= 1:
        return torch.zeros_like(values)
    _, idx = values.sort()
    ranks = torch.empty_like(values)
    ranks[idx] = torch.arange(n, dtype=values.dtype, device=values.device)
    return ranks / (n - 1) - 0.5


class EGGROLL(Optimizer):
    """
    EGGROLL Evolution Strategies optimizer.

    Args:
        params: iterable of trainable parameters or param groups.
        lr: learning rate applied to the rank-1 update.
        sigma: noise std for perturbations.
        population_size: total number of fitness evaluations per step. Must be
            even — half are positive antithetic, half are negative.
        rank: rank of the perturbation per matrix (1 in the original paper).
            Implemented as a sum of `rank` independent rank-1 outer products.
        clip_norm: max Frobenius norm of the per-parameter delta_W.
        weight_decay: decoupled weight decay applied after the ES step.
        momentum: momentum on the per-parameter delta_W (0 disables).
        fitness_shaping: "rank" (centered ranks in [-0.5, 0.5]) or "zscore".
        adaptive_sigma: enable dynamic σ adjustment based on fitness variance.
        sigma_min, sigma_max, sigma_target_var, sigma_adapt_rate: σ adaptation
            controls (used only when adaptive_sigma=True).
        seed: base seed for the perturbation RNG; per-step seeds derive from it.
        antithetic: use mirror sampling (always True in the paper).

    Notes:
        * Works with any tensor shape. 0D/1D parameters use vector noise (no
          outer product). 2D uses rank-1 outer products. ND≥3 (e.g. conv
          weights [out, in, kh, kw]) is reshaped to 2D [out, prod(rest)].
        * The optimizer is gradient-free: .grad is ignored, and an exception
          is raised if step() is called without a closure.
    """

    is_eggroll_optimizer = True  # used by sd-scripts to route the training loop

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.05,
        sigma: float = 0.02,
        population_size: int = 16,
        rank: int = 1,
        clip_norm: float = 1.0,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        fitness_shaping: str = "rank",
        adaptive_sigma: bool = False,
        sigma_min: float = 0.005,
        sigma_max: float = 0.1,
        sigma_target_var: float = 0.1,
        sigma_adapt_rate: float = 0.01,
        seed: int = 0,
        antithetic: bool = True,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if sigma <= 0.0:
            raise ValueError(f"Invalid sigma: {sigma}")
        if population_size < 2 or population_size % 2 != 0:
            raise ValueError(
                f"population_size must be even and >= 2, got {population_size}"
            )
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if fitness_shaping not in ("rank", "zscore"):
            raise ValueError(f"Unknown fitness_shaping: {fitness_shaping}")
        if not antithetic:
            raise NotImplementedError("antithetic=False is not supported")

        defaults = dict(
            lr=lr,
            sigma=sigma,
            population_size=population_size,
            rank=rank,
            clip_norm=clip_norm,
            weight_decay=weight_decay,
            momentum=momentum,
            fitness_shaping=fitness_shaping,
            adaptive_sigma=adaptive_sigma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_target_var=sigma_target_var,
            sigma_adapt_rate=sigma_adapt_rate,
        )
        super().__init__(params, defaults)
        self._step_counter = 0
        self._base_seed = int(seed)
        # current σ per-group (mutated when adaptive_sigma=True)
        for group in self.param_groups:
            group.setdefault("_current_sigma", group["sigma"])

    # ------------------------------------------------------------------ noise
    def _generator_for(self, p_index: int, step: int, device: torch.device) -> torch.Generator:
        # Mix the base seed with the step and the parameter index to keep
        # samples deterministic and uncorrelated across parameters.
        gen = torch.Generator(device=device)
        # Use a 64-bit hash that won't overflow torch's int64 seed.
        s = (self._base_seed * 0x9E3779B97F4A7C15) ^ (step * 0x100000001B3) ^ (p_index * 0xC2B2AE3D27D4EB4F)
        gen.manual_seed(s & 0x7FFFFFFFFFFFFFFF)
        return gen

    def _sample_perturbations(
        self, p: torch.Tensor, M: int, rank: int, p_index: int, step: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Sequence[int]]:
        """
        Sample M rank-`rank` perturbations for parameter `p`.

        Returns (A, B, original_shape) where:
            - For 2D-flattenable params: A is [M, rank, dim_out],
              B is [M, rank, dim_in]; perturbation j is sum_k A[j,k] ⊗ B[j,k].
            - For 1D/0D params: A is [M, numel] (vector noise), B is None.
        """
        gen = self._generator_for(p_index, step, p.device)
        flat, shape = _flatten_2d(p.detach())
        d_out, d_in = flat.shape

        if p.dim() <= 1:
            # vector / scalar noise — no outer product
            A = torch.randn(M, p.numel(), generator=gen, device=p.device, dtype=p.dtype)
            return A, None, shape

        A = torch.randn(M, rank, d_out, generator=gen, device=p.device, dtype=p.dtype)
        B = torch.randn(M, rank, d_in, generator=gen, device=p.device, dtype=p.dtype)
        return A, B, shape

    @staticmethod
    def _materialize(A: torch.Tensor, B: Optional[torch.Tensor], j: int, shape: Sequence[int]) -> torch.Tensor:
        """Construct the j-th perturbation tensor of shape `shape`."""
        if B is None:
            return _unflatten(A[j], shape)
        # A[j]: [rank, d_out], B[j]: [rank, d_in]  ->  sum_k outer(A_k, B_k)
        E_flat = A[j].transpose(0, 1) @ B[j]  # [d_out, d_in]
        return _unflatten(E_flat, shape)

    @staticmethod
    def _aggregate_update(
        A: torch.Tensor,
        B: Optional[torch.Tensor],
        shaped_fitness: torch.Tensor,
        shape: Sequence[int],
    ) -> torch.Tensor:
        """Compute (1/M) Σ_j f_j * E_j efficiently and return tensor of `shape`."""
        M = A.shape[0]
        if B is None:
            # vector noise
            update = (shaped_fitness.unsqueeze(-1) * A).sum(dim=0) / M
            return _unflatten(update, shape)

        # A: [M, rank, d_out], B: [M, rank, d_in], f: [M]
        # delta_W = (1/M) Σ_j f_j * Σ_k A[j,k] ⊗ B[j,k]
        weighted_A = A * shaped_fitness.view(M, 1, 1)
        # reshape and matmul: collapse (M,rank) into one axis
        d_out = A.shape[-1]
        d_in = B.shape[-1]
        wA = weighted_A.reshape(M * A.shape[1], d_out)
        Bf = B.reshape(M * B.shape[1], d_in)
        delta_flat = wA.t() @ Bf / M
        return _unflatten(delta_flat, shape)

    # --------------------------------------------------------------- step API
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        """
        Perform one EGGROLL step.

        Args:
            closure: a callable that re-runs the forward pass under the CURRENT
                parameter values and returns a scalar loss tensor (smaller is
                better — we minimize). It must NOT call .backward().

        Returns:
            The scalar loss evaluated at the post-update parameters
            (approximate — we return the average antithetic loss seen during
            the step rather than re-running the closure to save compute).
        """
        if closure is None:
            raise RuntimeError(
                "EGGROLL.step() requires a closure that recomputes the training "
                "loss; call optimizer.step(closure=...). EGGROLL is gradient-free."
            )

        step = self._step_counter
        self._step_counter += 1

        # ---- collect every trainable parameter, grouped, with stable index
        flat_params: List[Tuple[int, dict, torch.Tensor]] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    flat_params.append((len(flat_params), group, p))

        if not flat_params:
            return torch.tensor(float("nan"))

        # ---- save originals so we can perturb in-place and restore
        originals = [p.data.clone() for _, _, p in flat_params]

        # ---- decide M (population/2). All param groups must agree on pop size.
        pop_sizes = {g["population_size"] for _, g, _ in flat_params}
        assert len(pop_sizes) == 1, "all param groups must share population_size"
        pop_size = pop_sizes.pop()
        M = pop_size // 2

        # ---- sample noise for every parameter once
        rank_per_group = {id(g): g["rank"] for _, g, _ in flat_params}
        sigma_per_group = {id(g): g["_current_sigma"] for _, g, _ in flat_params}
        samples: List[Tuple[torch.Tensor, Optional[torch.Tensor], Sequence[int]]] = []
        for idx, group, p in flat_params:
            samples.append(
                self._sample_perturbations(p, M, rank_per_group[id(group)], idx, step)
            )

        # ---- evaluate fitness for each antithetic pair
        fitness_pos = torch.empty(M)
        fitness_neg = torch.empty(M)
        for j in range(M):
            # positive perturbation
            for (idx, group, p), (A, B, shape), orig in zip(flat_params, samples, originals):
                sigma = sigma_per_group[id(group)]
                E = self._materialize(A, B, j, shape)
                p.data.copy_(orig + sigma * E)
            loss = closure()
            fitness_pos[j] = -float(loss.detach())  # negate: we want to maximize fitness

            # negative (mirror) perturbation
            for (idx, group, p), (A, B, shape), orig in zip(flat_params, samples, originals):
                sigma = sigma_per_group[id(group)]
                E = self._materialize(A, B, j, shape)
                p.data.copy_(orig - sigma * E)
            loss = closure()
            fitness_neg[j] = -float(loss.detach())

        # ---- restore originals before we apply the aggregated update
        for (_, _, p), orig in zip(flat_params, originals):
            p.data.copy_(orig)

        # ---- compute antithetic delta and shape it
        delta_f = (fitness_pos - fitness_neg) / 2.0  # [M]
        # use the first group's setting; all groups share fitness_shaping in practice
        shaping = flat_params[0][1]["fitness_shaping"]
        if shaping == "rank":
            shaped = rank_shape(delta_f)
        else:  # zscore
            std = delta_f.std().clamp(min=1e-8)
            shaped = (delta_f - delta_f.mean()) / std

        # ---- apply the per-parameter update
        avg_loss = -float((fitness_pos.mean() + fitness_neg.mean()) / 2.0)
        for (idx, group, p), (A, B, shape) in zip(flat_params, samples):
            shaped_dev = shaped.to(device=p.device, dtype=p.dtype)
            delta_W = self._aggregate_update(A, B, shaped_dev, shape)

            # clip
            clip = group["clip_norm"]
            if clip is not None and clip > 0:
                norm = delta_W.norm()
                if norm > clip:
                    delta_W.mul_(clip / norm)

            # momentum
            if group["momentum"] > 0:
                state = self.state[p]
                buf = state.get("velocity")
                if buf is None:
                    buf = torch.zeros_like(p.data)
                    state["velocity"] = buf
                buf.mul_(group["momentum"]).add_(delta_W)
                delta_W = buf

            # parameter update
            p.data.add_(delta_W, alpha=group["lr"])

            # decoupled weight decay
            if group["weight_decay"] > 0:
                p.data.mul_(1.0 - group["lr"] * group["weight_decay"])

        # ---- adaptive sigma
        for group in self.param_groups:
            if not group["adaptive_sigma"]:
                continue
            var = float((fitness_pos - fitness_neg).var())
            target = group["sigma_target_var"]
            rate = group["sigma_adapt_rate"]
            cur = group["_current_sigma"]
            if var < target * 0.5:
                cur = min(cur * (1.0 + rate), group["sigma_max"])
            elif var > target * 2.0:
                cur = max(cur * (1.0 - rate), group["sigma_min"])
            group["_current_sigma"] = cur

        return torch.tensor(avg_loss)


__all__ = ["EGGROLL", "rank_shape"]
