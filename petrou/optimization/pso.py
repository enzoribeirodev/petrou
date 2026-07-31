"""
petrou.optimization.pso
========================
Particle Swarm Optimization with a pluggable inertia-weight registry.

The ``PSO`` class accepts a ``SearchSpace`` so it shares the same interface
as ``simulated_annealing``. Construction, the evaluate/track/move loop, and
inertia resolution live in ``SwarmOptimizer`` (``_pso_base.py``), shared
with every other swarm variant (e.g. ``PSOLF``); this module only adds the
classic velocity/position update rule and the built-in inertia strategies.

Built-in inertia strategies
----------------------------
Pass a float directly for constant inertia, or one of these strings:

    "random"              w = 0.5 + r/2,  r ~ Uniform(0, 1)
    "linearly decreasing" w decreases linearly from w_max to w_min
    "global-local best"   w_ij = 1.1 - g_ij / p_ij
    "chaotic descending"  decreasing trend + logistic chaos term
    "chaotic random"      random + logistic chaos term
    "psolf"                see petrou.optimization.psolf
"""

from __future__ import annotations

import numpy as np

from petrou.optimization._pso_base import InertiaRegistry, Particle, SwarmOptimizer
from petrou.optimization.search_space import SearchSpace

__all__ = ["InertiaRegistry", "PSO"]


# Register built-in strategies -------------------------------------------

@InertiaRegistry.register("random")
def _w_random(t, max_iter, p, g_pos, g_fit, rng):
    return 0.5 + rng.random() / 2.0


@InertiaRegistry.register("linearly decreasing")
def _w_linear(t, max_iter, p, g_pos, g_fit, rng, w_max=0.9, w_min=0.4):
    return w_max - ((w_max - w_min) / max_iter) * t


@InertiaRegistry.register("global-local best")
def _w_global_local(t, max_iter, p, g_pos, g_fit, rng):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p.best_position != 0, g_pos / p.best_position, 1.0)
    return 1.1 - ratio


@InertiaRegistry.register("chaotic descending")
def _w_chaotic_desc(t, max_iter, p, g_pos, g_fit, rng, w_max=0.9, w_min=0.4):
    r = rng.random()
    z = 4.0 * r * (1.0 - r)
    return (w_max - w_min) * ((max_iter - t) / max_iter) + w_min * z


@InertiaRegistry.register("chaotic random")
def _w_chaotic_rand(t, max_iter, p, g_pos, g_fit, rng):
    r1, r2 = rng.random(), rng.random()
    return 0.5 * r1 + 0.5 * (4.0 * r2 * (1.0 - r2))


# ---------------------------------------------------------------------------
# PSO
# ---------------------------------------------------------------------------

class PSO(SwarmOptimizer):
    """
    Particle Swarm Optimization.

    Parameters
    ----------
    objective_fn : Callable[[np.ndarray], float]
    num_particles : int
    search_space : SearchSpace
        Preferred initialisation — provides per-dimension bounds.
    mode : {"max", "min"}
        Default ``"max"`` — consistent with all other petrou optimizers.
    v_max : float or None
        Maximum velocity per dimension. Computed from ``k`` if not given.
    k : float
        ``v_max = k * (upper - lower) / 2`` when ``v_max`` is None.
    c1, c2 : float
        Cognitive and social acceleration coefficients.
    seed : int or None

    Examples
    --------
    >>> from petrou.optimization import SearchSpace, PSO
    >>> space = SearchSpace([{"name": "x", "type": "float", "bounds": (-5.0, 5.0), "step": 0.1}])
    >>> pso = PSO(lambda s: -(s[0]**2), 20, search_space=space, mode="max", seed=0)
    >>> pos, fit = pso.optimize(100)
    >>> space.decode(pos)
    {"x": 0.0}
    """

    default_inertia_strategy = 0.5

    def __init__(
        self,
        objective_fn,
        num_particles: int,
        *,
        search_space: SearchSpace,
        mode: str = "max",
        v_max: float | np.ndarray | None = None,
        k: float = 0.5,
        c1: float = 2.0,
        c2: float = 2.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(objective_fn, num_particles, search_space=search_space, mode=mode, seed=seed)
        self.c1, self.c2 = c1, c2
        self.v_max = v_max if v_max is not None else k * (self.upper - self.lower) / 2.0

    def _apply_update(self, p: Particle, w: float) -> None:
        r1, r2 = self.rng.random(p.dimensions), self.rng.random(p.dimensions)
        p.velocity = (
            w * p.velocity
            + self.c1 * r1 * (p.best_position - p.position)
            + self.c2 * r2 * (self.global_best_position - p.position)
        )
        p.velocity = np.clip(p.velocity, -self.v_max, self.v_max)
        p.position = p.position + p.velocity
