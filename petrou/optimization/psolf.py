"""
petrou.optimization.psolf
==========================
PSOLF — Particle Swarm Optimization with a Levy-flight velocity update
(Jensi & Wiselin Jiji, 2016).

Shares its particle state, construction, and evaluate/track/move loop with
``PSO`` via ``SwarmOptimizer`` (``_pso_base.py``). PSOLF's own contribution
is the update rule: each iteration, every particle's move is either the
classic PSO update or a Levy-flight update, chosen by a coin flip
(``levy_probability``, default 0.5). Levy step lengths are drawn per
particle via Mantegna's algorithm with a fixed stability index ``beta``
(default 1.5, per Jensi).
"""

from __future__ import annotations

import math

import numpy as np

from petrou.optimization._pso_base import InertiaRegistry, Particle, SwarmOptimizer
from petrou.optimization.search_space import SearchSpace

__all__ = ["PSOLF"]


@InertiaRegistry.register("psolf")
def _w_psolf(t, max_iter, p, g_pos, g_fit, rng, w_max=0.9, w_min=0.1):
    """Jensi's inertia schedule — linear decrease from ``w_max`` to ``w_min``."""
    return w_min + (w_max - w_min) * (1.0 - t / max_iter)


def _mantegna_stepsize(beta: float, rng: np.random.Generator) -> float:
    """One scalar Levy step length via Mantegna's algorithm (Jensi)."""
    num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0)
    sigma_u = (num / den) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u)
    v = rng.normal(0.0, 1.0)
    S = u / (abs(v) ** (1.0 / beta))
    return 0.01 * S


class PSOLF(SwarmOptimizer):
    """
    PSOLF — Particle Swarm Optimization with Levy Flight (Jensi, 2016).

    Parameters
    ----------
    objective_fn : Callable[[np.ndarray], float]
    num_particles : int
        Jensi's default: 25.
    search_space : SearchSpace
    mode : {"max", "min"}
        Default ``"max"`` — consistent with all other petrou optimizers.
    c1, c2 : float
        Cognitive and social acceleration coefficients. Jensi: ``1.2``, ``1.8``.
    beta : float
        Levy stability index for Mantegna's algorithm, ``0 < beta <= 2``. Jensi: ``1.5``.
    levy_probability : float
        Per-particle, per-iteration chance of taking the Levy-flight update
        instead of the classic PSO update. Jensi: ``0.5``.
    seed : int or None

    Notes
    -----
    No ``v_max`` velocity clamp — Jensi only clips the *position* back into
    bounds after each move, so that's all this class does.

    Jensi's reported setup is ``NP=25``, ``Max FEs=125000``, i.e.
    ``max_iterations = 125000 / 25 = 5000``.

    Examples
    --------
    >>> from petrou.optimization import SearchSpace, PSOLF
    >>> space = SearchSpace([{"name": "x", "type": "float", "bounds": (-5.0, 5.0), "step": 0.1}])
    >>> psolf = PSOLF(lambda s: -(s[0]**2), 25, search_space=space, mode="max", seed=0)
    >>> pos, fit = psolf.optimize(200)
    >>> space.decode(pos)
    {"x": 0.0}
    """

    default_inertia_strategy = "psolf"

    def __init__(
        self,
        objective_fn,
        num_particles: int = 25,
        *,
        search_space: SearchSpace,
        mode: str = "max",
        c1: float = 1.2,
        c2: float = 1.8,
        beta: float = 1.5,
        levy_probability: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(objective_fn, num_particles, search_space=search_space, mode=mode, seed=seed)
        self.c1, self.c2 = c1, c2
        self.beta = beta
        self.levy_probability = levy_probability

    def _apply_update(self, p: Particle, w: float) -> None:
        if self.rng.random() < self.levy_probability:
            self._levy_move(p, w)
        else:
            self._standard_move(p, w)

    def _standard_move(self, p: Particle, w: float) -> None:
        r1, r2 = self.rng.random(p.dimensions), self.rng.random(p.dimensions)
        p.velocity = (
            w * p.velocity
            + self.c1 * r1 * (p.best_position - p.position)
            + self.c2 * r2 * (self.global_best_position - p.position)
        )
        p.position = p.position + p.velocity

    def _levy_move(self, p: Particle, w: float) -> None:
        step = _mantegna_stepsize(self.beta, self.rng) * p.position
        levy_walk = p.position + step * self.rng.random(p.dimensions)

        r1, r2 = self.rng.random(p.dimensions), self.rng.random(p.dimensions)
        p.velocity = (
            w * levy_walk
            + self.c1 * r1 * (p.best_position - p.position)
            + self.c2 * r2 * (self.global_best_position - p.position)
        )
        p.position = p.velocity.copy()
