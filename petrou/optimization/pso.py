"""
petrou.optimization.pso
========================
Particle Swarm Optimization with a pluggable inertia-weight registry.

The ``PSO`` class accepts a ``SearchSpace`` so it shares the same interface
as ``simulated_annealing``.  Inertia strategies are registered with the
``InertiaRegistry`` decorator and can be added at runtime without touching
library code.

Built-in inertia strategies
----------------------------
Pass a float directly for constant inertia, or one of these strings:

    "random"              w = 0.5 + r/2,  r ~ Uniform(0, 1)
    "linearly decreasing" w decreases linearly from w_max to w_min
    "global-local best"   w_ij = 1.1 - g_ij / p_ij
    "chaotic descending"  decreasing trend + logistic chaos term
    "chaotic random"      random + logistic chaos term
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from petrou.optimization.search_space import SearchSpace

__all__ = ["InertiaRegistry", "PSO"]

# ---------------------------------------------------------------------------
# Inertia registry
# ---------------------------------------------------------------------------

class InertiaRegistry:
    """
    Registry of named inertia-weight strategies for PSO.

    Strategies are plain callables with the signature::

        fn(t, max_iter, particle, g_pos, g_fit, rng) -> float | np.ndarray

    where:
        t         int                current iteration (0-indexed)
        max_iter  int                total iterations planned
        particle  _Particle          current particle object
        g_pos     np.ndarray         global best position so far
        g_fit     float              global best fitness so far
        rng       np.random.Generator

    Register a new strategy with the decorator::

        @InertiaRegistry.register("my strategy")
        def my_fn(t, max_iter, particle, g_pos, g_fit, rng):
            return 0.7
    """

    _registry: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator — register ``fn`` under ``name``.

        Parameters
        ----------
        name : str
            Case-insensitive strategy name.

        Examples
        --------
        >>> @InertiaRegistry.register("sigmoid")
        ... def sigmoid_w(t, max_iter, particle, g_pos, g_fit, rng):
        ...     x = 10 * (t / max_iter - 0.5)
        ...     return 1.0 / (1.0 + np.exp(x))
        """
        def decorator(fn: Callable) -> Callable:
            cls._registry[name.strip().lower()] = fn
            return fn
        return decorator

    @classmethod
    def list_strategies(cls) -> list[str]:
        """Return the names of all registered strategies."""
        return sorted(cls._registry)

    @classmethod
    def get(cls, strategy: float | int | str) -> Callable:
        """
        Resolve a strategy to a callable.

        Parameters
        ----------
        strategy : float, int, or str
            A numeric value returns a constant-inertia function.
            A string looks up the registry (case-insensitive).

        Raises
        ------
        ValueError
            Unknown strategy name.
        TypeError
            Strategy is neither a number nor a string.
        """
        if isinstance(strategy, (int, float)):
            w = float(strategy)
            return lambda t, max_iter, p, g_pos, g_fit, rng: w
        if isinstance(strategy, str):
            key = strategy.strip().lower()
            if key not in cls._registry:
                raise ValueError(
                    f"Unknown inertia strategy '{strategy}'. "
                    f"Available: {cls.list_strategies()}."
                )
            return cls._registry[key]
        raise TypeError(
            f"strategy must be a number or str, got {type(strategy).__name__}."
        )


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
# Particle (internal)
# ---------------------------------------------------------------------------

class _Particle:
    __slots__ = ("position", "velocity", "best_position", "best_fitness", "dimensions")

    def __init__(self, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator, mode: str) -> None:
        self.dimensions = len(lower)
        self.position = rng.uniform(lower, upper)
        self.velocity = np.zeros(self.dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = np.inf if mode == "min" else -np.inf

    def update_velocity(self, g_pos, rng, w, c1, c2, v_max) -> None:
        r1, r2 = rng.random(self.dimensions), rng.random(self.dimensions)
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.best_position - self.position)
            + c2 * r2 * (g_pos - self.position)
        )
        if v_max is not None:
            self.velocity = np.clip(self.velocity, -v_max, v_max)

    def update_position(self, lower, upper) -> None:
        self.position = np.clip(self.position + self.velocity, lower, upper)


# ---------------------------------------------------------------------------
# PSO
# ---------------------------------------------------------------------------

class PSO:
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

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
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
        self._fn = objective_fn
        self.mode = mode.lower()
        self.c1, self.c2 = c1, c2
        self.rng = np.random.default_rng(seed)

        self.lower, self.upper = search_space.pso_bounds
        self.v_max = v_max if v_max is not None else k * (self.upper - self.lower) / 2.0

        self._particles = [
            _Particle(self.lower, self.upper, self.rng, self.mode)
            for _ in range(num_particles)
        ]
        self.global_best_position: np.ndarray | None = None
        self.global_best_fitness = np.inf if self.mode == "min" else -np.inf

    def _better(self, a: float, b: float) -> bool:
        return a < b if self.mode == "min" else a > b

    def optimize(
        self,
        max_iterations: int,
        inertia_strategy: float | str = 0.5,
        return_history: bool = False,
    ) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, list[dict]]:
        """
        Run PSO for ``max_iterations`` iterations.

        Parameters
        ----------
        max_iterations : int
        inertia_strategy : float or str
            Constant weight (float) or a named strategy from ``InertiaRegistry``.
            See module docstring for built-in names.
        return_history : bool
            When ``True``, returns a third element: list of
            ``{"iter": int, "best": float}`` dicts.

        Returns
        -------
        best_position : np.ndarray
        best_fitness : float
        history : list[dict], only when ``return_history=True``
        """
        w_fn = InertiaRegistry.get(inertia_strategy)
        history: list[dict] = []

        for t in range(max_iterations):
            for p in self._particles:
                fit = self._fn(p.position)
                if self._better(fit, p.best_fitness):
                    p.best_position = p.position.copy()
                    p.best_fitness = fit
                if self._better(fit, self.global_best_fitness):
                    self.global_best_position = p.position.copy()
                    self.global_best_fitness = fit

            for p in self._particles:
                w = w_fn(t, max_iterations, p, self.global_best_position,
                         self.global_best_fitness, self.rng)
                p.update_velocity(self.global_best_position, self.rng,
                                  w, self.c1, self.c2, self.v_max)
                p.update_position(self.lower, self.upper)

            if return_history:
                history.append({"iter": t, "best": self.global_best_fitness})

        if return_history:
            return self.global_best_position, self.global_best_fitness, history
        return self.global_best_position, self.global_best_fitness
