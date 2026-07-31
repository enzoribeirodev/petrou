"""
petrou.optimization._pso_base
==============================
Shared internals for every particle-swarm optimizer in petrou (PSO, PSOLF,
and any future variant).

What lives here, and why it isn't copy-pasted per variant:

* ``Particle``        — position/velocity/personal-best state. Every swarm
                         variant needs exactly this; a variant that needs
                         more (e.g. a stagnation counter) subclasses it and
                         adds its own ``__slots__``.
* ``InertiaRegistry``  — pluggable named inertia-weight strategies. Defined
                         here rather than in ``pso.py`` so that ``pso.py``,
                         ``psolf.py``, and any variant added later can
                         register and consume strategies without importing
                         each other.
* ``SwarmOptimizer``   — the outer iteration loop: evaluate fitness, track
                         personal/global best, resolve this iteration's
                         inertia weight, then hand off to the subclass's
                         ``_apply_update`` for the one thing that actually
                         differs between variants — the velocity/position
                         update rule.

A new swarm variant subclasses ``SwarmOptimizer`` and implements
``_apply_update(particle, w)``; construction, the evaluate/track/move loop,
and history recording are all inherited.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from petrou.optimization.search_space import SearchSpace

__all__ = ["InertiaRegistry", "Particle", "SwarmOptimizer"]


# ---------------------------------------------------------------------------
# Inertia registry
# ---------------------------------------------------------------------------

class InertiaRegistry:
    """
    Registry of named inertia-weight strategies, shared by every swarm optimizer.

    Strategies are plain callables with the signature::

        fn(t, max_iter, particle, g_pos, g_fit, rng) -> float | np.ndarray

    where:
        t         int                current iteration (0-indexed)
        max_iter  int                total iterations planned
        particle  Particle            current particle object
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


# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------

class Particle:
    """Position/velocity/personal-best state shared by every swarm variant."""

    __slots__ = ("position", "velocity", "best_position", "best_fitness", "dimensions")

    def __init__(self, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator, mode: str) -> None:
        self.dimensions = len(lower)
        self.position = rng.uniform(lower, upper)
        self.velocity = np.zeros(self.dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = np.inf if mode == "min" else -np.inf

    def clip(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self.position = np.clip(self.position, lower, upper)


# ---------------------------------------------------------------------------
# SwarmOptimizer — shared iteration loop
# ---------------------------------------------------------------------------

class SwarmOptimizer:
    """
    Template base for particle-swarm optimizers sharing the SearchSpace interface.

    Subclasses may set ``particle_cls`` (only needed if extra per-particle
    state is required) and must implement ``_apply_update(particle, w)``,
    which sets ``particle.velocity`` and ``particle.position`` in place.
    Everything else — construction, the evaluate/track/move loop, inertia
    resolution, and history — is handled here.
    """

    particle_cls = Particle
    default_inertia_strategy: float | str = 0.5

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        num_particles: int,
        *,
        search_space: SearchSpace,
        mode: str = "max",
        seed: int | None = None,
    ) -> None:
        self._fn = objective_fn
        self.mode = mode.lower()
        self.rng = np.random.default_rng(seed)
        self.lower, self.upper = search_space.pso_bounds

        self._particles = [
            self.particle_cls(self.lower, self.upper, self.rng, self.mode)
            for _ in range(num_particles)
        ]
        self.global_best_position: np.ndarray | None = None
        self.global_best_fitness = np.inf if self.mode == "min" else -np.inf

    def _better(self, a: float, b: float) -> bool:
        return a < b if self.mode == "min" else a > b

    def _apply_update(self, p: Particle, w: float) -> None:
        """Move one particle — set ``p.velocity`` and ``p.position``. Implemented by subclasses."""
        raise NotImplementedError

    def optimize(
        self,
        max_iterations: int,
        inertia_strategy: float | str | None = None,
        return_history: bool = False,
    ) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, list[dict]]:
        """
        Run the swarm for ``max_iterations`` iterations.

        Parameters
        ----------
        max_iterations : int
        inertia_strategy : float, str, or None
            Constant weight (float) or a named strategy from ``InertiaRegistry``.
            Defaults to ``self.default_inertia_strategy`` when omitted.
        return_history : bool
            When ``True``, returns a third element: list of
            ``{"iter": int, "best": float}`` dicts.

        Returns
        -------
        best_position : np.ndarray
        best_fitness : float
        history : list[dict], only when ``return_history=True``
        """
        if inertia_strategy is None:
            inertia_strategy = self.default_inertia_strategy
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
                self._apply_update(p, w)
                p.clip(self.lower, self.upper)

            if return_history:
                history.append({"iter": t, "best": self.global_best_fitness})

        if return_history:
            return self.global_best_position, self.global_best_fitness, history
        return self.global_best_position, self.global_best_fitness
