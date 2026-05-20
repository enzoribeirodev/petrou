"""
petrou.optimization.sa
======================
Simulated Annealing — N-dimensional, mixed-type, SearchSpace-native.

The algorithm follows the classical Metropolis–Hastings schedule:
at each step a candidate neighbour is generated via
``search_space.neighbour``, accepted unconditionally when it improves the
objective, or accepted with probability ``exp(Δ / (k·T))`` otherwise.

Because the SearchSpace encodes how to move in each dimension, SA works on
any combination of float and integer variables with no extra code.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from petrou.exceptions import OptimizationError
from petrou.optimization.search_space import SearchSpace

__all__ = ["simulated_annealing"]


def simulated_annealing(
    objective_fn: Callable[[np.ndarray], float],
    search_space: SearchSpace,
    *,
    T_init: float = 100.0,
    T_min: float = 1e-3,
    alpha: float = 0.9,
    markov_length: int = 20,
    boltzmann_k: float = 1.0,
    max_iter: int = 1_000,
    maximize: bool = True,
    perturbation: str = "independent",
    random_state: int | None = None,
    return_history: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, list[dict]]:
    """
    Simulated Annealing optimizer.

    Parameters
    ----------
    objective_fn : Callable[[np.ndarray], float]
        Function to optimize. Receives the raw ``float64`` state vector.
        Use ``search_space.decode(state)`` inside if you need typed values.
    search_space : SearchSpace
        Defines bounds, types, and step sizes for every variable.
    T_init : float
        Initial temperature. Higher → more exploration early on.
    T_min : float
        Stopping temperature.
    alpha : float in (0, 1)
        Geometric cooling factor. After each Markov chain: ``T ← T * alpha``.
    markov_length : int
        Number of candidate evaluations per temperature level.
    boltzmann_k : float
        Boltzmann constant analogue in the acceptance formula. Leave at 1.0
        for a dimensionless formulation.
    max_iter : int
        Hard cap on total objective evaluations.
    maximize : bool
        ``True`` to maximise ``objective_fn``, ``False`` to minimise.
    perturbation : {"independent", "single"}
        Neighbourhood strategy forwarded to ``SearchSpace.neighbour``.
    random_state : int or None
        Seed for reproducibility.
    return_history : bool
        When ``True``, also return a convergence trace as a list of dicts
        ``[{"iter": int, "T": float, "best": float}, …]``.

    Returns
    -------
    best_state : np.ndarray, shape (ndim,)
        Raw float64 vector of the best state found.
        Decode with ``search_space.decode(best_state)``.
    best_score : float
    history : list[dict]
        Only returned when ``return_history=True``.

    Raises
    ------
    OptimizationError
        If ``objective_fn`` raises on the initial state evaluation.

    Notes
    -----
    Total chains ≈ log(T_min / T_init) / log(alpha).
    Total evaluations ≈ chains × markov_length, capped at max_iter.

    Examples
    --------
    >>> from petrou.optimization import SearchSpace, simulated_annealing
    >>> space = SearchSpace([
    ...     {"name": "x", "type": "float", "bounds": (-5.0, 5.0), "step": 0.2},
    ... ])
    >>> best, score = simulated_annealing(
    ...     lambda s: -(s[0] ** 2),
    ...     space,
    ...     T_init=50.0, max_iter=500, maximize=True, random_state=0,
    ... )
    >>> space.decode(best)
    {"x": 0.01}
    """
    rng = np.random.default_rng(random_state)
    state = search_space.initial_state(rng)

    try:
        f_cur = objective_fn(state)
    except Exception as exc:
        raise OptimizationError(
            f"objective_fn raised on the initial state: {exc}"
        ) from exc

    best_state = state.copy()
    best_score = f_cur
    T = T_init
    iters = 0
    history: list[dict] = []

    while T > T_min and iters < max_iter:
        for _ in range(markov_length):
            s_new = search_space.neighbour(state, rng, perturbation=perturbation)
            f_new = objective_fn(s_new)
            delta = (f_new - f_cur) if maximize else (f_cur - f_new)

            if delta >= 0 or rng.random() < np.exp(delta / (boltzmann_k * T)):
                state, f_cur = s_new, f_new
                if (maximize and f_cur > best_score) or (not maximize and f_cur < best_score):
                    best_state = state.copy()
                    best_score = f_cur

            iters += 1
            if iters >= max_iter:
                break

        if return_history:
            history.append({"iter": iters, "T": T, "best": best_score})
        T *= alpha

    if return_history:
        return best_state, best_score, history
    return best_state, best_score
