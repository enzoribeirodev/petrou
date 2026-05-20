"""
petrou.optimization.exhaustive
===============================
Exact exhaustive search over a 1-D integer range.

This is the correct and preferred method for 1-D integer problems like the
Otsu criterion, where the objective can be evaluated for all candidate values
in a single O(N) pass.  Use a stochastic optimizer (SA, PSO) only when the
problem is too large or continuous to enumerate.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["exhaustive_search"]


def exhaustive_search(
    objective_fn: Callable[[int], float],
    search_range: tuple[int, int] = (1, 255),
    maximize: bool = True,
) -> tuple[int, float]:
    """
    Exact search over every integer in ``[search_range[0], search_range[1])``.

    Parameters
    ----------
    objective_fn : Callable[[int], float]
        Maps each candidate integer to a scalar score.
    search_range : (int, int)
        Half-open interval ``[lo, hi)`` — ``hi`` is exclusive.
    maximize : bool
        ``True`` (default) to find the maximum; ``False`` for the minimum.

    Returns
    -------
    best_t : int
    best_score : float

    Examples
    --------
    >>> exhaustive_search(lambda t: -(t - 100) ** 2, (1, 255))
    (100, 0.0)
    """
    lo, hi = search_range
    best_score = -np.inf if maximize else np.inf
    best_t = lo

    for t in range(lo, hi):
        score = objective_fn(t)
        if (maximize and score > best_score) or (not maximize and score < best_score):
            best_score = score
            best_t = t

    return best_t, best_score
