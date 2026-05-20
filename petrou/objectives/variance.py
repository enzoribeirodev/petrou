"""
petrou.objectives.variance
===========================
Variance-based thresholding objective: Otsu's between-class variance.

A single function ``otsu_criterion`` handles both use cases:

* ``t=None`` — vectorised: O(256) NumPy pass returning all variances at once.
  Used by ``find_otsu_threshold(optimizer="exhaustive")``.

* ``t=int``  — pointwise scalar for SA / PSO objective functions.

No duplication, no separate "compute_all" helper.
"""

from __future__ import annotations

import numpy as np

__all__ = ["otsu_criterion"]


def otsu_criterion(
    hist: np.ndarray,
    t: int | None = None,
) -> float | np.ndarray:
    """
    Between-class variance (Otsu criterion).

    Parameters
    ----------
    hist : np.ndarray, shape (256,)
        Absolute-frequency histogram.
    t : int or None
        * ``None`` — return an ``np.ndarray`` of shape ``(256,)`` with
          the between-class variance for every candidate threshold.
          Index ``i`` corresponds to threshold ``t = i``.
        * ``int``  — return the scalar variance for threshold ``t``.

    Returns
    -------
    float or np.ndarray
        Higher value = better threshold in both modes.

    Notes
    -----
    The vectorised path (``t=None``) is an O(256) cumulative-sum computation
    with no Python loop — always prefer it over 255 individual calls when
    doing exhaustive search.
    """
    total = hist.sum()
    if total == 0:
        return np.zeros(256) if t is None else 0.0

    p = hist.astype(np.float64) / total
    eps = 1e-12

    if t is None:
        w_bg = np.cumsum(p) + eps
        w_fg = 1.0 - w_bg + eps
        mu_bg = np.cumsum(p * np.arange(256)) / w_bg
        total_mean = float(np.dot(p, np.arange(256)))
        mu_fg = (total_mean - np.cumsum(p * np.arange(256))) / w_fg
        return w_bg * w_fg * (mu_bg - mu_fg) ** 2

    cumsum = np.cumsum(p)
    PA = float(cumsum[t])
    PB = 1.0 - PA
    if PA == 0.0 or PB == 0.0:
        return 0.0
    muA = float(np.dot(p[: t + 1], np.arange(t + 1)) / PA)
    muB = float(np.dot(p[t + 1 :], np.arange(t + 1, 256)) / PB)
    return PA * PB * (muA - muB) ** 2
