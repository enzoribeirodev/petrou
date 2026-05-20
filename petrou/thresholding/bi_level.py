"""
petrou.thresholding.bi_level
=============================
Bi-level (binary) thresholding: Otsu, Tsallis, MASI.

Every public function returns a :class:`ThresholdResult` and follows the
same three-phase pipeline:

1. **Histogram** — computed once from ``img_region`` and reused.
2. **Parameter resolution** — scalar parameters (``q``, ``r``) are either
    fixed by the caller, estimated analytically, or co-optimized jointly
   with ``t``.
3. **Search** — dispatched to the chosen optimizer via :func:`_run_optimizer`.

Optimizer dispatch
------------------
``"exhaustive"``
    Deterministic scan.  Best for Otsu (vectorised O(256)).  Available for
    Tsallis / MASI only when the scalar parameter is already resolved
    (strategy ``"automatic"`` or ``"fixed"``).

``"sa"``
    :func:`~petrou.optimization.sa.simulated_annealing`.  Works for 1-D
    (``t`` only) and 2-D (parameter + ``t``) spaces without restrictions.

``"pso"``
    :class:`~petrou.optimization.pso.PSO`.  Same SearchSpace interface as SA.

Adding a new optimizer
-----------------------
Add a branch to :func:`_run_optimizer`.  Nothing else needs to change — all
threshold finders and the multi-level engine call that single dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from petrou.exceptions import EmptyHistogramError, IncompatibleStrategyError
from petrou.objectives.entropy import (
    masi_entropy,
    masi_r_adaptive,
    tsallis_entropy,
    tsallis_q_automatic,
)
from petrou.objectives.variance import otsu_criterion
from petrou.optimization.exhaustive import exhaustive_search
from petrou.optimization.pso import PSO
from petrou.optimization.sa import simulated_annealing
from petrou.optimization.search_space import SearchSpace

__all__ = [
    "ThresholdResult",
    "find_otsu_threshold",
    "find_tsallis_threshold",
    "find_masi_threshold",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    """
    Structured output of every bi-level threshold finder.

    Attributes
    ----------
    threshold : int
        Optimal threshold in ``[0, 255]``.  Apply with ``img > threshold``.
    score : float
        Objective value at the optimum.  Higher = better for all criteria.
    params : dict[str, float | int]
        Criterion-specific estimated or optimized parameters.

        - Otsu → ``{}``
        - Tsallis → ``{"q": <value>}``
        - MASI → ``{"r": <value>}``
    optimizer : str
        Which optimizer produced this result.

    Examples
    --------
    >>> result = find_otsu_threshold(img)
    >>> binary = img > result.threshold
    >>> print(result.score, result.optimizer)
    """

    threshold: int
    score: float
    params: dict[str, float | int] = field(default_factory=dict)
    optimizer: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _histogram(img: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    if hist.sum() == 0:
        raise EmptyHistogramError("Image region is empty — histogram sums to zero.")
    return hist


def _run_optimizer(
    objective_fn: Any,
    space: SearchSpace,
    optimizer: str,
    config: dict,
) -> tuple[np.ndarray, float]:
    """
    Dispatch ``objective_fn`` + ``space`` to the chosen optimizer.

    To add a new optimizer, add a branch here.  Every threshold finder and
    the multi-level engine reach this function, so one change wires everything.

    Parameters
    ----------
    objective_fn : Callable[[np.ndarray], float]
    space : SearchSpace
    optimizer : str
    config : dict
        Keyword arguments forwarded to the optimizer (consumed in-place).

    Returns
    -------
    (best_state: np.ndarray, best_score: float)
    """
    if optimizer == "sa":
        cfg = {"maximize": True, **config}
        result = simulated_annealing(objective_fn, space, **cfg)
        return result[0], result[1]

    if optimizer == "pso":
        n = config.pop("n_particles", 20)
        iters = config.pop("max_iterations", 100)
        mode = config.pop("mode", "max")
        pso = PSO(objective_fn, n, search_space=space, mode=mode, **config)
        return pso.optimize(iters)

    raise IncompatibleStrategyError(
        f"Unknown optimizer '{optimizer}'. Choose 'exhaustive', 'sa', or 'pso'."
    )


def _1d_space(search_range: tuple[int, int], step: int) -> SearchSpace:
    """Convenience: build a single-integer-variable SearchSpace for ``t``."""
    return SearchSpace([{"name": "t", "type": "int", "bounds": search_range, "step": step}])


# ---------------------------------------------------------------------------
# Otsu
# ---------------------------------------------------------------------------

def find_otsu_threshold(
    img_region: np.ndarray,
    *,
    optimizer: str = "exhaustive",
    search_range: tuple[int, int] = (0, 255),
    optimizer_config: dict[str, Any] | None = None,
) -> ThresholdResult:
    """
    Find the Otsu bi-level threshold.

    Parameters
    ----------
    img_region : np.ndarray
        Grayscale pixel array, any shape.
    optimizer : {"exhaustive", "sa", "pso"}
        ``"exhaustive"`` is strongly recommended — it calls
        ``otsu_criterion(hist)`` (no ``t``), computing all 256 variances in
        one vectorised pass with no loop.
    search_range : (int, int)
        ``[lo, hi)`` inclusive-exclusive threshold interval.
    optimizer_config : dict, optional
        Keyword arguments forwarded to the chosen optimizer.
        SA extra key: ``step`` (int threshold step, default 10).
        PSO extra keys: ``n_particles`` (default 20), ``max_iterations``
        (default 100).

    Returns
    -------
    ThresholdResult
        ``.params`` is always ``{}``.
    """
    config = dict(optimizer_config or {})
    hist = _histogram(img_region)

    if optimizer == "exhaustive":
        variances = otsu_criterion(hist)          # shape (256,) vectorised
        lo, hi = search_range
        region = variances[lo:hi]
        if region.size == 0:
            return ThresholdResult(threshold=lo, score=-np.inf, optimizer=optimizer)
        idx = int(np.argmax(region))
        return ThresholdResult(threshold=lo + idx, score=float(region[idx]),
                               optimizer=optimizer)

    step = config.pop("step", 10)
    space = _1d_space(search_range, step)
    obj = lambda s: otsu_criterion(hist, int(round(s[0])))
    best, score = _run_optimizer(obj, space, optimizer, config)
    return ThresholdResult(threshold=int(space.decode(best)["t"]),
                           score=score, optimizer=optimizer)


# ---------------------------------------------------------------------------
# Tsallis
# ---------------------------------------------------------------------------

def find_tsallis_threshold(
    img_region: np.ndarray,
    *,
    q_strategy: str = "automatic",
    q_fixed: float | None = None,
    optimizer: str = "exhaustive",
    search_range: tuple[int, int] = (1, 255),
    q_bounds: tuple[float, float] = (0.01, 3.0),
    q_step: float = 0.05,
    add_log_noise: bool = False,
    optimizer_config: dict[str, Any] | None = None,
) -> ThresholdResult:
    """
    Find the Tsallis bi-level threshold.

    Parameters
    ----------
    img_region : np.ndarray
    q_strategy : {"automatic", "fixed", "optimize"}
        ``"automatic"``
            Estimate ``q`` analytically via :func:`tsallis_q_automatic`.
            Fast and parameter-free.  Compatible with all optimizers.
        ``"fixed"``
            Use ``q_fixed`` directly.  Compatible with all optimizers.
        ``"optimize"``
            Treat ``q`` as a free variable and co-optimize with ``t``.
            Requires ``optimizer`` in ``{"sa", "pso"}`` — exhaustive search
            cannot enumerate a continuous variable.
    q_fixed : float, optional
        Required when ``q_strategy="fixed"``.
    optimizer : {"exhaustive", "sa", "pso"}
    search_range : (int, int)
        Threshold search bounds ``[lo, hi)``.
    q_bounds : (float, float)
        Bounds for ``q`` when ``q_strategy="optimize"``.
    q_step : float
        Gaussian σ for ``q`` perturbation in SA / PSO.
    add_log_noise : bool
        Add ε = 1e-12 inside log.  Recommended with ``q_strategy="optimize"``.
    optimizer_config : dict, optional
        Extra SA key: ``t_step`` (default 5).
        Extra PSO keys: ``n_particles``, ``max_iterations``.

    Returns
    -------
    ThresholdResult
        ``.params`` contains ``{"q": <value>}``.
    """
    config = dict(optimizer_config or {})
    hist = _histogram(img_region)

    if q_strategy == "optimize":
        if optimizer == "exhaustive":
            raise IncompatibleStrategyError(
                "q_strategy='optimize' needs optimizer='sa' or 'pso'. "
                "The joint (q, t) space cannot be exhaustively enumerated. "
                "Use q_strategy='automatic' or 'fixed' with exhaustive."
            )
        t_step = config.pop("t_step", 5)
        space = SearchSpace([
            {"name": "q", "type": "float", "bounds": q_bounds, "step": q_step},
            {"name": "t", "type": "int",   "bounds": search_range, "step": t_step},
        ])
        obj = lambda s: tsallis_entropy(hist, float(s[0]), int(round(s[1])), add_log_noise)
        best, score = _run_optimizer(obj, space, optimizer, config)
        dec = space.decode(best)
        return ThresholdResult(threshold=int(dec["t"]), score=score,
                               params={"q": float(dec["q"])}, optimizer=optimizer)

    if q_strategy == "automatic":
        q, _ = tsallis_q_automatic(hist)
    elif q_strategy == "fixed":
        if q_fixed is None:
            raise IncompatibleStrategyError("q_strategy='fixed' requires q_fixed.")
        q = float(q_fixed)
    else:
        raise IncompatibleStrategyError(
            f"Unknown q_strategy '{q_strategy}'. Choose 'automatic', 'fixed', or 'optimize'."
        )

    obj_1d = partial(tsallis_entropy, hist, q, add_log_noise=add_log_noise)

    if optimizer == "exhaustive":
        t, score = exhaustive_search(obj_1d, search_range)
        return ThresholdResult(threshold=t, score=score, params={"q": q}, optimizer=optimizer)

    t_step = config.pop("t_step", 5)
    space = _1d_space(search_range, t_step)
    obj = lambda s: obj_1d(int(round(s[0])))
    best, score = _run_optimizer(obj, space, optimizer, config)
    return ThresholdResult(threshold=int(space.decode(best)["t"]), score=score,
                           params={"q": q}, optimizer=optimizer)


# ---------------------------------------------------------------------------
# MASI
# ---------------------------------------------------------------------------

def find_masi_threshold(
    img_region: np.ndarray,
    *,
    r_strategy: str = "adaptive",
    r_fixed: float | None = None,
    optimizer: str = "exhaustive",
    search_range: tuple[int, int] = (1, 255),
    r_bounds: tuple[float, float] = (0.01, 3.0),
    r_step: float = 0.05,
    add_log_noise: bool = True,
    optimizer_config: dict[str, Any] | None = None,
) -> ThresholdResult:
    """
    Find the MASI bi-level threshold.

    Parameters
    ----------
    img_region : np.ndarray
    r_strategy : {"adaptive", "fixed", "optimize"}
        ``"adaptive"``
            Estimate ``r`` from ``argmax(hist) / max(pixel_value)``.
            Fast.  Compatible with all optimizers.
        ``"fixed"``
            Use ``r_fixed`` directly.  Compatible with all optimizers.
        ``"optimize"``
            Co-optimize ``r`` and ``t`` jointly.
            Requires ``optimizer`` in ``{"sa", "pso"}``.
    r_fixed : float, optional
        Required when ``r_strategy="fixed"``.
    optimizer : {"exhaustive", "sa", "pso"}
    search_range : (int, int)
    r_bounds : (float, float)
    r_step : float
    add_log_noise : bool
    optimizer_config : dict, optional

    Returns
    -------
    ThresholdResult
        ``.params`` contains ``{"r": <value>}``.
    """
    config = dict(optimizer_config or {})
    hist = _histogram(img_region)

    if r_strategy == "optimize":
        if optimizer == "exhaustive":
            raise IncompatibleStrategyError(
                "r_strategy='optimize' needs optimizer='sa' or 'pso'. "
                "The joint (r, t) space cannot be exhaustively enumerated."
            )
        t_step = config.pop("t_step", 5)
        space = SearchSpace([
            {"name": "r", "type": "float", "bounds": r_bounds, "step": r_step},
            {"name": "t", "type": "int",   "bounds": search_range, "step": t_step},
        ])
        obj = lambda s: masi_entropy(hist, float(s[0]), int(round(s[1])), add_log_noise)
        best, score = _run_optimizer(obj, space, optimizer, config)
        dec = space.decode(best)
        return ThresholdResult(threshold=int(dec["t"]), score=score,
                               params={"r": float(dec["r"])}, optimizer=optimizer)

    if r_strategy == "adaptive":
        r = masi_r_adaptive(hist, img_region)
    elif r_strategy == "fixed":
        if r_fixed is None:
            raise IncompatibleStrategyError("r_strategy='fixed' requires r_fixed.")
        r = float(r_fixed)
    else:
        raise IncompatibleStrategyError(
            f"Unknown r_strategy '{r_strategy}'. Choose 'adaptive', 'fixed', or 'optimize'."
        )

    obj_1d = partial(masi_entropy, hist, r, add_log_noise=add_log_noise)

    if optimizer == "exhaustive":
        t, score = exhaustive_search(obj_1d, search_range)
        return ThresholdResult(threshold=t, score=score, params={"r": r}, optimizer=optimizer)

    t_step = config.pop("t_step", 5)
    space = _1d_space(search_range, t_step)
    obj = lambda s: obj_1d(int(round(s[0])))
    best, score = _run_optimizer(obj, space, optimizer, config)
    return ThresholdResult(threshold=int(space.decode(best)["t"]), score=score,
                           params={"r": r}, optimizer=optimizer)
