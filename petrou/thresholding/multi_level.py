"""
petrou.thresholding.multi_level
================================
Multi-level thresholding via recursive bi-level splitting.

A single engine :func:`_multilevel_engine` drives all criteria.  It receives
any ``threshold_finder`` callable with the signature::

    threshold_finder(img_region: np.ndarray) -> ThresholdResult

and is therefore criterion-agnostic — adding a new criterion to
``bi_level.py`` automatically makes it available here.

Algorithm
---------
The engine performs ``k − 1`` splits to produce ``k`` classes:

1. Start with the full image as a single region.
2. Apply the threshold finder to the current region list.
3. Each split adds one threshold and two sub-regions.
4. Repeat until ``k − 1`` thresholds are collected.

``k`` can be any integer ≥ 2 (no longer restricted to powers of 2).
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import numpy as np

from petrou.thresholding.bi_level import (
    ThresholdResult,
    find_masi_threshold,
    find_otsu_threshold,
    find_tsallis_threshold,
)

__all__ = ["multilevel_otsu", "multilevel_tsallis", "multilevel_masi"]


def _multilevel_engine(
    img: np.ndarray,
    k: int,
    threshold_finder,
) -> tuple[np.ndarray, dict]:
    """
    Generic multi-level thresholding engine.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image, any shape.
    k : int
        Number of output intensity classes (k ≥ 2).
    threshold_finder : Callable[[np.ndarray], ThresholdResult]

    Returns
    -------
    segmented : np.ndarray, same shape as ``img``, dtype float32
        Each pixel replaced by the mean intensity of its class.
    info : dict
        ``{"thresholds": list[int]}`` — sorted list of the k−1 thresholds.
    """
    if k < 2:
        raise ValueError(f"k must be ≥ 2, got {k}.")

    regions: list[np.ndarray] = [img.ravel()]
    thresholds: list[int] = []

    while len(thresholds) < k - 1 and regions:
        next_regions: list[np.ndarray] = []
        for region in regions:
            if len(thresholds) >= k - 1:
                next_regions.append(region)
                continue
            if region.size == 0:
                continue
            result: ThresholdResult = threshold_finder(img_region=region)
            t = result.threshold
            if t in thresholds:
                warnings.warn(
                    f"Duplicate threshold {t} — region will not be further split.",
                    RuntimeWarning, stacklevel=4,
                )
                next_regions.append(region)
                continue
            thresholds.append(t)
            next_regions.extend([region[region <= t], region[region > t]])
        regions = next_regions

    if not thresholds:
        warnings.warn("No thresholds found — returning original image.", RuntimeWarning, stacklevel=2)
        return img.copy(), {"thresholds": []}

    ts = sorted(thresholds)
    flat = img.ravel().astype(np.float32)
    out = np.empty_like(flat)

    # class 0 — pixels ≤ ts[0]
    m = flat <= ts[0]
    out[m] = flat[m].mean() if m.any() else 0.0

    # intermediate classes
    for i in range(len(ts) - 1):
        m = (flat > ts[i]) & (flat <= ts[i + 1])
        out[m] = flat[m].mean() if m.any() else (ts[i] + ts[i + 1]) / 2.0

    # last class — pixels > ts[-1]
    m = flat > ts[-1]
    out[m] = flat[m].mean() if m.any() else 255.0

    return out.reshape(img.shape), {"thresholds": ts}


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

def multilevel_otsu(
    img: np.ndarray,
    k: int,
    *,
    optimizer: str = "exhaustive",
    search_range: tuple[int, int] = (0, 255),
    optimizer_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Multi-level Otsu thresholding.

    Parameters
    ----------
    img : np.ndarray
    k : int
        Number of output classes (k ≥ 2).
    optimizer : {"exhaustive", "sa", "pso"}
    search_range : (int, int)
    optimizer_config : dict, optional

    Returns
    -------
    segmented : np.ndarray
    info : dict — ``{"thresholds": list[int]}``
    """
    finder = partial(find_otsu_threshold, optimizer=optimizer,
                     search_range=search_range, optimizer_config=optimizer_config)
    return _multilevel_engine(img, k, finder)


def multilevel_tsallis(
    img: np.ndarray,
    k: int,
    *,
    q_strategy: str = "automatic",
    q_fixed: float | None = None,
    optimizer: str = "exhaustive",
    search_range: tuple[int, int] = (1, 255),
    q_bounds: tuple[float, float] = (0.01, 3.0),
    q_step: float = 0.05,
    add_log_noise: bool = False,
    optimizer_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Multi-level Tsallis entropy thresholding.

    All parameters mirror :func:`~petrou.thresholding.bi_level.find_tsallis_threshold`.

    Returns
    -------
    segmented : np.ndarray
    info : dict — ``{"thresholds": list[int]}``
    """
    finder = partial(
        find_tsallis_threshold,
        q_strategy=q_strategy, q_fixed=q_fixed, optimizer=optimizer,
        search_range=search_range, q_bounds=q_bounds, q_step=q_step,
        add_log_noise=add_log_noise, optimizer_config=optimizer_config,
    )
    return _multilevel_engine(img, k, finder)


def multilevel_masi(
    img: np.ndarray,
    k: int,
    *,
    r_strategy: str = "adaptive",
    r_fixed: float | None = None,
    optimizer: str = "exhaustive",
    search_range: tuple[int, int] = (1, 255),
    r_bounds: tuple[float, float] = (0.01, 3.0),
    r_step: float = 0.05,
    add_log_noise: bool = True,
    optimizer_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Multi-level MASI entropy thresholding.

    All parameters mirror :func:`~petrou.thresholding.bi_level.find_masi_threshold`.

    Returns
    -------
    segmented : np.ndarray
    info : dict — ``{"thresholds": list[int]}``
    """
    finder = partial(
        find_masi_threshold,
        r_strategy=r_strategy, r_fixed=r_fixed, optimizer=optimizer,
        search_range=search_range, r_bounds=r_bounds, r_step=r_step,
        add_log_noise=add_log_noise, optimizer_config=optimizer_config,
    )
    return _multilevel_engine(img, k, finder)
