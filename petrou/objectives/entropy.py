"""
petrou.objectives.entropy
==========================
Entropy-based thresholding objective functions: Tsallis and MASI.

All functions accept a pre-computed histogram as their first argument so
that the caller owns when the histogram is computed.  This avoids repeating
``np.histogram`` inside every optimizer iteration.

Convention: higher score = better threshold for all functions here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "tsallis_entropy",
    "tsallis_q_automatic",
    "masi_entropy",
    "masi_r_adaptive",
]


def tsallis_entropy(
    hist: np.ndarray,
    q: float,
    t: int,
    add_log_noise: bool = False,
) -> float:
    """
    Tsallis non-extensive entropy criterion for bi-level thresholding.

    Splits the histogram at ``t`` into background (A, pixels ≤ t) and
    foreground (B, pixels > t) and computes the non-additive combination:

        S_q(A,B) = S_q(A) + S_q(B) + (1 − q)·S_q(A)·S_q(B)

    When ``q → 1`` the formula reduces to the Shannon entropy sum (Kapur).

    Parameters
    ----------
    hist : np.ndarray, shape (256,)
        Absolute-frequency histogram of the image region.
    q : float
        Entropic index.  ``q = 1`` → Shannon.  Typical useful range: (0.01, 3.0).
    t : int
        Candidate threshold in ``[0, 255]``.
    add_log_noise : bool
        Add ε = 1e-12 inside logarithms to prevent log(0).  Recommended
        when jointly optimizing ``q`` over sparse histograms.

    Returns
    -------
    float
        Tsallis entropy value — higher is better.
    """
    p = hist.astype(np.float64) / hist.sum()
    cumsum = np.cumsum(p)
    PA, PB = float(cumsum[t]), 1.0 - float(cumsum[t])

    if PA == 0.0 or PB == 0.0:
        return 0.0

    pA = p[: t + 1] / PA
    pB = p[t + 1 :] / PB

    if abs(q - 1.0) < 1e-9:          # Shannon limit
        eps = 1e-12 if add_log_noise else 0.0
        return float(-np.sum(pA * np.log(pA + eps)) - np.sum(pB * np.log(pB + eps)))

    SA = (1.0 - float(np.sum(pA ** q))) / (q - 1.0)
    SB = (1.0 - float(np.sum(pB ** q))) / (q - 1.0)
    return SA + SB + (1.0 - q) * SA * SB


def tsallis_q_automatic(
    hist: np.ndarray,
    q_min: float = 0.01,
    q_max: float = 2.0,
    steps: int = 200,
) -> tuple[float, float]:
    """
    Estimate the optimal Tsallis ``q`` from the image histogram.

    Selects the ``q`` that minimises the ratio S_q / S_q_max, where S_q_max
    is the entropy of the uniform distribution.  A low ratio signals that the
    histogram is far from uniform (high contrast) — the regime where Tsallis
    is most discriminative.

    Parameters
    ----------
    hist : np.ndarray, shape (256,)
    q_min, q_max : float
    steps : int
        Number of ``q`` values evaluated.

    Returns
    -------
    q_opt : float
    ratio_min : float
        Diagnostic value — the minimum S_q / S_q_max ratio found.
    """
    p = hist.astype(np.float64) / hist.sum()
    q_vals = np.linspace(q_min, q_max, steps)
    ratios = np.empty(steps)

    for i, q in enumerate(q_vals):
        if abs(q - 1.0) < 1e-9:
            Sq = float(-np.sum(p * np.log(p + 1e-12)))
            Sm = float(np.log(256))
        else:
            Sq = float((1.0 - np.sum(p ** q)) / (q - 1.0))
            Sm = float((1.0 - 256 ** (1.0 - q)) / (q - 1.0))
        ratios[i] = Sq / Sm if Sm != 0 else 1.0

    idx = int(np.argmin(ratios))
    return float(q_vals[idx]), float(ratios[idx])


def masi_entropy(
    hist: np.ndarray,
    r: float,
    t: int,
    add_log_noise: bool = True,
    verbose: bool = False,
) -> float:
    """
    Modified Arimoto–Shepp–Information (MASI) entropy criterion.

    When ``r = 1`` the formula reduces to the Shannon entropy sum.

    Parameters
    ----------
    hist : np.ndarray, shape (256,)
    r : float
        Shape parameter.  ``r = 1`` → Shannon.  Typical range: (0.01, 3.0).
    t : int
        Candidate threshold.
    add_log_noise : bool
        Stabilise log(0) with ε = 1e-12.  Default ``True`` — MASI is more
        sensitive to log(0) than Tsallis.
    verbose : bool
        Emit a ``RuntimeWarning`` when the log argument is non-positive
        (``r`` too large for this image's entropy range).

    Returns
    -------
    float
        MASI entropy value — higher is better.
    """
    p = hist.astype(np.float64) / hist.sum()
    cumsum = np.cumsum(p)
    PA, PB = float(cumsum[t]), 1.0 - float(cumsum[t])

    if PA == 0.0 or PB == 0.0:
        return 0.0

    eps = 1e-12 if add_log_noise else 0.0
    pA = p[: t + 1] / PA
    pB = p[t + 1 :] / PB

    if abs(r - 1.0) < 1e-9:          # Shannon limit
        return float(-np.sum(pA * np.log(pA + eps)) - np.sum(pB * np.log(pB + eps)))

    EA = float(np.sum(pA * np.log(pA + eps)))
    EB = float(np.sum(pB * np.log(pB + eps)))
    argA = 1.0 - (1.0 - r) * EA
    argB = 1.0 - (1.0 - r) * EB

    if argA <= 0.0 or argB <= 0.0:
        if verbose:
            import warnings
            warnings.warn(
                f"masi_entropy: log argument ≤ 0 for r={r:.4f}. "
                "r may be too large for this image.",
                RuntimeWarning, stacklevel=2,
            )
        return 0.0

    return float(np.log(argA) / (1.0 - r) + np.log(argB) / (1.0 - r))


def masi_r_adaptive(hist: np.ndarray, img_region: np.ndarray) -> float:
    """
    Estimate the MASI ``r`` parameter adaptively.

    Heuristic: ``r = argmax(hist) / max(pixel_value)``.  This positions
    ``r`` relative to the dominant intensity mode of the image.

    Parameters
    ----------
    hist : np.ndarray, shape (256,)
        Pre-computed histogram.
    img_region : np.ndarray
        Pixel array (any shape, grayscale).

    Returns
    -------
    float in (0, 1]
    """
    f_max = int(img_region.max())
    return int(hist.argmax()) / f_max if f_max > 0 else 1.0
