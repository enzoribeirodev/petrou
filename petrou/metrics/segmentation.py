"""
petrou.metrics.segmentation
============================
Quantitative evaluation metrics for binary image segmentation.

All functions compare a **segmented** image against a **ground-truth**
image, both in binary form (pixel == 0 → background, pixel != 0 →
foreground), and return a scalar score.

Polarity detection
------------------
Some ground-truth datasets store labels with the convention **inverted**
relative to what petrou expects (white = background, black = foreground).
Pass ``invert_gt=True`` to flip the ground-truth polarity before computing
any metric.  You can also call :func:`detect_gt_polarity` to let the
library guess which convention the ground truth uses.

Notation
--------
::

    Bt  foreground mask of the segmented image   (pixel != 0)
    Bo  foreground mask of the ground truth       (pixel != 0)
    TP  |Bt ∩ Bo|   true positives
    FP  Bt minus Bo  false positives
    FN  Bo minus Bt  false negatives
    TN  total − TP − FP − FN
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "detect_gt_polarity",
    "misclassification_error",
    "false_positive_rate",
    "false_negative_rate",
    "jaccard_index",
    "dice_coefficient",
    "dice_loss",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")


def _masks(seg: np.ndarray, gt: np.ndarray, invert_gt: bool):
    """Return (Bt, Bo, background_t, background_o) boolean arrays."""
    Bt = seg != 0
    Bo = (gt == 0) if invert_gt else (gt != 0)
    return Bt, Bo, ~Bt, ~Bo


def _tp_fp_fn(Bt, Bo):
    TP = int((Bt & Bo).sum())
    FP = int((Bt & ~Bo).sum())
    FN = int((~Bt & Bo).sum())
    return TP, FP, FN


# ---------------------------------------------------------------------------
# Polarity detection
# ---------------------------------------------------------------------------

def detect_gt_polarity(
    segmented_img: np.ndarray,
    ground_truth_img: np.ndarray,
) -> bool:
    """
    Heuristically determine whether the ground-truth polarity is inverted.

    The function computes the Dice coefficient under both polarities and
    returns ``True`` (inverted) when inverting the ground truth yields a
    higher score.

    Use this when you are unsure whether the ground-truth dataset labels
    foreground as white (standard) or black (inverted).

    Parameters
    ----------
    segmented_img : np.ndarray
        Binary segmented image.
    ground_truth_img : np.ndarray
        Ground-truth binary image (polarity unknown).

    Returns
    -------
    bool
        ``True``  — ground truth is inverted (pass ``invert_gt=True`` to metrics).
        ``False`` — ground truth uses standard polarity.

    Examples
    --------
    >>> inv = detect_gt_polarity(seg, gt)
    >>> ji = jaccard_index(seg, gt, invert_gt=inv)
    """
    _check(segmented_img, ground_truth_img)
    dc_normal = dice_coefficient(segmented_img, ground_truth_img, invert_gt=False)
    dc_invert = dice_coefficient(segmented_img, ground_truth_img, invert_gt=True)
    return dc_invert > dc_normal


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def misclassification_error(
    segmented_img: np.ndarray,
    ground_truth_img: np.ndarray,
    *,
    invert_gt: bool = False,
) -> float:
    """
    Misclassification error (ME).

    ``ME = 1 − (TP + TN) / total``

    Lower is better. ME = 0 → perfect segmentation.

    Parameters
    ----------
    segmented_img : np.ndarray
    ground_truth_img : np.ndarray
    invert_gt : bool
        When ``True``, invert the ground-truth polarity before evaluation.
        Use :func:`detect_gt_polarity` to detect the right value automatically.

    Returns
    -------
    float in [0, 1]
    """
    _check(segmented_img, ground_truth_img)
    Bt, Bo, Bg_t, Bg_o = _masks(segmented_img, ground_truth_img, invert_gt)
    TP = int((Bt & Bo).sum())
    TN = int((Bg_t & Bg_o).sum())
    total = segmented_img.size
    return 1.0 - (TP + TN) / total


def false_positive_rate(
    segmented_img: np.ndarray,
    ground_truth_img: np.ndarray,
    *,
    invert_gt: bool = False,
) -> float:
    """
    False positive rate: FPR = FP / |background in GT|.

    Parameters
    ----------
    segmented_img : np.ndarray
    ground_truth_img : np.ndarray
    invert_gt : bool

    Returns
    -------
    float in [0, 1]
    """
    _check(segmented_img, ground_truth_img)
    Bt, Bo, _, Bg_o = _masks(segmented_img, ground_truth_img, invert_gt)
    FP = int((Bt & Bg_o).sum())
    denom = int(Bg_o.sum())
    return FP / denom if denom > 0 else 0.0


def false_negative_rate(
    segmented_img: np.ndarray,
    ground_truth_img: np.ndarray,
    *,
    invert_gt: bool = False,
) -> float:
    """
    False negative rate: FNR = FN / |foreground in GT|.

    Parameters
    ----------
    segmented_img : np.ndarray
    ground_truth_img : np.ndarray
    invert_gt : bool

    Returns
    -------
    float in [0, 1]
    """
    _check(segmented_img, ground_truth_img)
    Bt, Bo, Bg_t, _ = _masks(segmented_img, ground_truth_img, invert_gt)
    FN = int((Bg_t & Bo).sum())
    denom = int(Bo.sum())
    return FN / denom if denom > 0 else 0.0


def jaccard_index(
    segmented_img: np.ndarray,
    ground_truth_img: np.ndarray,
    *,
    invert_gt: bool = False,
) -> float:
    """
    Jaccard similarity index (Intersection over Union).

    ``JI = TP / (TP + FP + FN)``

    Higher is better. JI = 1 → perfect segmentation.

    Parameters
    ----------
    segmented_img : np.ndarray
    ground_truth_img : np.ndarray
    invert_gt : bool

    Returns
    -------
    float in [0, 1]
    """
    _check(segmented_img, ground_truth_img)
    Bt, Bo, _, _ = _masks(segmented_img, ground_truth_img, invert_gt)
    TP, FP, FN = _tp_fp_fn(Bt, Bo)
    denom = TP + FP + FN
    return TP / denom if denom > 0 else 0.0


def dice_coefficient(
    segmented_img: np.ndarray,
    ground_truth_img: np.ndarray,
    *,
    invert_gt: bool = False,
) -> float:
    """
    Dice similarity coefficient (F1 score).

    ``DC = 2·TP / (2·TP + FP + FN)``

    Higher is better. DC = 1 → perfect segmentation.

    Parameters
    ----------
    segmented_img : np.ndarray
    ground_truth_img : np.ndarray
    invert_gt : bool

    Returns
    -------
    float in [0, 1]
    """
    _check(segmented_img, ground_truth_img)
    Bt, Bo, _, _ = _masks(segmented_img, ground_truth_img, invert_gt)
    TP, FP, FN = _tp_fp_fn(Bt, Bo)
    denom = 2 * TP + FP + FN
    return (2 * TP) / denom if denom > 0 else 0.0


def dice_loss(
    segmented_img: np.ndarray,
    ground_truth_img: np.ndarray,
    *,
    invert_gt: bool = False,
) -> float:
    """
    Dice loss = 1 − Dice coefficient. Lower is better.

    Parameters
    ----------
    segmented_img : np.ndarray
    ground_truth_img : np.ndarray
    invert_gt : bool

    Returns
    -------
    float in [0, 1]
    """
    return 1.0 - dice_coefficient(segmented_img, ground_truth_img, invert_gt=invert_gt)
