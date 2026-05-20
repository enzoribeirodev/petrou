"""
petrou.analysis.profile
========================
Intensity profile extraction along a line segment (Bresenham algorithm).

OpenCV is an **optional** dependency used only for generating the
visualisation overlay.  The intensity profile itself is always returned.
Install the optional dependency with::

    pip install opencv-python
"""

from __future__ import annotations

import random as _random
import warnings

import numpy as np

__all__ = ["line_profile_bresenham"]

try:
    import cv2 as _cv2
    _CV2 = True
except ImportError:
    _cv2 = None
    _CV2 = False


def line_profile_bresenham(
    gray_image: np.ndarray,
    pt1: tuple[int, int] | None = None,
    pt2: tuple[int, int] | None = None,
) -> tuple[list[int], np.ndarray | None, tuple[int, int], tuple[int, int]]:
    """
    Extract the pixel-intensity profile along a line using Bresenham's algorithm.

    Parameters
    ----------
    gray_image : np.ndarray, shape (H, W)
        Grayscale image. Must be 2-D.
    pt1 : (x, y) or None
        Start point. Random if ``None``.
    pt2 : (x, y) or None
        End point. Random if ``None``.

    Returns
    -------
    intensities : list[int]
        Pixel values sampled along the line.
    vis_image : np.ndarray (H, W, 3) BGR or None
        Copy of the input in colour with sampled pixels marked in red.
        ``None`` when ``opencv-python`` is not installed.
    pt1 : (x, y)
        Effective start point.
    pt2 : (x, y)
        Effective end point.

    Raises
    ------
    ValueError
        If ``gray_image`` is not 2-D.
    """
    if gray_image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {gray_image.shape}.")

    h, w = gray_image.shape
    if pt1 is None:
        pt1 = (_random.randint(0, w - 1), _random.randint(0, h - 1))
    if pt2 is None:
        pt2 = (_random.randint(0, w - 1), _random.randint(0, h - 1))

    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    x, y = x1, y1
    intensities: list[int] = []
    coords: list[tuple[int, int]] = []

    while True:
        if 0 <= x < w and 0 <= y < h:
            intensities.append(int(gray_image[y, x]))
            coords.append((x, y))
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    vis: np.ndarray | None = None
    if _CV2:
        vis = _cv2.cvtColor(gray_image, _cv2.COLOR_GRAY2BGR)
        for cx, cy in coords:
            vis[cy, cx] = [0, 0, 255]
    else:
        warnings.warn(
            "opencv-python not installed — vis_image is None. "
            "Run: pip install opencv-python",
            RuntimeWarning, stacklevel=2,
        )

    return intensities, vis, pt1, pt2
