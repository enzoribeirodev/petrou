"""petrou.objectives — thresholding objective functions."""

from petrou.objectives.variance import otsu_criterion
from petrou.objectives.entropy import (
    tsallis_entropy,
    tsallis_q_automatic,
    masi_entropy,
    masi_r_adaptive,
)

__all__ = [
    "otsu_criterion",
    "tsallis_entropy",
    "tsallis_q_automatic",
    "masi_entropy",
    "masi_r_adaptive",
]
