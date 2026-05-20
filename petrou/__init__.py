"""
petrou
======
Image thresholding and segmentation optimization.

Import everything from here — no need to touch submodules for day-to-day use::

    from petrou import (
        SearchSpace,
        find_otsu_threshold, find_tsallis_threshold, find_masi_threshold,
        multilevel_otsu, multilevel_tsallis, multilevel_masi,
        PSO, InertiaRegistry, simulated_annealing, exhaustive_search,
        jaccard_index, dice_coefficient,
    )
"""


# Exceptions
from petrou.exceptions import (
    PetrouError,
    InvalidSearchSpaceError,
    EmptyHistogramError,
    OptimizationError,
    IncompatibleStrategyError,
)

# Optimization
from petrou.optimization import (
    SearchSpace,
    VariableDef,
    simulated_annealing,
    exhaustive_search,
    PSO,
    InertiaRegistry,
)

# Objectives
from petrou.objectives import (
    otsu_criterion,
    tsallis_entropy,
    tsallis_q_automatic,
    masi_entropy,
    masi_r_adaptive,
)

# Thresholding
from petrou.thresholding import (
    ThresholdResult,
    find_otsu_threshold,
    find_tsallis_threshold,
    find_masi_threshold,
    multilevel_otsu,
    multilevel_tsallis,
    multilevel_masi,
)

# Metrics
from petrou.metrics import (
    detect_gt_polarity,
    misclassification_error,
    false_positive_rate,
    false_negative_rate,
    jaccard_index,
    dice_coefficient,
    dice_loss,
)

# Analysis
from petrou.analysis import line_profile_bresenham

__version__ = "0.1.0"

__all__ = [
    # exceptions
    "PetrouError", "InvalidSearchSpaceError", "EmptyHistogramError",
    "OptimizationError", "IncompatibleStrategyError",
    # optimization
    "SearchSpace", "VariableDef",
    "simulated_annealing", "exhaustive_search",
    "PSO", "InertiaRegistry",
    # objectives
    "otsu_criterion",
    "tsallis_entropy", "tsallis_q_automatic",
    "masi_entropy", "masi_r_adaptive",
    # thresholding
    "ThresholdResult",
    "find_otsu_threshold", "find_tsallis_threshold", "find_masi_threshold",
    "multilevel_otsu", "multilevel_tsallis", "multilevel_masi",
    # metrics
    "detect_gt_polarity",
    "misclassification_error", "false_positive_rate", "false_negative_rate",
    "jaccard_index", "dice_coefficient", "dice_loss",
    # analysis
    "line_profile_bresenham",
]
