"""
petrou.exceptions
=================
Semantic exception hierarchy. Every petrou error inherits from ``PetrouError``,
so callers can ``except PetrouError`` to catch everything, or catch a specific
subclass to handle individual failure modes.
"""

__all__ = [
    "PetrouError",
    "InvalidSearchSpaceError",
    "EmptyHistogramError",
    "OptimizationError",
    "IncompatibleStrategyError",
]


class PetrouError(Exception):
    """Base class for all petrou errors."""


class InvalidSearchSpaceError(PetrouError, ValueError):
    """Malformed SearchSpace variable (wrong type string, lo >= hi, step <= 0, missing key)."""


class EmptyHistogramError(PetrouError, ValueError):
    """Image region contains no pixels — histogram sums to zero."""


class OptimizationError(PetrouError, RuntimeError):
    """Objective function raised an exception during optimization."""


class IncompatibleStrategyError(PetrouError, ValueError):
    """Logically invalid combination of strategies or parameters."""
