"""
petrou.optimization.search_space
=================================
Unified N-dimensional search-space definition and neighbourhood generation.

Every optimizer in petrou (SA, PSO, Exhaustive, and any you add later)
receives a ``SearchSpace`` instead of raw bounds arrays.  This one object
encodes variable names, types, bounds, and perturbation scales so that no
optimizer ever needs to branch on variable types or write its own neighbour
function.

Internal representation
-----------------------
Internally, every state is an ``np.ndarray`` of ``float64``, even for integer
variables.  This keeps optimizers uniform — they never see type distinctions.
Type semantics surface in exactly three places:

* ``initial_state``  — integer dims are rounded before the state is returned.
* ``neighbour``      — integer dims get a discrete uniform perturbation and
                       are rounded back.
* ``decode``         — converts the raw float64 vector to typed Python values.

Variable schema
---------------
Each variable is a plain ``dict`` (or a :class:`VariableDef`) with four keys::

    name   str             identifier; becomes the key in ``decode()`` results
    type   "float"|"int"   governs perturbation style and decode rounding
    bounds (lo, hi)        inclusive interval; lo < hi required
    step   float|int       perturbation scale
                           float  →  σ of Gaussian delta  N(0, step²)
                           int    →  half-width of discrete uniform delta

Examples
--------
>>> space = SearchSpace([
...     {"name": "q", "type": "float", "bounds": (0.01, 3.0), "step": 0.05},
...     {"name": "t", "type": "int",   "bounds": (1, 254),    "step": 5},
... ])
>>> rng = np.random.default_rng(0)
>>> state  = space.initial_state(rng)    # np.ndarray shape (2,)
>>> neigh  = space.neighbour(state, rng) # perturbed copy
>>> values = space.decode(state)         # {"q": 1.52, "t": 127}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from petrou.exceptions import InvalidSearchSpaceError

__all__ = ["VariableDef", "SearchSpace"]


@dataclass(frozen=True)
class VariableDef:
    """Immutable description of one optimization variable."""

    name: str
    type: str           # "float" | "int"
    bounds: tuple[float, float]
    step: float

    def __post_init__(self) -> None:
        if self.type not in {"float", "int"}:
            raise InvalidSearchSpaceError(
                f"Variable '{self.name}': type must be 'float' or 'int', got '{self.type}'."
            )
        lo, hi = self.bounds
        if lo >= hi:
            raise InvalidSearchSpaceError(
                f"Variable '{self.name}': lower bound ({lo}) must be strictly less than upper ({hi})."
            )
        if self.step <= 0:
            raise InvalidSearchSpaceError(
                f"Variable '{self.name}': step must be positive, got {self.step}."
            )


class SearchSpace:
    """
    N-dimensional mixed-type search space shared by all petrou optimizers.

    Parameters
    ----------
    variables : list[dict | VariableDef]
        Ordered list of variable definitions.

    Attributes
    ----------
    ndim : int
    lower : np.ndarray, shape (ndim,)
    upper : np.ndarray, shape (ndim,)
    """

    def __init__(self, variables: list[dict[str, Any] | VariableDef]) -> None:
        if not variables:
            raise InvalidSearchSpaceError("SearchSpace requires at least one variable.")

        self._vars: list[VariableDef] = []
        for v in variables:
            if isinstance(v, VariableDef):
                self._vars.append(v)
            elif isinstance(v, dict):
                try:
                    self._vars.append(
                        VariableDef(
                            name=v["name"],
                            type=v["type"],
                            bounds=tuple(v["bounds"]),  # type: ignore[arg-type]
                            step=float(v["step"]),
                        )
                    )
                except KeyError as exc:
                    raise InvalidSearchSpaceError(
                        f"Variable dict is missing required key: {exc}."
                    ) from exc
            else:
                raise InvalidSearchSpaceError(
                    f"Expected dict or VariableDef, got {type(v).__name__}."
                )

        self.ndim: int = len(self._vars)
        self.lower = np.array([v.bounds[0] for v in self._vars], dtype=np.float64)
        self.upper = np.array([v.bounds[1] for v in self._vars], dtype=np.float64)
        self._steps = np.array([v.step for v in self._vars], dtype=np.float64)
        self._is_int = np.array([v.type == "int" for v in self._vars], dtype=bool)

    # ------------------------------------------------------------------ state

    def initial_state(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Uniformly sample a random starting point within bounds.

        Integer dimensions are rounded so the initial state is already valid.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Supply an explicit generator for reproducibility.

        Returns
        -------
        np.ndarray, shape (ndim,), dtype float64
        """
        if rng is None:
            rng = np.random.default_rng()
        state = rng.uniform(self.lower, self.upper)
        state[self._is_int] = np.round(state[self._is_int])
        return np.clip(state, self.lower, self.upper)

    # ------------------------------------------------------------------ neighbourhood

    def neighbour(
        self,
        state: np.ndarray,
        rng: np.random.Generator,
        perturbation: str = "independent",
    ) -> np.ndarray:
        """
        Generate a neighbouring state.

        Per-dimension perturbation rules:

        * ``float`` — Gaussian delta: ``δ ~ N(0, step²)``, clipped to bounds.
        * ``int``   — Discrete uniform: ``δ ∈ {-step, …, -1, +1, …, +step}``
          (zero excluded to guarantee movement), clipped and rounded.

        Parameters
        ----------
        state : np.ndarray, shape (ndim,)
        rng : np.random.Generator
        perturbation : {"independent", "single"}
            ``"independent"`` (default) — all dims perturbed simultaneously.
            Best for low-dimensional correlated spaces (e.g. q and t together).

            ``"single"`` — one randomly chosen dim perturbed per call.
            Better for high-dimensional or independent spaces.

        Returns
        -------
        np.ndarray, shape (ndim,), dtype float64
        """
        new = state.copy()
        dims = (
            [int(rng.integers(0, self.ndim))]
            if perturbation == "single"
            else list(range(self.ndim))
        )
        for i in dims:
            v = self._vars[i]
            if v.type == "float":
                delta = rng.normal(0.0, self._steps[i])
                new[i] = float(np.clip(state[i] + delta, self.lower[i], self.upper[i]))
            else:
                step_i = max(1, int(round(self._steps[i])))
                delta = int(rng.integers(-step_i, step_i + 1))
                if delta == 0:
                    delta = 1 if rng.random() < 0.5 else -1
                new[i] = float(np.clip(round(state[i]) + delta, self.lower[i], self.upper[i]))
        return new

    # ------------------------------------------------------------------ decode

    def decode(self, state: np.ndarray) -> dict[str, int | float]:
        """
        Convert a raw float64 state vector to typed Python values.

        Call this *after* the optimizer returns to read results with correct
        types.  Inside hot objective-function lambdas, use
        ``float(state[i])`` / ``int(round(state[i]))`` directly to avoid
        per-call dict allocation.

        Parameters
        ----------
        state : np.ndarray, shape (ndim,)

        Returns
        -------
        dict[str, int | float]
            Variable name → typed value.

        Examples
        --------
        >>> space.decode(np.array([1.52, 127.0]))
        {"q": 1.52, "t": 127}
        """
        return {
            v.name: (int(round(float(state[i]))) if v.type == "int" else float(state[i]))
            for i, v in enumerate(self._vars)
        }

    def decode_value(self, state: np.ndarray, name: str) -> int | float:
        """Decode a single variable by name."""
        for i, v in enumerate(self._vars):
            if v.name == name:
                return int(round(float(state[i]))) if v.type == "int" else float(state[i])
        raise KeyError(f"Variable '{name}' not found in SearchSpace.")

    # ------------------------------------------------------------------ utilities

    def clip(self, state: np.ndarray) -> np.ndarray:
        """Clamp state to declared bounds (returns a copy)."""
        return np.clip(state, self.lower, self.upper)

    def contains(self, state: np.ndarray) -> bool:
        """True if every dimension is within bounds."""
        return bool(np.all(state >= self.lower) and np.all(state <= self.upper))

    @property
    def pso_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) tuple — convenience property for PSO initialisation."""
        return self.lower, self.upper

    def __len__(self) -> int:
        return self.ndim

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{v.name}({v.type}, {v.bounds}, step={v.step})" for v in self._vars
        )
        return f"SearchSpace([{parts}])"
