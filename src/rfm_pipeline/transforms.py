"""User-configurable algebraic transformation library.

Each :class:`TransformDef` encapsulates a SymPy expression string, a short
column-name label, and an optional human-readable name.  The expression is
compiled once with :func:`sympy.lambdify` into a NumPy-vectorized callable.
Invalid values (domain errors, overflows) silently become ``NaN`` rather than
raising; callers are responsible for detecting and reporting them.

Typical usage
-------------
>>> from rfm_pipeline.transforms import QUADRATIC, LOGARITHMIC
>>> col = QUADRATIC.column_name("income")  # "income_sq"
>>> vals = QUADRATIC.apply(np.array([2.0, 3.0]))  # array([4., 9.])

Defining a custom transform
----------------------------
>>> custom = TransformDef(expr="x**3", label="cube", name="cubic")

Loading from a YAML config fragment
-------------------------------------
>>> td = TransformDef.from_config({"expr": "log(x + 0.5)", "label": "log_shifted"})
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

# Deferred SymPy import so the module can be imported even in test environments
# that mock the dependency, but fail loudly at first *use*.
_SYMPY_AVAILABLE: bool | None = None


def _require_sympy() -> Any:
    """Return the sympy module, raising ImportError with a helpful message."""
    try:
        import sympy  # noqa: PLC0415

        return sympy
    except ImportError as exc:
        raise ImportError(
            "SymPy is required for algebraic transform compilation. "
            "Install it with: pip install 'sympy>=1.12'"
        ) from exc


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformDef:
    """An algebraic transformation expressed as a SymPy string.

    Parameters
    ----------
    expr
        SymPy-parseable expression in the single variable ``x``.
        Examples: ``"x**2"``, ``"log(x + 1)"``, ``"1/x"``, ``"sqrt(x)"``,
        ``"exp(x)"``, ``"x**3"``, ``"log(x + 0.5)"``.
    label
        Short suffix appended to the base-feature name to form the derived
        column name (e.g. ``"sq"`` → ``"income_sq"``).  Must be a valid
        Python identifier fragment (no spaces).
    name
        Optional human-readable name for documentation and reporting.
    """

    expr: str
    label: str
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate expr and label on construction."""
        if not self.expr:
            raise ValueError("TransformDef.expr must not be empty.")
        if not self.label or not self.label.replace("_", "").isalnum():
            raise ValueError(
                f"TransformDef.label must be a non-empty alphanumeric/underscore string, "
                f"got {self.label!r}."
            )

    # ------------------------------------------------------------------
    # Column naming
    # ------------------------------------------------------------------

    def column_name(self, base_feature: str) -> str:
        """Return the derived column name for this transform on *base_feature*.

        The convention is ``"{base_feature}_{self.label}"``.

        Parameters
        ----------
        base_feature
            Name of the source feature column.

        Returns
        -------
        str
            Derived column name, e.g. ``"income_sq"``.
        """
        return f"{base_feature}_{self.label}"

    # ------------------------------------------------------------------
    # Compilation and evaluation
    # ------------------------------------------------------------------

    @cached_property
    def _compiled(self):  # type: ignore[type-arg]
        """Compile the SymPy expression to a NumPy-vectorized callable."""
        sp = _require_sympy()
        x = sp.Symbol("x")
        parsed = sp.sympify(self.expr)
        return sp.lambdify([x], parsed, modules="numpy")

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply this transform element-wise; invalid results become ``NaN``.

        Overflow, division by zero, and domain errors are suppressed via
        :func:`numpy.errstate`.  Infinite results are also replaced with
        ``NaN``.  Callers should call :meth:`valid_fraction` if they want to
        detect columns that are mostly invalid.

        Parameters
        ----------
        values
            1-D (or any-shape) numeric array of raw feature values.

        Returns
        -------
        np.ndarray
            Transformed values of the same shape, with ``NaN`` where the
            transform is undefined.
        """
        arr = np.asarray(values, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = np.asarray(self._compiled(arr), dtype=float)
        result = np.where(np.isfinite(result), result, np.nan)
        return result

    def valid_fraction(self, values: np.ndarray) -> float:
        """Return the fraction of *values* for which this transform is finite.

        Parameters
        ----------
        values
            Raw feature values to probe.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``; ``1.0`` means all values are valid.
        """
        result = self.apply(values)
        n = result.size
        if n == 0:
            return 0.0
        return float(np.isfinite(result).sum()) / n

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> TransformDef:
        """Construct a :class:`TransformDef` from a YAML-deserialized dict.

        Expected keys
        ~~~~~~~~~~~~~
        ``expr`` (required)
            SymPy expression string.
        ``label`` (required)
            Column-name suffix.
        ``name`` (optional)
            Human-readable name.

        Parameters
        ----------
        data
            Dictionary with at least ``expr`` and ``label`` keys.

        Returns
        -------
        TransformDef

        Raises
        ------
        KeyError
            If ``expr`` or ``label`` is missing.
        """
        expr = str(data["expr"])
        label = str(data["label"])
        name = str(data["name"]) if "name" in data and data["name"] is not None else None
        return cls(expr=expr, label=label, name=name)

    def to_config(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for YAML output."""
        d: dict[str, Any] = {"expr": self.expr, "label": self.label}
        if self.name is not None:
            d["name"] = self.name
        return d

    def __repr__(self) -> str:
        """Return a concise string representation."""
        name_part = f", name={self.name!r}" if self.name else ""
        return f"TransformDef(expr={self.expr!r}, label={self.label!r}{name_part})"


# ---------------------------------------------------------------------------
# Standard transform library
# ---------------------------------------------------------------------------

#: Quadratic (x²).  Always domain-valid.
QUADRATIC = TransformDef(expr="x**2", label="sq", name="quadratic")

#: Natural log of (x + 1).  Valid when x > −1.
LOGARITHMIC = TransformDef(expr="log(x + 1)", label="log1p", name="logarithmic")

#: Multiplicative inverse (1/x).  Valid when x ≠ 0.
INVERSE = TransformDef(expr="1/x", label="inv", name="inverse")

#: Square root.  Valid when x ≥ 0.
SQRT = TransformDef(expr="sqrt(x)", label="sqrt", name="square_root")

#: Natural exponential (eˣ).  Always domain-valid; may overflow for large x.
EXPONENTIAL = TransformDef(expr="exp(x)", label="exp", name="exponential")

#: Default library used when no ``transform_library`` is specified in config.
#: Matches the families reported in the manuscript (§7): quadratic,
#: logarithmic, inverse, and exponential, plus square root.
DEFAULT_TRANSFORM_LIBRARY: list[TransformDef] = [
    QUADRATIC,
    LOGARITHMIC,
    INVERSE,
    SQRT,
    EXPONENTIAL,
]

# Convenience lookup by label for use in tests and downstream code.
_LIBRARY_BY_LABEL: dict[str, TransformDef] = {t.label: t for t in DEFAULT_TRANSFORM_LIBRARY}

# ---------------------------------------------------------------------------
# Warning helpers
# ---------------------------------------------------------------------------


def warn_nan_transforms(
    nan_report: dict[str, list[str]],
    *,
    stacklevel: int = 2,
) -> None:
    """Emit :class:`UserWarning` messages for features with NaN-producing transforms.

    Parameters
    ----------
    nan_report
        Mapping from transform label to list of base-feature names that
        produced at least one NaN for that transform.
    stacklevel
        Passed to :func:`warnings.warn` to point the warning at the caller's
        frame.
    """
    for label, features in sorted(nan_report.items()):
        if not features:
            continue
        feature_list = ", ".join(features[:5])
        suffix = f" (and {len(features) - 5} more)" if len(features) > 5 else ""
        warnings.warn(
            f"Transform '{label}' produced NaN for feature(s): {feature_list}{suffix}. "
            "Rows with invalid values are set to NaN in the expanded matrix.",
            UserWarning,
            stacklevel=stacklevel,
        )
