"""Feature parsing and categorical design-matrix utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class CategoricalInputDecl:
    """Declare one categorical predictor and its optional ordered levels."""

    name: str
    levels: list[str] | None = None


KNOWN_TRANSFORMATIONS = {
    # Accepted descriptive labels.
    "quadratic",
    "inverse",
    "log",
    "log1p",
    "square",
    "cubic",
    "cube",
    "reciprocal",
    "abs",
    "logarithmic",
    # Short labels from DEFAULT_TRANSFORM_LIBRARY (TransformDef.label values).
    "sq",  # quadratic  x**2
    "inv",  # inverse    1/x
    "sqrt",  # square root
}


def _strip_outer_brackets(text: str) -> str:
    """Strip one matching outer bracket pair from a feature label."""
    text = str(text).strip()
    if text.startswith("[") and text.endswith("]"):
        return text[1:-1].strip()
    return text


def _split_interaction(name: str) -> tuple[str | None, str | None]:
    """Split a second-order interaction into its component terms."""
    cleaned = _strip_outer_brackets(name)
    if "*" not in cleaned:
        return None, None
    left, right = cleaned.split("*", 1)
    return left.strip(), right.strip()


def _split_transformation(
    name: str,
    known_transformations: set[str] | None = None,
) -> tuple[str | None, str | None]:
    """Split a transformed feature into its base feature and transform label."""
    known_transformations = known_transformations or KNOWN_TRANSFORMATIONS
    cleaned = _strip_outer_brackets(name)
    for suffix in sorted(known_transformations, key=len, reverse=True):
        marker = f"_{suffix}"
        if cleaned.endswith(marker):
            return cleaned[: -len(marker)].strip(), suffix
    return None, None


def parse_selected_input_structure(
    feature_names: Iterable[str],
    known_transformations: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify selected inputs into first-order, interaction, and nonlinear terms.

    Parameters
    ----------
    feature_names
        Selected feature names in their original order.
    known_transformations
        Optional set of recognized transformation suffixes.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Structure table, type summary table, and transformation summary table.
    """
    records: list[dict[str, object]] = []
    known_transformations = known_transformations or KNOWN_TRANSFORMATIONS

    for raw_name in feature_names:
        name = str(raw_name)
        left, right = _split_interaction(name)
        if left is not None:
            records.append(
                {
                    "feature_name": name,
                    "input_type": "second_order",
                    "term_1": left,
                    "term_2": right,
                    "transformation": None,
                }
            )
            continue

        base, transformation = _split_transformation(
            name,
            known_transformations=known_transformations,
        )
        if base is not None:
            records.append(
                {
                    "feature_name": name,
                    "input_type": "nonlinear_transformation",
                    "term_1": base,
                    "term_2": None,
                    "transformation": transformation,
                }
            )
            continue

        records.append(
            {
                "feature_name": name,
                "input_type": "first_order",
                "term_1": name,
                "term_2": None,
                "transformation": None,
            }
        )

    structure_df = pd.DataFrame.from_records(records)
    type_summary_df = (
        structure_df.groupby("input_type", dropna=False)
        .size()
        .rename("n_inputs")
        .reset_index()
        .sort_values("input_type")
        .reset_index(drop=True)
    )
    transformation_summary_df = (
        structure_df.dropna(subset=["transformation"])
        .groupby("transformation", dropna=False)
        .size()
        .rename("n_inputs")
        .reset_index()
        .sort_values(["n_inputs", "transformation"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return structure_df, type_summary_df, transformation_summary_df


def canonical_module_from_factor_name(name: str) -> str:
    """Return the coarse module prefix implied by a factor name.

    Parameters
    ----------
    name
        Full factor or feature label.

    Returns
    -------
    str
        The left-hand scope before the first dot for scoped names, or ``Unscoped``
        otherwise.
    """
    name = str(name)
    if "." not in name:
        return "Unscoped"
    return name.split(".", 1)[0].strip() or "Unscoped"


# ---------------------------------------------------------------------------
# Design-matrix construction with categorical main effects and interactions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignMatrixSpec:
    """Specification for design-matrix construction with optional categoricals.

    Parameters
    ----------
    categorical_inputs:
        Declared categorical/block predictors. Each entry may
        optionally carry an explicit level list.
    interaction_pairs:
        Ordered pairs ``(categorical_name, scalar_name)`` for which a
        categorical×scalar product column should be added.  The categorical
        side is encoded with the same indicator scheme as the main effects.
    drop_first:
        When ``True`` (default), use drop-first dummy coding to avoid perfect
        multicollinearity.  When ``False``, emit all levels as indicators
        (useful for explicit contrast matrices or tests).
    """

    categorical_inputs: tuple[CategoricalInputDecl, ...] = field(default_factory=tuple)  # type: ignore[assignment]
    interaction_pairs: tuple[tuple[str, str], ...] = ()
    drop_first: bool = True


@dataclass(frozen=True)
class DesignMatrixResult:
    """Materialized design matrix and its ordered column catalog.

    Attributes
    ----------
    matrix:
        The assembled design matrix.  Scalar columns retain their original
        dtype; indicator columns are ``uint8``; interaction columns are
        ``float64``.
    ordered_columns:
        Stable ordered tuple of all column names in ``matrix``.
    categorical_columns:
        Column names introduced by categorical main-effect encoding.
    interaction_columns:
        Column names introduced by categorical×scalar interactions.
    """

    matrix: pd.DataFrame
    ordered_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    interaction_columns: tuple[str, ...]


def _indicator_columns(
    series: pd.Series,
    name: str,
    levels: list[str] | None,
    *,
    drop_first: bool,
) -> pd.DataFrame:
    """Encode a single categorical series as indicator columns.

    Parameters
    ----------
    series:
        Raw categorical/boolean column values.
    name:
        Column name in the source DataFrame (used for indicator naming).
    levels:
        Explicit ordered levels.  When ``None``, levels are inferred from the
        unique sorted values of *series*.
    drop_first:
        Whether to omit the first level (reference category).

    Returns
    -------
    pandas.DataFrame
        One column per retained level named ``"{name}__{level}"``.
    """
    if series.isna().any():
        raise ValueError(f"categorical input {name!r} must not contain missing values")
    observed = set(series.astype(str))
    if levels is None:
        levels = sorted(observed)
    else:
        levels = [str(level) for level in levels]
        if not levels:
            raise ValueError(f"categorical input {name!r} must declare at least one level")
        if len(levels) != len(set(levels)):
            raise ValueError(f"categorical input {name!r} contains duplicate declared levels")
        unknown = sorted(observed.difference(levels))
        if unknown:
            raise ValueError(f"unknown levels for categorical input {name!r}: {unknown}")
    if drop_first and len(levels) > 1:
        levels = levels[1:]
    result: dict[str, pd.Series] = {}
    for lvl in levels:
        col_name = f"{name}__{lvl}"
        result[col_name] = (series.astype(str) == str(lvl)).astype("uint8")
    return pd.DataFrame(result, index=series.index)


def build_design_matrix(
    X: pd.DataFrame,
    spec: DesignMatrixSpec | None = None,
) -> DesignMatrixResult:
    """Construct a design matrix with optional categorical main effects and interactions.

    When *spec* is ``None`` or contains no categorical inputs and no interaction
    pairs, the output ``matrix`` is byte-for-byte identical to the input *X*
    (same dtypes, same column order, same index).

    Parameters
    ----------
    X:
        Input feature matrix.  Must contain a column for every name mentioned
        in *spec*.
    spec:
        Design-matrix specification.  Pass ``None`` or an empty
        :class:`DesignMatrixSpec` to reproduce the no-categorical baseline.

    Returns
    -------
    DesignMatrixResult
        Assembled design matrix, ordered column catalog, and column provenance.

    Raises
    ------
    KeyError
        If a column referenced in *spec* is absent from *X*.
    ValueError
        If a scalar column referenced in an interaction pair is also declared
        as a categorical input.
    """
    if spec is None:
        spec = DesignMatrixSpec()

    declared_names = [decl.name for decl in spec.categorical_inputs]
    if len(declared_names) != len(set(declared_names)):
        raise ValueError("categorical input declarations must have unique names")
    cat_names = set(declared_names)
    undeclared_interactions = sorted(
        {cat_col for cat_col, _ in spec.interaction_pairs if cat_col not in cat_names}
    )
    if undeclared_interactions:
        raise ValueError(
            f"interaction categorical columns {undeclared_interactions} must be declared "
            "in categorical_inputs"
        )
    scalar_cols = [c for c in X.columns if c not in cat_names]

    # Validate all referenced columns exist.
    for decl in spec.categorical_inputs:
        if decl.name not in X.columns:
            raise KeyError(f"build_design_matrix: categorical column {decl.name!r} not in X")
    for cat_col, scalar_col in spec.interaction_pairs:
        if cat_col not in X.columns:
            raise KeyError(f"build_design_matrix: categorical column {cat_col!r} not in X")
        if scalar_col not in X.columns:
            raise KeyError(f"build_design_matrix: scalar column {scalar_col!r} not in X")
        if scalar_col in cat_names:
            raise ValueError(
                f"build_design_matrix: interaction scalar {scalar_col!r} is also "
                "declared as a categorical input."
            )

    # Fast path: no categoricals, no interactions → return X unchanged.
    if not spec.categorical_inputs and not spec.interaction_pairs:
        return DesignMatrixResult(
            matrix=X.copy(),
            ordered_columns=tuple(X.columns),
            categorical_columns=(),
            interaction_columns=(),
        )

    # Build parts: scalars → categorical indicators → interaction columns.
    parts: list[pd.DataFrame] = [X[scalar_cols].copy()]
    categorical_col_names: list[str] = []
    interaction_col_names: list[str] = []

    # Precompute indicator frames keyed by categorical name for reuse in interactions.
    indicator_frames: dict[str, pd.DataFrame] = {}
    for decl in spec.categorical_inputs:
        ind = _indicator_columns(
            X[decl.name],
            name=decl.name,
            levels=decl.levels,
            drop_first=spec.drop_first,
        )
        indicator_frames[decl.name] = ind
        parts.append(ind)
        categorical_col_names.extend(ind.columns.tolist())

    # Add interaction columns: each indicator column × scalar value.
    for cat_col, scalar_col in spec.interaction_pairs:
        ind = indicator_frames[cat_col]
        scalar_values = X[scalar_col]
        for ind_col in ind.columns:
            ix_col_name = f"{ind_col}_x_{scalar_col}"
            parts.append(
                pd.DataFrame(
                    {ix_col_name: ind[ind_col].astype(float) * scalar_values},
                    index=X.index,
                )
            )
            interaction_col_names.append(ix_col_name)

    matrix = pd.concat(parts, axis=1)
    return DesignMatrixResult(
        matrix=matrix,
        ordered_columns=tuple(matrix.columns),
        categorical_columns=tuple(categorical_col_names),
        interaction_columns=tuple(interaction_col_names),
    )


def resolve_spec_levels(X: pd.DataFrame, spec: DesignMatrixSpec) -> DesignMatrixSpec:
    """Return a new DesignMatrixSpec with all categorical levels resolved from *X*.

    For each :class:`CategoricalInputDecl` whose ``levels``
    is ``None``, the sorted unique non-null values in the corresponding column of
    *X* are used as the explicit level list.  Entries that already carry an
    explicit level list are kept unchanged.

    This should be called at fit time so the stored spec reproduces the same
    indicator columns at predict time regardless of the prediction batch size.

    Parameters
    ----------
    X:
        Training input frame.  Must contain every column named in *spec*.
    spec:
        Design-matrix specification, possibly with ``levels=None`` entries.

    Returns
    -------
    DesignMatrixSpec
        Equivalent spec with all level lists fully populated.
    """
    resolved: list[CategoricalInputDecl] = []
    for decl in spec.categorical_inputs:
        if decl.levels is None:
            levels = sorted(X[decl.name].dropna().unique().astype(str).tolist())
            resolved.append(CategoricalInputDecl(name=decl.name, levels=levels))
        else:
            resolved.append(decl)
    return DesignMatrixSpec(
        categorical_inputs=tuple(resolved),
        interaction_pairs=spec.interaction_pairs,
        drop_first=spec.drop_first,
    )
