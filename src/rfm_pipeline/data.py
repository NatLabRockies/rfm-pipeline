"""Data preparation helpers for the reduced-form workflow."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Sealed split protocol
# ---------------------------------------------------------------------------


class SealedTestAccessError(RuntimeError):
    """Raised when the sealed test partition is read without explicit unsealing."""


class SealedSplitResult:
    """Three-way stratified split with a guarded sealed-test partition.

    Attributes
    ----------
    train
        Training partition (always accessible).
    val
        Internal-validation partition (always accessible).
    strata_balance
        DataFrame showing row counts per stratum for each partition.

    Notes
    -----
    The sealed-test partition is not a public attribute. Access it through the
    :attr:`test` property, which raises :class:`SealedTestAccessError` until
    :meth:`unseal` is called once for the single final evaluation; each unseal
    event is recorded in :attr:`unseal_log`.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        strata_balance: pd.DataFrame,
    ) -> None:
        self.train: pd.DataFrame = train
        self.val: pd.DataFrame = val
        self._test_data: pd.DataFrame = test
        self.strata_balance: pd.DataFrame = strata_balance
        self._sealed: bool = True
        self._unseal_log: list[dict[str, Any]] = []

    @property
    def test(self) -> pd.DataFrame:
        """Return the sealed-test partition, or raise if still sealed."""
        if self._sealed:
            raise SealedTestAccessError(
                "Sealed test partition accessed before unsealing. "
                "Call .unseal(reason=...) once, at final evaluation only."
            )
        return self._test_data

    def unseal(self, *, reason: str) -> None:
        """Unlock the test partition and record the event.

        Parameters
        ----------
        reason
            Human-readable justification for accessing the sealed test set.
        """
        self._sealed = False
        self._unseal_log.append(
            {
                "reason": reason,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )

    @property
    def unseal_log(self) -> list[dict[str, Any]]:
        """Return a snapshot of unseal events (read-only copy)."""
        return list(self._unseal_log)


def make_sealed_split(
    data: pd.DataFrame,
    *,
    strata_column: str,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> SealedSplitResult:
    """Create train / internal-validation / sealed-test partitions.

    Stratification is performed on ``strata_column`` so that each partition
    preserves the original stratum proportions.

    Parameters
    ----------
    data
        Input DataFrame containing ``strata_column``.
    strata_column
        Column name used for stratification.
    val_fraction
        Fraction of total rows assigned to internal validation.
    test_fraction
        Fraction of total rows assigned to the sealed test set.
    random_state
        Base seed for deterministic, reproducible splits.

    Returns
    -------
    SealedSplitResult
        Object containing ``train``, ``val``, and a guarded ``test`` partition,
        plus a ``strata_balance`` DataFrame.

    Raises
    ------
    ValueError
        Raised when ``strata_column`` is absent from ``data``, or when the
        combined val+test fraction would leave no training rows.
    """
    if strata_column not in data.columns:
        raise ValueError(
            f"strata_column {strata_column!r} not found in data columns {list(data.columns)}."
        )
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction={val_fraction} and test_fraction={test_fraction} must both be "
            "positive and sum to less than 1."
        )

    stratify = data[strata_column].astype(str)

    # First split: peel off the sealed test set.
    dev_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=test_fraction,
        random_state=random_state,
        stratify=stratify,
    )

    # Second split: divide the dev set into train and val.
    # val_fraction is expressed relative to the full dataset; recompute relative
    # to the dev subset so that the final proportions are correct.
    dev_size = len(dev_idx)
    val_fraction_of_dev = val_fraction / (1.0 - test_fraction)
    stratify_dev = stratify.iloc[dev_idx].reset_index(drop=True)

    train_sub_idx, val_sub_idx = train_test_split(
        np.arange(dev_size),
        test_size=val_fraction_of_dev,
        random_state=random_state + 1,
        stratify=stratify_dev,
    )

    train_idx = np.sort(dev_idx[np.sort(train_sub_idx)])
    val_idx = np.sort(dev_idx[np.sort(val_sub_idx)])
    test_idx = np.sort(test_idx)

    train_df = data.iloc[train_idx].copy()
    val_df = data.iloc[val_idx].copy()
    test_df = data.iloc[test_idx].copy()

    # Build strata balance summary.
    strata_order = sorted(data[strata_column].unique().tolist())
    balance_records: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for stratum in strata_order:
        balance_records["train"].append(int((train_df[strata_column] == stratum).sum()))
        balance_records["val"].append(int((val_df[strata_column] == stratum).sum()))
        balance_records["test"].append(int((test_df[strata_column] == stratum).sum()))

    strata_balance = pd.DataFrame(balance_records, index=strata_order)
    strata_balance.index.name = strata_column

    return SealedSplitResult(
        train=train_df,
        val=val_df,
        test=test_df,
        strata_balance=strata_balance,
    )


@dataclass(frozen=True)
class StandardizationBundle:
    """Container holding train-only standardization objects and transformed arrays.

    Attributes
    ----------
    x_scaler
        Fitted scaler for model inputs.
    y_scaler
        Fitted scaler for model outputs.
    x_feature_order
        Input feature order used when constructing the scaled arrays.
    y_output_order
        Output order used when constructing the scaled arrays.
    x_train_scaled
        Training inputs transformed with ``x_scaler``.
    x_holdout_scaled
        Holdout inputs transformed with training statistics only.
    y_train_scaled
        Training outputs transformed with ``y_scaler``.
    y_holdout_scaled
        Holdout outputs transformed with training statistics only.
    """

    x_scaler: StandardScaler
    y_scaler: StandardScaler
    x_feature_order: list[str]
    y_output_order: list[str]
    x_train_scaled: np.ndarray
    x_holdout_scaled: np.ndarray
    y_train_scaled: np.ndarray
    y_holdout_scaled: np.ndarray


def ensure_id_columns(frame: pd.DataFrame, id_columns: Sequence[str]) -> pd.DataFrame:
    """Ensure identifier variables are present as columns.

    Parameters
    ----------
    frame
        Input frame whose identifiers may exist in columns or the index.
    id_columns
        Identifier names required by downstream code.

    Returns
    -------
    pandas.DataFrame
        Copy of ``frame`` with identifiers exposed as columns.

    Raises
    ------
    ValueError
        Raised when one or more requested identifiers are unavailable.
    """
    missing = [column for column in id_columns if column not in frame.columns]
    if not missing:
        return frame.copy()
    if isinstance(frame.index, pd.MultiIndex):
        index_names = list(frame.index.names)
    else:
        index_names = [frame.index.name] if frame.index.name is not None else []
    if all(column in index_names for column in id_columns):
        return frame.reset_index()
    raise ValueError(f"Missing id columns {missing} in columns or index names {index_names}.")


def align_xy(X: pd.DataFrame, Y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-align input and output matrices on the row index.

    Parameters
    ----------
    X
        Input design matrix.
    Y
        Output response matrix.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Aligned copies of ``X`` and ``Y`` sharing a common row index.

    Raises
    ------
    ValueError
        Raised when the aligned result is empty.
    """
    X_aligned, Y_aligned = X.align(Y, join="inner", axis=0)
    if len(X_aligned) == 0:
        raise ValueError("X and Y have no overlapping row index after alignment.")
    return X_aligned, Y_aligned


def _require_matching_xy(X: pd.DataFrame, Y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return copies of an unambiguously aligned labeled input/response pair."""
    if not isinstance(X, pd.DataFrame) or not isinstance(Y, pd.DataFrame):
        raise TypeError("X and Y must be pandas DataFrames")
    if X.index.has_duplicates or Y.index.has_duplicates:
        raise ValueError("X and Y must not contain duplicate row indexes")
    if X.columns.has_duplicates or Y.columns.has_duplicates:
        raise ValueError("X and Y must not contain duplicate column names")
    for label, frame in (("X", X), ("Y", Y)):
        if not all(isinstance(column, str) and column for column in frame.columns):
            raise ValueError(f"{label} column names must be non-empty strings")
    if not X.index.equals(Y.index):
        raise ValueError("X and Y must have identical row indexes in the same order")
    return X.copy(), Y.copy()


def combination_labels(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    output_column: str = "combination",
) -> pd.Series:
    """Create deterministic combination labels from arbitrary categorical columns.

    Parameters
    ----------
    frame
        Input frame containing the requested ``columns``.
    columns
        Ordered sequence of column names whose values are joined to form each
        label.  Column order is preserved verbatim in the output.
    output_column
        Name attached to the returned series.

    Returns
    -------
    pandas.Series
        Series of strings of the form ``f"{col1}={val1}|{col2}={val2}|..."``.

    Raises
    ------
    ValueError
        Raised when ``columns`` is empty or references missing columns.
    """
    columns = list(columns)
    if not columns:
        raise ValueError("combination_labels requires at least one column.")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in frame columns {list(frame.columns)}.")

    parts = [frame[col].astype(str).map(lambda v, c=col: f"{c}={v}") for col in columns]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined.str.cat(part, sep="|")
    return pd.Series(joined.to_numpy(), index=frame.index, name=output_column)


def stratified_subset_by_combination(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    n_per_combination: int = 5000,
    random_state: int = 123,
    require_all_combinations: bool = True,
) -> pd.DataFrame:
    """Sample a balanced subset stratified by observed value combinations.

    Strata are the observed combinations of the given ``columns`` (sorted
    deterministically).  Within each stratum, ``n_per_combination`` rows are
    drawn without replacement using a NumPy ``RandomState`` seeded by
    ``random_state``.

    Parameters
    ----------
    frame
        Candidate rows containing all of ``columns``.
    columns
        Ordered sequence of column names defining the stratification.
    n_per_combination
        Number of rows to draw from each observed combination.
    random_state
        Base seed for deterministic sampling.
    require_all_combinations
        When True (default), every observed combination must have
        ``>= n_per_combination`` rows; a shortfall raises ``ValueError``.
        When False, strata smaller than ``n_per_combination`` are skipped.

    Returns
    -------
    pandas.DataFrame
        Row subset containing sampled rows per stratum, concatenated in sorted
        stratum order.

    Raises
    ------
    ValueError
        Raised when a stratum is too small under ``require_all_combinations=True``
        or when ``columns`` is empty / references missing columns.
    """
    labels = combination_labels(frame, columns=columns)
    strata_order = sorted(pd.unique(labels).tolist())

    rng = np.random.RandomState(random_state)
    parts: list[pd.DataFrame] = []
    for stratum in strata_order:
        mask = labels == stratum
        block = frame.loc[mask]
        if len(block) < int(n_per_combination):
            if require_all_combinations:
                raise ValueError(
                    f"Combination {stratum} has only {len(block)} rows; "
                    f"cannot draw {n_per_combination}."
                )
            continue
        sampled_index = rng.choice(
            block.index.to_numpy(),
            size=int(n_per_combination),
            replace=False,
        )
        parts.append(block.loc[sampled_index].copy())
    if not parts:
        return frame.iloc[0:0].copy()
    return pd.concat(parts, axis=0)


def stratified_holdout_split(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    holdout_fraction: float = 0.05,
    random_state: int = 123,
    scenario_column: str = "scenario",
    stratify_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a deterministic holdout split with scenario-aware stratification.

    Parameters
    ----------
    X
        Input design matrix.
    Y
        Output response matrix.
    holdout_fraction
        Fraction of aligned rows assigned to the external holdout set.
    random_state
        Random seed forwarded to ``train_test_split``.
    scenario_column
        Scenario column or index-level name used as a fallback stratification label
        when ``stratify_columns`` is empty or not all columns are present in ``X``.
    stratify_columns
        Ordered sequence of column names used to build the stratification label.
        When all columns are present in the aligned ``X``, their string values are
        joined with ``"_"`` to form a composite label.  When empty or any column is
        absent, the function falls back to ``scenario_column``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        ``X_train``, ``X_holdout``, ``Y_train``, ``Y_holdout``.
    """
    X_aligned, Y_aligned = align_xy(X, Y)

    stratify_columns = list(stratify_columns)
    stratify = None
    if stratify_columns and all(c in X_aligned.columns for c in stratify_columns):
        label = X_aligned[stratify_columns[0]].astype(str)
        for col in stratify_columns[1:]:
            label = label + "_" + X_aligned[col].astype(str)
        stratify = label
    elif scenario_column in X_aligned.columns:
        stratify = X_aligned[scenario_column].astype(str)
    elif isinstance(X_aligned.index, pd.MultiIndex) and scenario_column in X_aligned.index.names:
        scenario_values = X_aligned.index.get_level_values(scenario_column)
        stratify = pd.Series(scenario_values, index=X_aligned.index).astype(str)

    stratify_series = None
    if stratify is not None and pd.Series(stratify).nunique() > 1:
        stratify_series = stratify

    train_idx, holdout_idx = train_test_split(
        np.arange(len(X_aligned)),
        test_size=holdout_fraction,
        random_state=random_state,
        stratify=stratify_series,
    )
    train_idx = np.sort(train_idx)
    holdout_idx = np.sort(holdout_idx)
    return (
        X_aligned.iloc[train_idx].copy(),
        X_aligned.iloc[holdout_idx].copy(),
        Y_aligned.iloc[train_idx].copy(),
        Y_aligned.iloc[holdout_idx].copy(),
    )


def fit_standardizers(
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_holdout: pd.DataFrame,
    *,
    x_feature_order: Sequence[str] | None = None,
    y_output_order: Sequence[str] | None = None,
) -> StandardizationBundle:
    """Fit train-only standardizers and transform train and holdout matrices.

    Parameters
    ----------
    X_train
        Training input frame.
    X_holdout
        Holdout input frame.
    Y_train
        Training output frame.
    Y_holdout
        Holdout output frame.
    x_feature_order
        Explicit input feature ordering. Defaults to the training frame column order.
    y_output_order
        Explicit output ordering. Defaults to the training frame column order.

    Returns
    -------
    StandardizationBundle
        Fitted scalers and transformed train/holdout arrays.
    """
    x_feature_order = list(x_feature_order if x_feature_order is not None else X_train.columns)
    y_output_order = list(y_output_order if y_output_order is not None else Y_train.columns)

    x_scaler = StandardScaler(with_mean=True, with_std=True)
    y_scaler = StandardScaler(with_mean=True, with_std=True)

    x_train = X_train.loc[:, x_feature_order].to_numpy(dtype=np.float64, copy=True)
    x_holdout = X_holdout.loc[:, x_feature_order].to_numpy(dtype=np.float64, copy=True)
    y_train = Y_train.loc[:, y_output_order].to_numpy(dtype=np.float64, copy=True)
    y_holdout = Y_holdout.loc[:, y_output_order].to_numpy(dtype=np.float64, copy=True)

    return StandardizationBundle(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        x_feature_order=x_feature_order,
        y_output_order=y_output_order,
        x_train_scaled=x_scaler.fit_transform(x_train),
        x_holdout_scaled=x_scaler.transform(x_holdout),
        y_train_scaled=y_scaler.fit_transform(y_train),
        y_holdout_scaled=y_scaler.transform(y_holdout),
    )
