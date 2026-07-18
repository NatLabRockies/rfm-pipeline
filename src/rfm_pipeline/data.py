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


def _parse_on_off_flag(label: str, token_index: int, prefix_len: int) -> int:
    """Parse a scenario token with an ``on``/``off`` suffix."""
    parts = str(label).split("_")
    if token_index >= len(parts):
        raise ValueError(f"Scenario label {label!r} does not have token index {token_index}.")
    value = parts[token_index][prefix_len:]
    if value not in {"on", "off"}:
        raise ValueError(f"Scenario token {parts[token_index]!r} does not end in 'on'/'off'.")
    return 1 if value == "on" else 0


def add_scenario_flags(
    frame: pd.DataFrame,
    *,
    scenario_column: str = "scenario",
    afsc_column: str = "AFSC",
    uaeoro_column: str = "UAEORO",
) -> pd.DataFrame:
    """Add AFSC and UAEORO flags from scenario labels.

    Parameters
    ----------
    frame
        Input frame containing scenario labels either in a column or in a MultiIndex
        level.
    scenario_column
        Column or index-level name containing scenario labels.
    afsc_column
        Name of the output AFSC indicator column.
    uaeoro_column
        Name of the output UAEORO indicator column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``frame`` with integer AFSC and UAEORO indicator columns added.

    Raises
    ------
    ValueError
        Raised when scenario labels cannot be found.
    """
    if scenario_column in frame.columns:
        scenario_series = frame[scenario_column].astype(str)
    elif isinstance(frame.index, pd.MultiIndex) and scenario_column in frame.index.names:
        scenario_values = frame.index.get_level_values(scenario_column)
        scenario_series = pd.Series(scenario_values, index=frame.index).astype(str)
    else:
        msg = f"Could not locate scenario labels in column or index: {scenario_column!r}"
        raise ValueError(msg)

    out = frame.copy()
    out[afsc_column] = scenario_series.map(lambda label: _parse_on_off_flag(label, 0, 4)).astype(
        np.int8
    )
    out[uaeoro_column] = scenario_series.map(lambda label: _parse_on_off_flag(label, 1, 6)).astype(
        np.int8
    )
    return out


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


def make_boolean_combination_labels(
    frame: pd.DataFrame,
    *,
    afsc_column: str = "AFSC",
    uaeoro_column: str = "UAEORO",
    output_column: str = "scenario_bool_combo",
) -> pd.Series:
    """Create canonical four-scenario boolean-combination labels.

    Parameters
    ----------
    frame
        Input frame containing AFSC and UAEORO indicator columns.
    afsc_column
        Name of the AFSC indicator column.
    uaeoro_column
        Name of the UAEORO indicator column.
    output_column
        Name attached to the returned series.

    Returns
    -------
    pandas.Series
        Series with labels of the form ``AFSC{0|1}_UAEORO{0|1}``.
    """
    afsc = frame[afsc_column].astype(int).astype(str)
    uaeoro = frame[uaeoro_column].astype(int).astype(str)
    return pd.Series(
        "AFSC" + afsc + "_UAEORO" + uaeoro,
        index=frame.index,
        name=output_column,
    )


def stratified_subset_by_boolean_combination(
    frame: pd.DataFrame,
    *,
    n_per_combination: int = 5000,
    random_state: int = 123,
    afsc_column: str = "AFSC",
    uaeoro_column: str = "UAEORO",
    require_all_four: bool = True,
) -> pd.DataFrame:
    """Sample a canonical 20k subset by boolean-input combination.

    This implements the recovered subset-generation rule used to move from the
    300k-run archive to the 20k modeling set: sample run identifiers separately
    within each boolean combination and draw ``n_per_combination`` rows from each
    of the four scenario strata.

    Parameters
    ----------
    frame
        Candidate rows containing AFSC and UAEORO indicator columns.
    n_per_combination
        Number of rows to draw from each boolean combination.
    random_state
        Base seed for deterministic sampling.
    afsc_column
        Name of the AFSC indicator column.
    uaeoro_column
        Name of the UAEORO indicator column.
    require_all_four
        Whether to require all four boolean combinations to be present.

    Returns
    -------
    pandas.DataFrame
        Row subset containing equal counts from each boolean combination, ordered by
        sampled row index within each stratum and then concatenated across strata.

    Raises
    ------
    ValueError
        Raised when the required strata are missing or too small for the requested
        sample size.
    """
    scenario_labels = make_boolean_combination_labels(
        frame,
        afsc_column=afsc_column,
        uaeoro_column=uaeoro_column,
    )
    expected = [
        "AFSC0_UAEORO0",
        "AFSC0_UAEORO1",
        "AFSC1_UAEORO0",
        "AFSC1_UAEORO1",
    ]
    present = sorted(pd.unique(scenario_labels))
    if require_all_four and present != expected:
        raise ValueError(f"Expected all four boolean combinations {expected}, found {present}.")

    rng = np.random.RandomState(random_state)
    parts: list[pd.DataFrame] = []
    scenario_order = expected if require_all_four else present
    for scenario in scenario_order:
        mask = scenario_labels == scenario
        block = frame.loc[mask]
        if len(block) < int(n_per_combination):
            raise ValueError(
                f"Scenario {scenario} has only {len(block)} rows; cannot draw {n_per_combination}."
            )
        sampled_index = rng.choice(
            block.index.to_numpy(),
            size=int(n_per_combination),
            replace=False,
        )
        parts.append(block.loc[sampled_index].copy())
    return pd.concat(parts, axis=0)


def stratified_holdout_split(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    holdout_fraction: float = 0.05,
    random_state: int = 123,
    scenario_column: str = "scenario",
    afsc_column: str = "AFSC",
    uaeoro_column: str = "UAEORO",
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
        Scenario column or index-level name used as a fallback stratification label.
    afsc_column
        AFSC indicator column used in the preferred stratification label.
    uaeoro_column
        UAEORO indicator column used in the preferred stratification label.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        ``X_train``, ``X_holdout``, ``Y_train``, ``Y_holdout``.
    """
    X_aligned, Y_aligned = align_xy(X, Y)

    stratify = None
    if afsc_column in X_aligned.columns and uaeoro_column in X_aligned.columns:
        stratify = X_aligned[afsc_column].astype(str) + "_" + X_aligned[uaeoro_column].astype(str)
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
