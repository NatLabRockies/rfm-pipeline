# Copyright (c) 2026 Dylan Hettinger
"""Synthetic stress-test harness for screening blind spots (M6).

Five DGP families covering the failure modes described in M6:

- ``pure_interaction``: output depends only on a pairwise interaction term;
  linear screening on main effects will miss it, but feature-expanded
  ElasticNet (including the interaction term) can recover it.
- ``symmetric_nonlinearity``: output depends on an even function (x²); linear
  screening is blind because the marginal X-Y correlation is zero.
- ``rare_output_signal``: signal appears in only a small fraction of outputs;
  multi-output screening may not retain the input.
- ``pca_threshold``: signal lives in a low-variance output PC near the
  threshold; PCA compression at the threshold discards the component.
- ``range_threshold``: signal input has low empirical range; range-based
  pre-filtering removes it before screening.

For each DGP the harness plants a known signal, runs the relevant screening
stage, and returns :class:`StressTestResult` with false exclusions and
downstream holdout error.

Usage::

    spec = StressTestSpec(
        dgp_type="pure_interaction",
        n_runs=200,
        n_inputs=8,
        n_outputs=4,
        signal_strength=5.0,
        seed=42,
    )
    result = run_stress_test(spec)
    print(result.false_exclusion_rate, result.holdout_nrmse)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, MultiTaskElasticNetCV
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DGP_TYPES: frozenset[str] = frozenset(
    [
        "pure_interaction",
        "symmetric_nonlinearity",
        "rare_output_signal",
        "pca_threshold",
        "range_threshold",
    ]
)

# Bulk scale for pca_threshold DGP — dominant outputs have std = _PCA_BULK_SCALE.
_PCA_BULK_SCALE: float = 5.0

# Relative std of the signal input in range_threshold DGP as a fraction of
# range_threshold.  At signal_strength=1 the std = range_threshold *
# _RANGE_SIGNAL_STD_FACTOR and expected range ≈ 0.5 * range_threshold (well
# below the filter cutoff).
_RANGE_SIGNAL_STD_FACTOR: float = 0.08


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StressTestSpec:
    """Fully parameterized stress-test configuration.

    Parameters
    ----------
    dgp_type
        One of ``VALID_DGP_TYPES``.
    n_runs
        Total number of simulated runs (train + holdout).
    n_inputs
        Number of input columns.
    n_outputs
        Number of output columns.
    signal_strength
        Amplitude of the planted signal.  Zero produces a pure-noise DGP.
    seed
        Random seed for full reproducibility.
    holdout_fraction
        Fraction of runs reserved for holdout evaluation.
    rare_output_fraction
        Fraction of outputs that receive the planted signal (``rare_output_signal``
        DGP only).
    pca_variance_threshold
        Cumulative explained-variance cutoff for PCA compression
        (``pca_threshold`` DGP only).
    range_threshold
        Minimum empirical range required for an input to survive the range
        filter (``range_threshold`` DGP only).
    screening_cv
        Number of cross-validation folds for :class:`MultiTaskElasticNetCV`.
    screening_l1_ratios
        ElasticNet mixing parameters searched during cross-validation.
    """

    dgp_type: str
    n_runs: int
    n_inputs: int
    n_outputs: int
    signal_strength: float
    seed: int
    holdout_fraction: float = 0.2
    rare_output_fraction: float = 0.1
    pca_variance_threshold: float = 0.95
    range_threshold: float = 0.3
    screening_cv: int = 3
    screening_l1_ratios: tuple[float, ...] = (0.5, 0.8, 1.0)

    def __post_init__(self) -> None:
        """Validate stress-test configuration values after initialization."""
        if self.dgp_type not in VALID_DGP_TYPES:
            raise ValueError(
                f"dgp_type must be one of {sorted(VALID_DGP_TYPES)}; got {self.dgp_type!r}"
            )
        if self.n_runs < 10:
            raise ValueError(f"n_runs must be >= 10; got {self.n_runs}")
        if self.n_inputs < 2:
            raise ValueError(f"n_inputs must be >= 2; got {self.n_inputs}")
        if self.n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1; got {self.n_outputs}")
        if self.signal_strength < 0.0:
            raise ValueError(f"signal_strength must be >= 0; got {self.signal_strength}")
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in (0, 1); got {self.holdout_fraction}")
        if not 0.0 < self.rare_output_fraction <= 1.0:
            raise ValueError(
                f"rare_output_fraction must be in (0, 1]; got {self.rare_output_fraction}"
            )
        if not 0.0 < self.pca_variance_threshold < 1.0:
            raise ValueError(
                f"pca_variance_threshold must be in (0, 1); got {self.pca_variance_threshold}"
            )
        if self.range_threshold <= 0.0:
            raise ValueError(f"range_threshold must be > 0; got {self.range_threshold}")
        if self.screening_cv < 2:
            raise ValueError(f"screening_cv must be >= 2; got {self.screening_cv}")
        if len(self.screening_l1_ratios) == 0:
            raise ValueError("screening_l1_ratios must not be empty")


@dataclass(frozen=True, slots=True)
class StressDGPData:
    """Intermediate container produced by a stress-test DGP generator.

    Attributes
    ----------
    X_train
        Input matrix, training split.
    Y_train
        Output matrix, training split.
    X_holdout
        Input matrix, holdout split.
    Y_holdout
        Output matrix, holdout split.
    true_signal_features
        Feature names (in the *expanded* screening space) that carry the
        planted signal.  An empty tuple means zero planted features.
    dgp_type
        DGP family identifier.
    """

    X_train: pd.DataFrame
    Y_train: pd.DataFrame
    X_holdout: pd.DataFrame
    Y_holdout: pd.DataFrame
    true_signal_features: tuple[str, ...]
    dgp_type: str


@dataclass(frozen=True, slots=True)
class StressTestResult:
    """Outcome of one stress-test run.

    Attributes
    ----------
    dgp_type
        DGP family identifier.
    signal_strength
        Signal amplitude used for this run.
    n_true_signals
        Number of planted signal features.
    n_false_exclusions
        Number of planted signal features *not* selected by the screening stage.
    false_exclusion_rate
        ``n_false_exclusions / max(n_true_signals, 1)``.
    false_excluded_features
        Names of the excluded planted features.
    holdout_nrmse
        Normalized RMSE on holdout using an OLS model fitted on the selected
        features (normalized by the holdout output standard deviation).
    selected_features
        Features selected by the screening stage.
    """

    dgp_type: str
    signal_strength: float
    n_true_signals: int
    n_false_exclusions: int
    false_exclusion_rate: float
    false_excluded_features: tuple[str, ...]
    holdout_nrmse: float
    selected_features: tuple[str, ...]


# ---------------------------------------------------------------------------
# DGP generators
# ---------------------------------------------------------------------------


def make_pure_interaction_dgp(spec: StressTestSpec) -> StressDGPData:
    """Generate a DGP where signal comes only from x0 × x1.

    Main effects have zero true coefficient, so linear screening on the
    original features cannot recover the signal.  The expanded feature space
    contains the interaction term ``"x0:x1"`` as the sole true signal feature.

    Parameters
    ----------
    spec
        Stress-test configuration; ``dgp_type`` must be ``"pure_interaction"``.
    """
    rng = np.random.default_rng(spec.seed)
    X = rng.uniform(0.0, 1.0, (spec.n_runs, spec.n_inputs))

    # Centered interaction; normalized to unit std for clean SNR interpretation.
    raw_interaction = (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
    raw_std = float(raw_interaction.std(ddof=1))
    if raw_std < 1e-9:
        raw_std = 1.0
    interaction = raw_interaction / raw_std

    # Broadcast signal across all outputs; add unit-noise.
    Y = spec.signal_strength * interaction[:, None] * np.ones((1, spec.n_outputs))
    Y += rng.standard_normal((spec.n_runs, spec.n_outputs))

    return _build_dgp_data(X, Y, true_feature="x0:x1", spec=spec, dgp_type="pure_interaction")


def make_symmetric_nonlinearity_dgp(spec: StressTestSpec) -> StressDGPData:
    """Generate a DGP where signal comes only from (x0 − 0.5)².

    The linear correlation between x0 and Y is zero by design (even function
    centred at 0.5), so linear screening is blind.  The expanded feature space
    adds ``"sq_x0" = (x0 − mean(x0))²`` as the true signal feature.

    Parameters
    ----------
    spec
        Stress-test configuration; ``dgp_type`` must be
        ``"symmetric_nonlinearity"``.
    """
    rng = np.random.default_rng(spec.seed)
    X = rng.uniform(0.0, 1.0, (spec.n_runs, spec.n_inputs))

    # Even function centred at 0.5; has zero linear correlation with uniform x0.
    raw_sq = (X[:, 0] - 0.5) ** 2
    raw_std = float(raw_sq.std(ddof=1))
    if raw_std < 1e-9:
        raw_std = 1.0
    sq_signal = raw_sq / raw_std  # unit std, mean ≈ 1/12 / raw_std

    Y = spec.signal_strength * sq_signal[:, None] * np.ones((1, spec.n_outputs))
    Y += rng.standard_normal((spec.n_runs, spec.n_outputs))

    return _build_dgp_data(X, Y, true_feature="sq_x0", spec=spec, dgp_type="symmetric_nonlinearity")


def make_rare_output_signal_dgp(spec: StressTestSpec) -> StressDGPData:
    """Generate a DGP where x0 affects only a small fraction of outputs.

    A multi-output screener calibrated on the majority of outputs may not
    retain x0 because the per-output signal-to-noise ratio is diluted across
    the null outputs.

    Parameters
    ----------
    spec
        Stress-test configuration; ``dgp_type`` must be
        ``"rare_output_signal"``.
    """
    rng = np.random.default_rng(spec.seed)
    X = rng.uniform(0.0, 1.0, (spec.n_runs, spec.n_inputs))

    x0 = (X[:, 0] - 0.5) / max(float(X[:, 0].std(ddof=1)), 1e-9)  # unit std

    n_signal_outputs = max(1, round(spec.n_outputs * spec.rare_output_fraction))
    Y = rng.standard_normal((spec.n_runs, spec.n_outputs))
    Y[:, :n_signal_outputs] += spec.signal_strength * x0[:, None]

    return _build_dgp_data(X, Y, true_feature="x0", spec=spec, dgp_type="rare_output_signal")


def make_pca_threshold_dgp(spec: StressTestSpec) -> StressDGPData:
    """Generate a DGP where x0's signal lives in a low-variance output PC.

    There are ``n_outputs - 1`` bulk outputs with high variance
    (std = ``_PCA_BULK_SCALE``) that dominate the PCA spectrum, plus one
    signal output driven by x0 with std ≈ 1 at ``signal_strength = 0``.

    When ``signal_strength`` is large enough the signal output's variance
    exceeds the fraction needed to push cumulative explained variance above
    ``pca_variance_threshold``; the signal PC is retained and x0 is detected.
    At zero strength the signal output is pure noise with small variance and
    PCA at the default threshold discards it.

    For the blind spot to manifest, ``n_outputs`` must be ≥ 5 so that the
    bulk columns dominate the cumulative variance.

    Parameters
    ----------
    spec
        Stress-test configuration; ``dgp_type`` must be ``"pca_threshold"``.
    """
    if spec.n_outputs < 5:
        raise ValueError(f"pca_threshold DGP requires n_outputs >= 5; got {spec.n_outputs}")
    rng = np.random.default_rng(spec.seed)
    X = rng.uniform(0.0, 1.0, (spec.n_runs, spec.n_inputs))

    # Bulk: (n_outputs - 1) independent high-variance outputs.
    Y_bulk = rng.standard_normal((spec.n_runs, spec.n_outputs - 1)) * _PCA_BULK_SCALE

    # Signal output: x0 (normalized) + unit noise.
    x0 = (X[:, 0] - X[:, 0].mean()) / max(float(X[:, 0].std(ddof=1)), 1e-9)
    y_signal_out = spec.signal_strength * x0 + rng.standard_normal(spec.n_runs)

    Y = np.column_stack([Y_bulk, y_signal_out])

    return _build_dgp_data(X, Y, true_feature="x0", spec=spec, dgp_type="pca_threshold")


def make_range_threshold_dgp(spec: StressTestSpec) -> StressDGPData:
    """Generate a DGP where the signal input has artificially small range.

    ``x_signal`` (column ``"x0"``) is drawn from a narrow distribution whose
    empirical range is well below ``range_threshold``.  The remaining inputs
    are uniform on [0, 1].  The output is driven by x0 with amplitude
    ``signal_strength``.

    Because the range filter removes x0 before screening, x0 is always a
    false exclusion at any positive ``signal_strength``.

    Parameters
    ----------
    spec
        Stress-test configuration; ``dgp_type`` must be ``"range_threshold"``.
    """
    rng = np.random.default_rng(spec.seed)

    # Signal input: narrow distribution, expected range << range_threshold.
    signal_std = spec.range_threshold * _RANGE_SIGNAL_STD_FACTOR
    x_signal = rng.normal(0.5, signal_std, spec.n_runs)

    # Remaining inputs: uniform [0, 1] with range ≈ 1 >> range_threshold.
    X_rest = rng.uniform(0.0, 1.0, (spec.n_runs, spec.n_inputs - 1))
    X_arr = np.column_stack([x_signal, X_rest])

    # Output driven by x_signal.
    x_signal_norm = (x_signal - x_signal.mean()) / max(float(x_signal.std(ddof=1)), 1e-9)
    Y = spec.signal_strength * x_signal_norm[:, None] * np.ones((1, spec.n_outputs))
    Y += rng.standard_normal((spec.n_runs, spec.n_outputs))

    return _build_dgp_data(X_arr, Y, true_feature="x0", spec=spec, dgp_type="range_threshold")


# ---------------------------------------------------------------------------
# Main harness entry point
# ---------------------------------------------------------------------------


def run_stress_test(spec: StressTestSpec) -> StressTestResult:
    """Run the full stress-test pipeline for one :class:`StressTestSpec`.

    Steps
    -----
    1. Generate synthetic DGP data.
    2. Run the DGP-specific screening stage (feature expansion, PCA
       compression, or range filtering as appropriate).
    3. Identify false exclusions (planted signals not selected).
    4. Compute holdout nRMSE using an OLS model on selected features.

    Parameters
    ----------
    spec
        Fully parameterized test configuration.

    Returns
    -------
    StressTestResult
        False-exclusion metrics and downstream error for this run.
    """
    dgp_data = _make_dgp(spec)
    selected, X_train_stage, X_holdout_stage = _run_stage(dgp_data, spec)

    false_excluded = tuple(f for f in dgp_data.true_signal_features if f not in selected)
    n_true = len(dgp_data.true_signal_features)
    n_false_excl = len(false_excluded)
    rate = n_false_excl / max(n_true, 1)

    nrmse = _compute_holdout_nrmse(
        X_train_stage=X_train_stage,
        Y_train=dgp_data.Y_train,
        X_holdout_stage=X_holdout_stage,
        Y_holdout=dgp_data.Y_holdout,
        selected_features=selected,
    )

    return StressTestResult(
        dgp_type=spec.dgp_type,
        signal_strength=spec.signal_strength,
        n_true_signals=n_true,
        n_false_exclusions=n_false_excl,
        false_exclusion_rate=rate,
        false_excluded_features=false_excluded,
        holdout_nrmse=nrmse,
        selected_features=selected,
    )


# ---------------------------------------------------------------------------
# DGP dispatch and helpers
# ---------------------------------------------------------------------------


def _make_dgp(spec: StressTestSpec) -> StressDGPData:
    dispatch = {
        "pure_interaction": make_pure_interaction_dgp,
        "symmetric_nonlinearity": make_symmetric_nonlinearity_dgp,
        "rare_output_signal": make_rare_output_signal_dgp,
        "pca_threshold": make_pca_threshold_dgp,
        "range_threshold": make_range_threshold_dgp,
    }
    return dispatch[spec.dgp_type](spec)


def _build_dgp_data(
    X_arr: np.ndarray,
    Y_arr: np.ndarray,
    *,
    true_feature: str,
    spec: StressTestSpec,
    dgp_type: str,
) -> StressDGPData:
    """Split X/Y arrays into train/holdout and wrap in :class:`StressDGPData`."""
    n_holdout = max(1, round(spec.n_runs * spec.holdout_fraction))
    n_holdout = min(n_holdout, spec.n_runs - 1)
    n_train = spec.n_runs - n_holdout

    input_cols = [f"x{i}" for i in range(X_arr.shape[1])]
    output_cols = [f"y{i}" for i in range(Y_arr.shape[1])]

    X_df = pd.DataFrame(X_arr, columns=input_cols)
    Y_df = pd.DataFrame(Y_arr, columns=output_cols)

    X_train = X_df.iloc[:n_train].reset_index(drop=True)
    Y_train = Y_df.iloc[:n_train].reset_index(drop=True)
    X_holdout = X_df.iloc[n_train:].reset_index(drop=True)
    Y_holdout = Y_df.iloc[n_train:].reset_index(drop=True)

    return StressDGPData(
        X_train=X_train,
        Y_train=Y_train,
        X_holdout=X_holdout,
        Y_holdout=Y_holdout,
        true_signal_features=(true_feature,),
        dgp_type=dgp_type,
    )


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _run_stage(
    dgp_data: StressDGPData,
    spec: StressTestSpec,
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Dispatch to the DGP-specific screening stage.

    Returns ``(selected_features, X_train_stage, X_holdout_stage)`` where
    ``X_train_stage`` and ``X_holdout_stage`` are the feature matrices
    presented to the screening estimator (possibly expanded or filtered).
    """
    if spec.dgp_type == "pure_interaction":
        return _stage_interaction(dgp_data, spec)
    if spec.dgp_type == "symmetric_nonlinearity":
        return _stage_symmetric_nonlinearity(dgp_data, spec)
    if spec.dgp_type == "rare_output_signal":
        return _stage_linear(dgp_data, spec)
    if spec.dgp_type == "pca_threshold":
        return _stage_pca_threshold(dgp_data, spec)
    if spec.dgp_type == "range_threshold":
        return _stage_range_threshold(dgp_data, spec)
    raise AssertionError(f"Unhandled dgp_type: {spec.dgp_type!r}")  # pragma: no cover


def _stage_interaction(
    dgp_data: StressDGPData,
    spec: StressTestSpec,
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Expand with pairwise interactions, run ElasticNet."""
    max_inp = min(len(dgp_data.X_train.columns), 10)
    X_tr_exp = _add_interaction_features(dgp_data.X_train, max_inputs=max_inp)
    X_ho_exp = _add_interaction_features(dgp_data.X_holdout, max_inputs=max_inp)
    selected = _elastic_net_screen(X_tr_exp, dgp_data.Y_train, spec)
    return selected, X_tr_exp, X_ho_exp


def _stage_symmetric_nonlinearity(
    dgp_data: StressDGPData,
    spec: StressTestSpec,
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Add squared features, run ElasticNet."""
    X_tr_exp = _add_squared_features(dgp_data.X_train)
    X_ho_exp = _add_squared_features(dgp_data.X_holdout)
    selected = _elastic_net_screen(X_tr_exp, dgp_data.Y_train, spec)
    return selected, X_tr_exp, X_ho_exp


def _stage_linear(
    dgp_data: StressDGPData,
    spec: StressTestSpec,
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Run ElasticNet on original features without expansion."""
    selected = _elastic_net_screen(dgp_data.X_train, dgp_data.Y_train, spec)
    return selected, dgp_data.X_train.copy(), dgp_data.X_holdout.copy()


def _stage_pca_threshold(
    dgp_data: StressDGPData,
    spec: StressTestSpec,
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Compress Y by PCA to the variance threshold, then screen X → Y_pca."""
    Y_arr = dgp_data.Y_train.to_numpy(dtype=float)
    n_comp_max = min(Y_arr.shape[0] - 1, Y_arr.shape[1])

    pca = PCA(n_components=n_comp_max)
    pca.fit(Y_arr)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, spec.pca_variance_threshold - 1e-9)) + 1
    n_keep = min(n_keep, n_comp_max)

    Y_pca = pd.DataFrame(
        pca.transform(Y_arr)[:, :n_keep],
        columns=[f"pc{i}" for i in range(n_keep)],
    )

    selected = _elastic_net_screen(dgp_data.X_train, Y_pca, spec)
    return selected, dgp_data.X_train.copy(), dgp_data.X_holdout.copy()


def _stage_range_threshold(
    dgp_data: StressDGPData,
    spec: StressTestSpec,
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Filter inputs by empirical range, then screen filtered X → Y."""
    X_tr = dgp_data.X_train
    ranges = X_tr.max(axis=0) - X_tr.min(axis=0)
    kept = [col for col in X_tr.columns if ranges[col] >= spec.range_threshold]

    if not kept:
        return (), X_tr.copy(), dgp_data.X_holdout.copy()

    X_tr_filt = X_tr[kept].copy()
    X_ho_filt = dgp_data.X_holdout[kept].copy()
    selected = _elastic_net_screen(X_tr_filt, dgp_data.Y_train, spec)
    return selected, X_tr_filt, X_ho_filt


# ---------------------------------------------------------------------------
# Feature expansion helpers
# ---------------------------------------------------------------------------


def _add_interaction_features(X: pd.DataFrame, *, max_inputs: int) -> pd.DataFrame:
    """Return a copy of X with all pairwise interaction columns appended."""
    X_exp = X.copy()
    cols = X.columns.tolist()[:max_inputs]
    for c1, c2 in combinations(cols, 2):
        X_exp[f"{c1}:{c2}"] = X[c1].to_numpy() * X[c2].to_numpy()
    return X_exp


def _add_squared_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with mean-centred squared terms appended."""
    X_exp = X.copy()
    for col in X.columns:
        vals = X[col].to_numpy(dtype=float)
        centred_sq = (vals - vals.mean()) ** 2
        X_exp[f"sq_{col}"] = centred_sq
    return X_exp


# ---------------------------------------------------------------------------
# Screening primitive
# ---------------------------------------------------------------------------


def _elastic_net_screen(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    spec: StressTestSpec,
) -> tuple[str, ...]:
    """Multi-output ElasticNet screening; return selected feature names.

    Features with all-zero coefficients across every output are treated as
    unselected.

    Parameters
    ----------
    X_train
        Feature matrix.
    Y_train
        Response matrix (may be PCA-compressed for the pca_threshold stage).
    spec
        Stress-test spec providing CV folds, l1_ratios, and seed.
    """
    X_arr = X_train.to_numpy(dtype=float)
    Y_arr = Y_train.to_numpy(dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr[:, None]

    n_rows = len(X_arr)
    cv = min(spec.screening_cv, n_rows // 2)
    cv = max(cv, 2)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_x.fit_transform(X_arr)
    Ys = scaler_y.fit_transform(Y_arr)

    est = MultiTaskElasticNetCV(
        l1_ratio=list(spec.screening_l1_ratios),
        cv=cv,
        max_iter=5000,
        random_state=spec.seed,
    )
    est.fit(Xs, Ys)

    coef = np.asarray(est.coef_, dtype=float)  # (n_outputs, n_features)
    selected_mask = np.any(np.abs(coef) > 1e-8, axis=0)
    return tuple(col for col, keep in zip(X_train.columns, selected_mask, strict=True) if keep)


# ---------------------------------------------------------------------------
# Holdout nRMSE
# ---------------------------------------------------------------------------


def _compute_holdout_nrmse(
    *,
    X_train_stage: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_holdout_stage: pd.DataFrame,
    Y_holdout: pd.DataFrame,
    selected_features: tuple[str, ...],
) -> float:
    """OLS on selected features; return normalized RMSE on holdout.

    Normalization is by the RMS of per-output standard deviations on the
    holdout set.  When no features are selected the model predicts the
    training-set column means.

    Parameters
    ----------
    X_train_stage
        Feature matrix presented to the screening stage (may be expanded or
        filtered).
    Y_train
        Original output matrix, training split.
    X_holdout_stage
        Feature matrix for holdout (same columns as ``X_train_stage``).
    Y_holdout
        Original output matrix, holdout split.
    selected_features
        Feature names selected by the screening stage.
    """
    y_holdout = Y_holdout.to_numpy(dtype=float)

    avail = [
        f for f in selected_features if f in X_train_stage.columns and f in X_holdout_stage.columns
    ]

    if avail:
        lr = LinearRegression()
        lr.fit(X_train_stage[avail].to_numpy(dtype=float), Y_train.to_numpy(dtype=float))
        y_pred = lr.predict(X_holdout_stage[avail].to_numpy(dtype=float))
    else:
        y_pred = np.tile(Y_train.to_numpy(dtype=float).mean(axis=0), (len(y_holdout), 1))

    residuals = y_holdout - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))

    per_out_std = y_holdout.std(axis=0, ddof=1)
    scale = float(np.sqrt(np.mean(per_out_std**2)))
    if scale < 1e-9:
        scale = 1.0

    return rmse / scale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "VALID_DGP_TYPES",
    "StressDGPData",
    "StressTestResult",
    "StressTestSpec",
    "make_pure_interaction_dgp",
    "make_pca_threshold_dgp",
    "make_range_threshold_dgp",
    "make_rare_output_signal_dgp",
    "make_symmetric_nonlinearity_dgp",
    "run_stress_test",
]
