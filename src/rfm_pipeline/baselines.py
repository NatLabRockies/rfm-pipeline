"""Competitive baseline model interface and comparison harness.

Provides a generic fit/predict protocol and concrete baseline implementations:
- Ridge regression (reduced-rank via PLS or ridge)
- Elastic-net sparse-linear baseline
- Per-stratum first-order (OLS) models
- OracleOLSBaseline (fits OLS on the planted support; unattainable diagnostic)
- GBTBaseline (gradient-boosted tree nonlinear surrogate)

All baselines operate on arbitrary NumPy input and output arrays.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class BaselineProtocol(Protocol):
    """Minimal interface every baseline must satisfy."""

    @property
    def name(self) -> str:
        """Return a stable human-readable baseline name."""
        ...

    def fit(self, X: np.ndarray, Y: np.ndarray) -> BaselineProtocol:
        """Fit the baseline to input and response arrays."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict responses for input rows."""
        ...

    def model_size_bytes(self) -> int:
        """Return an approximate fitted-parameter memory footprint."""
        ...


# ---------------------------------------------------------------------------
# Concrete baselines
# ---------------------------------------------------------------------------


@dataclass
class RidgeBaseline:
    """Multi-output ridge regression baseline."""

    alpha: float = 1.0
    _model: Ridge | None = field(default=None, repr=False, init=False)

    @property
    def name(self) -> str:
        """Return the configured ridge baseline name."""
        return f"ridge(alpha={self.alpha})"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> RidgeBaseline:
        """Fit ridge regression on the provided arrays."""
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict responses with the fitted ridge model."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._model.predict(X))

    def model_size_bytes(self) -> int:
        """Return the fitted ridge coefficient and intercept size in bytes."""
        if self._model is None:
            return 0
        coef = self._model.coef_
        intercept = self._model.intercept_
        return int(np.asarray(coef).nbytes + np.asarray(intercept).nbytes)


@dataclass
class PLSBaseline:
    """Partial least-squares (reduced-rank) baseline."""

    n_components: int = 5
    _model: PLSRegression | None = field(default=None, repr=False, init=False)

    @property
    def name(self) -> str:
        """Return the configured PLS baseline name."""
        return f"pls(n_components={self.n_components})"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> PLSBaseline:
        """Fit PLS regression using no more components than data allow."""
        n_components = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self._model = PLSRegression(n_components=n_components)
        self._model.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict responses with the fitted PLS model."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._model.predict(X))

    def model_size_bytes(self) -> int:
        """Return the fitted PLS loading, weight, and coefficient size in bytes."""
        if self._model is None:
            return 0
        total = 0
        for attr in ("x_loadings_", "y_loadings_", "x_weights_", "y_weights_", "coef_"):
            arr = getattr(self._model, attr, None)
            if arr is not None:
                total += np.asarray(arr).nbytes
        return total


@dataclass
class ElasticNetBaseline:
    """Sparse-linear elastic-net baseline (multi-output via MultiTaskElasticNet or per-column)."""

    alpha: float = 0.01
    l1_ratio: float = 0.5
    max_iter: int = 2000
    _models: list[ElasticNet] | None = field(default=None, repr=False, init=False)
    _multi: bool = field(default=False, repr=False, init=False)

    @property
    def name(self) -> str:
        """Return the configured elastic-net baseline name."""
        return f"elasticnet(alpha={self.alpha},l1_ratio={self.l1_ratio})"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ElasticNetBaseline:
        """Fit multitask elastic net, falling back to per-output models on failure."""
        Y2d = np.atleast_2d(Y.T).T  # ensure (n, p)
        if Y2d.ndim == 1 or Y2d.shape[1] == 1:
            y_col = Y2d.ravel()
            m = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
            m.fit(X, y_col)
            self._models = [m]
        else:
            # Try MultiTaskElasticNet; fall back to per-column if it fails
            try:
                mt = MultiTaskElasticNet(
                    alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter
                )
                mt.fit(X, Y2d)
                self._models = [mt]  # type: ignore[list-item]
                self._multi = True
            except Exception:
                models = []
                for j in range(Y2d.shape[1]):
                    m = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
                    m.fit(X, Y2d[:, j])
                    models.append(m)
                self._models = models
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict responses with the fitted elastic-net model or models."""
        if self._models is None:
            raise RuntimeError("Call fit() before predict().")
        if self._multi or len(self._models) == 1:
            out = self._models[0].predict(X)
            return np.asarray(out)
        cols = [m.predict(X) for m in self._models]
        return np.column_stack(cols)

    def model_size_bytes(self) -> int:
        """Return the fitted elastic-net coefficient and intercept size in bytes."""
        if not self._models:
            return 0
        total = 0
        for m in self._models:
            for attr in ("coef_", "intercept_"):
                arr = getattr(m, attr, None)
                if arr is not None:
                    total += np.asarray(arr).nbytes
        return total


@dataclass
class PerStratumFirstOrderBaseline:
    """Per-stratum OLS (first-order) baseline.

    Fits an independent Ridge(alpha≈0) per stratum value.  Falls back to a
    global model for strata not seen during training.

    Parameters
    ----------
    strata_train
        1-D array of stratum labels aligned with training rows.
    strata_eval
        1-D array of stratum labels aligned with evaluation rows (set in
        :meth:`predict` via the ``strata`` keyword, or pass at construction
        for convenience).
    """

    strata_train: np.ndarray | None = None
    alpha: float = 1e-6
    _models: dict[Any, Ridge] = field(default_factory=dict, repr=False, init=False)
    _fallback: Ridge | None = field(default=None, repr=False, init=False)

    @property
    def name(self) -> str:
        """Return the per-stratum OLS baseline name."""
        return "per_stratum_ols"

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        strata: np.ndarray | None = None,
    ) -> PerStratumFirstOrderBaseline:
        """Fit one ridge-OLS model per observed stratum plus a global fallback."""
        labels = strata if strata is not None else self.strata_train
        if labels is None:
            raise ValueError("strata labels required for PerStratumFirstOrderBaseline.fit()")
        labels = np.asarray(labels)
        for s in np.unique(labels):
            mask = labels == s
            if mask.sum() < 2:
                continue
            m = Ridge(alpha=self.alpha)
            m.fit(X[mask], Y[mask] if Y.ndim > 1 else Y[mask])
            self._models[s] = m
        self._fallback = Ridge(alpha=self.alpha)
        self._fallback.fit(X, Y)
        return self

    def predict(
        self,
        X: np.ndarray,
        strata: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict rows using matching stratum models or the fitted fallback."""
        if not self._models and self._fallback is None:
            raise RuntimeError("Call fit() before predict().")
        if strata is None:
            if self._fallback is None:
                raise ValueError("strata required for predict() or fit a fallback first")
            return np.asarray(self._fallback.predict(X))
        labels = np.asarray(strata)
        # Allocate output array using fallback shape
        sample_pred = self._fallback.predict(X[:1]) if self._fallback else None  # type: ignore
        out = np.empty(
            (X.shape[0],)
            + (
                np.asarray(sample_pred).shape[1:]
                if sample_pred is not None and np.asarray(sample_pred).ndim > 1
                else ()
            ),
            dtype=float,
        )
        for i, s in enumerate(labels):
            model = self._models.get(s, self._fallback)
            if model is None:
                continue
            pred = model.predict(X[i : i + 1])
            out[i] = pred[0]
        return out

    def model_size_bytes(self) -> int:
        """Return total fitted coefficient and intercept size across stratum models."""
        total = 0
        for m in list(self._models.values()) + ([self._fallback] if self._fallback else []):
            if m is None:
                continue
            for attr in ("coef_", "intercept_"):
                arr = getattr(m, attr, None)
                if arr is not None:
                    total += np.asarray(arr).nbytes
        return total


@dataclass
class OracleOLSBaseline:
    """OLS fitted on the planted (oracle) support columns only.

    This is explicitly an *unattainable diagnostic*: it uses the true
    active-input set that a real analysis cannot know in advance.

    Parameters
    ----------
    true_active_inputs
        Set of feature names (e.g. ``{"x0", "x3"}``) that are truly active.
        Columns are selected by matching against ``feature_names``.
    feature_names
        Ordered column names of ``X`` passed to :meth:`fit`.  If ``None``,
        all columns are used (no oracle selection).
    """

    true_active_inputs: frozenset[str]
    feature_names: list[str] | None = None
    _model: Ridge | None = field(default=None, repr=False, init=False)
    _active_cols: list[int] | None = field(default=None, repr=False, init=False)

    @property
    def name(self) -> str:
        """Return the oracle OLS baseline name."""
        return "oracle_ols"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> OracleOLSBaseline:
        """Fit OLS on the oracle-selected columns."""
        if self.feature_names is not None:
            cols = [
                i for i, fname in enumerate(self.feature_names) if fname in self.true_active_inputs
            ]
        else:
            cols = list(range(X.shape[1]))
        if not cols:
            cols = list(range(X.shape[1]))
        self._active_cols = cols
        self._model = Ridge(alpha=1e-8)
        self._model.fit(X[:, cols], Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict responses using the oracle-column model."""
        if self._model is None or self._active_cols is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._model.predict(X[:, self._active_cols]))

    def model_size_bytes(self) -> int:
        """Return the fitted oracle OLS coefficient and intercept size in bytes."""
        if self._model is None:
            return 0
        total = 0
        for attr in ("coef_", "intercept_"):
            arr = getattr(self._model, attr, None)
            if arr is not None:
                total += np.asarray(arr).nbytes
        return total


@dataclass
class GBTBaseline:
    """Nonlinear predictive surrogate using gradient-boosted regression trees.

    Multi-output data is handled via :class:`~sklearn.multioutput.MultiOutputRegressor`.
    This baseline serves as a nonlinear comparator for linear methods.

    Parameters
    ----------
    n_estimators
        Number of boosting rounds.
    max_depth
        Maximum tree depth.
    learning_rate
        Shrinkage applied to each tree.
    """

    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    _model: Any = field(default=None, repr=False, init=False)

    @property
    def name(self) -> str:
        """Return the GBT baseline name."""
        return f"gbt(n={self.n_estimators},depth={self.max_depth})"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> GBTBaseline:
        """Fit gradient-boosted trees, wrapping in MultiOutputRegressor for multi-output Y."""
        Y2d = np.atleast_2d(Y.T).T
        gbt = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
        )
        if Y2d.ndim == 1 or Y2d.shape[1] == 1:
            gbt.fit(X, Y2d.ravel())
            self._model = gbt
        else:
            mo = MultiOutputRegressor(gbt)
            mo.fit(X, Y2d)
            self._model = mo
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict responses with the fitted GBT model."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._model.predict(X))

    def model_size_bytes(self) -> int:
        """Return 0 (GBT tree structures do not have a simple byte footprint)."""
        return 0


# ---------------------------------------------------------------------------
# Comparison harness
# ---------------------------------------------------------------------------


@dataclass
class ComparisonRecord:
    """Full record for one baseline evaluated on one split."""

    name: str
    rmse: float
    r2: float
    model_size_bytes: int
    fit_time_s: float
    eval_time_s: float
    peak_memory_mb: float
    extra: dict[str, Any] = field(default_factory=dict)


def _rmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))


def _r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum((Y_true - np.mean(Y_true, axis=0)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compare_baselines(
    baselines: Sequence[Any],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    *,
    fit_kwargs: dict[str, dict[str, Any]] | None = None,
    predict_kwargs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Fit and evaluate a sequence of baselines; return comparison DataFrame.

    Parameters
    ----------
    baselines
        Sequence of objects satisfying :class:`BaselineProtocol`.
    X_train, Y_train
        Training arrays (n_train × p) and (n_train × q).
    X_eval, Y_eval
        Evaluation arrays with the same column count.
    fit_kwargs
        Optional per-baseline extra keyword args for ``fit()``, keyed by
        ``baseline.name``.
    predict_kwargs
        Optional per-baseline extra keyword args for ``predict()``, keyed by
        ``baseline.name``.

    Returns
    -------
    pd.DataFrame
        One row per baseline with columns matching :class:`ComparisonRecord`.
    """
    fit_kwargs = fit_kwargs or {}
    predict_kwargs = predict_kwargs or {}
    records = []

    for bl in baselines:
        bname = bl.name

        # --- fit ---
        tracemalloc.start()
        t0 = time.perf_counter()
        bl.fit(X_train, Y_train, **fit_kwargs.get(bname, {}))
        fit_time = time.perf_counter() - t0
        _, fit_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # --- eval ---
        tracemalloc.start()
        t1 = time.perf_counter()
        Y_pred = bl.predict(X_eval, **predict_kwargs.get(bname, {}))
        eval_time = time.perf_counter() - t1
        _, eval_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = max(fit_peak, eval_peak) / 1024 / 1024

        records.append(
            ComparisonRecord(
                name=bname,
                rmse=_rmse(Y_eval, Y_pred),
                r2=_r2(Y_eval, Y_pred),
                model_size_bytes=bl.model_size_bytes(),
                fit_time_s=fit_time,
                eval_time_s=eval_time,
                peak_memory_mb=peak_mb,
            )
        )

    return pd.DataFrame(
        [
            {
                "name": r.name,
                "rmse": r.rmse,
                "r2": r.r2,
                "model_size_bytes": r.model_size_bytes,
                "fit_time_s": r.fit_time_s,
                "eval_time_s": r.eval_time_s,
                "peak_memory_mb": r.peak_memory_mb,
            }
            for r in records
        ]
    )
