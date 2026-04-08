"""Adapters for the recovered null-screening source workflow.

This module does not reimplement the permutation-null Delta workflow. Instead, it
provides a thin interface layer so the refactored package can call the recovered
``null_distribution.py`` script as the canonical implementation for the
null-screening stage.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import pandas as pd


class SupportsNullScreening(Protocol):
    """Structural protocol for the recovered null-screening script.

    The recovered source module must provide the functions used here. The concrete
    canonical implementation is the archived ``null_distribution.py`` script.
    """

    def run_delta_for_screening(
        self,
        items: Iterable[Any],
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[Any, Any]]:
        """Run the recovered Delta-screening workflow."""

    def determine_influential_inputs(
        self,
        sig_all: pd.DataFrame,
        *,
        min_outputs: int = 1,
    ) -> dict[Any, list[str]]:
        """Determine influential inputs from the recovered script."""

    def ensure_influentials_or_fallback(
        self,
        infl_map: dict[Any, list[str]],
        observed_all: pd.DataFrame,
        *,
        min_outputs: int,
    ) -> dict[Any, list[str]]:
        """Apply the recovered fallback rule."""


@dataclass(frozen=True)
class NullScreeningConfig:
    """Configuration for the recovered null-screening workflow.

    Parameters
    ----------
    B
        Number of permutation-null replicates.
    alpha
        Family-wise error control target passed to the recovered script.
    method
        Null-cutoff method from the recovered script.
    n_jobs_perm
        Number of workers for permutation computation.
    seed
        Base random seed.
    return_full_null
        Whether to request full permutation arrays from the recovered script.
    perm_sample_size
        Optional subset size used inside each null permutation.
    perm_sample_with_replacement
        Whether the permutation subset draw uses replacement.
    cache_dir
        Directory for cached Delta/null artifacts.
    use_cache
        Whether to load existing cached null artifacts.
    save_cache
        Whether to save newly computed null artifacts.
    save_full_null
        Whether to persist the full permutation-null arrays.
    min_outputs
        Minimum number of significant outputs required for an input to be retained as
        influential before fallback logic is applied.
    """

    B: int = 200
    alpha: float = 0.05
    method: str = "fwer-max"
    n_jobs_perm: int = -1
    seed: int = 123
    return_full_null: bool = True
    perm_sample_size: int | None = None
    perm_sample_with_replacement: bool = False
    cache_dir: str | None = "delta_cache"
    use_cache: bool = True
    save_cache: bool = True
    save_full_null: bool = True
    min_outputs: int = 1


@dataclass(frozen=True)
class NullScreeningResult:
    """Outputs from the recovered null-screening stage.

    Attributes
    ----------
    observed
        Observed Delta values for each dataset, output, and variable.
    significant
        Significance flags comparing observed Delta values against null cutoffs.
    cutoffs_map
        Per-dataset cutoff matrices returned by the recovered script.
    influential_inputs
        Influential-input map after applying the recovered fallback rule.
    provenance
        Minimal provenance payload describing the source-script adapter call.
    """

    observed: pd.DataFrame
    significant: pd.DataFrame
    cutoffs_map: dict[Any, Any]
    influential_inputs: dict[Any, list[str]]
    provenance: dict[str, Any]


def load_source_module(
    path: str | Path,
    *,
    module_name: str = "bsm_rfm_source_null_distribution",
) -> ModuleType:
    """Load a Python module from disk for use as the canonical source workflow.

    Parameters
    ----------
    path
        Path to the recovered source script.
    module_name
        Import name to assign to the loaded module.

    Returns
    -------
    types.ModuleType
        Loaded module object.

    Raises
    ------
    ImportError
        Raised when the module cannot be loaded from the supplied path.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_null_screening_with_source(
    source_module: SupportsNullScreening,
    items: Iterable[Any],
    *,
    config: NullScreeningConfig | None = None,
) -> NullScreeningResult:
    """Run the recovered null-screening workflow through a stable adapter.

    Parameters
    ----------
    source_module
        Loaded recovered source module. In the canonical workflow this is the
        archived ``null_distribution.py`` implementation.
    items
        Iterable of dataset objects accepted by the recovered source script.
    config
        Null-screening configuration. When omitted, default settings are used.

    Returns
    -------
    NullScreeningResult
        Observed Delta values, significance table, per-dataset cutoffs, influential
        inputs, and minimal provenance metadata.
    """
    cfg = config or NullScreeningConfig()
    observed, significant, cutoffs_map = source_module.run_delta_for_screening(
        items,
        B=cfg.B,
        alpha=cfg.alpha,
        method=cfg.method,
        n_jobs_perm=cfg.n_jobs_perm,
        seed=cfg.seed,
        return_full_null=cfg.return_full_null,
        perm_sample_size=cfg.perm_sample_size,
        perm_sample_with_replacement=cfg.perm_sample_with_replacement,
        cache_dir=cfg.cache_dir,
        use_cache=cfg.use_cache,
        save_cache=cfg.save_cache,
        save_full_null=cfg.save_full_null,
    )
    influential = source_module.determine_influential_inputs(
        significant,
        min_outputs=cfg.min_outputs,
    )
    influential = source_module.ensure_influentials_or_fallback(
        influential,
        observed,
        min_outputs=cfg.min_outputs,
    )
    provenance = {
        "source_workflow": "recovered null_distribution.py",
        "screening_method": "SALib delta sensitivity with permutation null",
        "B": int(cfg.B),
        "alpha": float(cfg.alpha),
        "method": cfg.method,
        "perm_sample_size": (
            int(cfg.perm_sample_size) if cfg.perm_sample_size is not None else None
        ),
        "perm_sample_with_replacement": bool(cfg.perm_sample_with_replacement),
        "min_outputs": int(cfg.min_outputs),
    }
    return NullScreeningResult(
        observed=observed,
        significant=significant,
        cutoffs_map=cutoffs_map,
        influential_inputs=influential,
        provenance=provenance,
    )
