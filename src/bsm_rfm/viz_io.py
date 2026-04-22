"""Read-only loaders for canonical post-fit artifact tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def canonical_bundle_loader_keys() -> list[str]:
    """Return the stable logical table names exposed by the bundle loader.

    Returns
    -------
    list of str
        Canonical logical keys returned by :func:`load_postfit_bundle`.
    """
    return [
        "all_input_metadata",
        "selected_input_metadata",
        "output_metadata",
        "coef_matrix_standardized",
        "coef_matrix_raw_scale",
        "x_standardization",
        "y_standardization",
        "nrmse_summary",
    ]


def load_table(path_no_ext: Path) -> pd.DataFrame:
    """Load a canonical artifact table from Parquet or CSV.

    Parameters
    ----------
    path_no_ext
        Path without extension. The loader checks for ``.parquet`` first and then
        ``.csv``.

    Returns
    -------
    pandas.DataFrame
        Loaded table.

    Raises
    ------
    FileNotFoundError
        Raised when neither a Parquet nor CSV artifact exists.
    """
    parquet_path = path_no_ext.with_suffix(".parquet")
    csv_path = path_no_ext.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"Expected {parquet_path.name} or {csv_path.name} under {path_no_ext.parent}"
    )


def load_postfit_bundle(root: Path) -> dict[str, pd.DataFrame]:
    """Load the canonical post-fit visualization inputs.

    Parameters
    ----------
    root
        Export bundle directory containing the ``postfit_diagnostics`` subtree.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from logical artifact name to loaded table.
    """
    postfit = root / "postfit_diagnostics"
    return {
        artifact_name: load_table(postfit / artifact_name)
        for artifact_name in canonical_bundle_loader_keys()
    }


def load_pipeline_outputs(root: Path) -> dict[str, pd.DataFrame]:
    """Backward-compatible alias for the canonical post-fit bundle loader.

    Parameters
    ----------
    root
        Export bundle directory containing the ``postfit_diagnostics`` subtree.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from logical artifact name to loaded table.
    """
    return load_postfit_bundle(root)
