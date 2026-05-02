"""Compare nodewise and debias helpers between notebook and public implementation.

This test extracts the nodewise_precision and debias helpers from the recovered
LASSO_to_OLS_v9.ipynb and runs them on small synthetic data, comparing shapes and
basic numeric sanity to the public implementation in src/bsm_rfm/debiased_lasso.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bsm_rfm import debiased_lasso as dl

NOTEBOOK = Path("docs/final_scripts_from_hpc/LASSO_to_OLS_v9.ipynb")


def _extract_funcs_from_notebook(nb_path: Path) -> dict:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    import_lines = []
    func_cells = []
    targets = (
        "def nodewise_precision",
        "def debias_coeffs_batched",
        "def ztests_from_debias_scores",
    )

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # collect import lines
        for line in src.splitlines():
            s = line.strip()
            # skip __future__ imports which must be at top of a file
            if s.startswith("from __future__"):
                continue
            # skip optional heavy imports not required for nodewise/debias helpers
            if "statsmodels" in s or "matplotlib" in s:
                continue
            if s.startswith("import ") or s.startswith("from "):
                import_lines.append(line)
        # collect any cell that defines our targets
        if any(t in src for t in targets):
            # remove future imports inside function cells
            cleaned_lines = [line for line in src.splitlines() if "from __future__" not in line]
            # drop heavy or plotting imports from function cells
            cleaned_lines = [
                line
                for line in cleaned_lines
                if ("statsmodels" not in line and "matplotlib" not in line and "plt" not in line)
            ]
            cleaned = "\n".join(cleaned_lines)
            func_cells.append(cleaned)

    # dedupe import lines while preserving order
    seen = set()
    uniq_imports = []
    for li in import_lines:
        if li not in seen:
            seen.add(li)
            uniq_imports.append(li)

    code = "\n".join(uniq_imports + ["\n# ---- extracted functions ----\n"] + func_cells)
    env: dict = {}
    # provide minimal globals
    imports = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.linear_model import Lasso\n"
        "from scipy import stats\n"
    )
    exec(imports, env)
    exec(code, env)
    return env


def test_notebook_vs_public_nodewise_debias_basic():
    env = _extract_funcs_from_notebook(NOTEBOOK)

    # small deterministic toy data
    rs = np.random.RandomState(1)
    n = 120
    p = 20
    q = 3
    X = rs.normal(size=(n, p)).astype(np.float32)
    B = np.zeros((p, q), dtype=float)
    B[:4, :] = np.array([[2.0, 1.0, 0.0]] * 4)
    Y = (X @ B + rs.normal(scale=0.1, size=(n, q))).astype(np.float32)

    # run notebook functions
    nodewise_nb = env.get("nodewise_precision")
    debias_nb = env.get("debias_coeffs_batched")
    ztests_nb = env.get("ztests_from_debias_scores")
    assert nodewise_nb is not None and debias_nb is not None and ztests_nb is not None

    # prepare DataFrame inputs (notebook helpers often expect DataFrames)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    Ydf = pd.DataFrame(Y, columns=[f"y{i}" for i in range(q)])

    # call notebook functions with numpy arrays (their extracted helpers expect arrays)
    Theta_nb, tau2_nb = nodewise_nb(X, alpha_node=1e-3)
    assert Theta_nb.shape == (p, p)
    assert tau2_nb.shape == (p,)
    assert np.isfinite(Theta_nb).all()

    # run public implementation for comparison

    Theta_impl, tau2_impl = dl.nodewise_precision(Xdf, alpha_node=1e-3)
    assert Theta_impl.shape == (p, p)
    assert tau2_impl.shape == (p,)
    assert np.isfinite(Theta_impl).all()

    # basic numeric similarity (diagonals should be positive and roughly comparable)
    diag_nb = np.diag(Theta_nb)
    diag_impl = np.diag(Theta_impl)
    assert (diag_nb > 0).all()
    assert (diag_impl > 0).all()
    # allow some tolerance given implementation differences: check correlation and median ratio
    corr = np.corrcoef(diag_nb, diag_impl)[0, 1]
    med_ratio = float(np.median(diag_nb) / np.median(diag_impl))
    # implementations differ; require basic positive correlation and reasonable scale agreement
    assert corr > 0.2, f"Nodewise diagonal correlation too low: {corr:.3f}"
    assert 0.4 < med_ratio < 2.5, f"Median diagonal ratio unexpected: {med_ratio:.3f}"

    # test debias step shapes
    # construct a simple B_hat (zeros)
    B_hat = np.zeros((p, q), dtype=np.float32)
    # call notebook debias helper with numpy arrays (extracted helpers expect arrays)
    B_tilde_nb, sigma2_nb = debias_nb(X, Y, B_hat, Theta_nb, batch=32)
    assert B_tilde_nb.shape == (p, q)
    assert sigma2_nb.shape == (q,)

    B_tilde_impl, sigma2_impl = dl.debias_coeffs_batched(Xdf, Ydf, B_hat, Theta_impl)
    assert B_tilde_impl.shape == (p, q)
    assert sigma2_impl.shape == (q,)

    # compare a few entries
    assert np.allclose(sigma2_nb, sigma2_impl, rtol=1e-2, atol=1e-6)
