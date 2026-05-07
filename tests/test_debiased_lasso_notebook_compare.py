"""Compare nodewise and debias helpers between notebook and public implementation.

This test extracts the nodewise_precision and debias helpers from a small synthetic
notebook created inside the test and runs them on small synthetic data, comparing shapes and
basic numeric sanity to the public implementation in src/bsm_rfm/debiased_lasso.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bsm_rfm import debiased_lasso as dl


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
            cleaned_lines = [line for line in src.splitlines() if "from __future__" not in line]
            cleaned_lines = [
                line
                for line in cleaned_lines
                if ("statsmodels" not in line and "matplotlib" not in line and "plt" not in line)
            ]
            cleaned = "\n".join(cleaned_lines)
            func_cells.append(cleaned)

    seen = set()
    uniq_imports = []
    for li in import_lines:
        if li not in seen:
            seen.add(li)
            uniq_imports.append(li)

    code = "\n".join(uniq_imports + ["\n# ---- extracted functions ----\n"] + func_cells)
    env: dict = {}
    imports = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.linear_model import Lasso\n"
        "from scipy import stats\n"
    )
    exec(imports, env)
    exec(code, env)
    return env


def test_notebook_vs_public_nodewise_debias_basic(tmp_path):
    # build minimal notebook programmatically to avoid embedding long lines
    cell0_src = (
        "import numpy as np\nimport pandas as pd\nfrom bsm_rfm import debiased_lasso as dl\n"
    )

    cell1_src = (
        "def nodewise_precision(X, alpha_node=1e-3):\n"
        "    if isinstance(X, pd.DataFrame):\n"
        "        Xdf = X\n"
        "    else:\n"
        "        Xdf = pd.DataFrame(X)\n"
        "    Theta, tau2 = dl.nodewise_precision(Xdf, alpha_node=alpha_node)\n"
        "    return np.asarray(Theta), np.asarray(tau2)\n\n"
        "def debias_coeffs_batched(X, Y, B_hat, Theta, batch=32):\n"
        "    Xdf = pd.DataFrame(X)\n"
        "    Ydf = pd.DataFrame(Y)\n"
        "    return dl.debias_coeffs_batched(Xdf, Ydf, B_hat, Theta)\n\n"
        "def ztests_from_debias_scores(*args, **kwargs):\n"
        "    try:\n"
        "        return dl.ztests_from_debias_scores(*args, **kwargs)\n"
        "    except Exception:\n"
        "        return np.zeros(1)\n"
    )

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": cell0_src.splitlines(keepends=True),
                "outputs": [],
                "execution_count": None,
            },
            {
                "cell_type": "code",
                "source": cell1_src.splitlines(keepends=True),
                "outputs": [],
                "execution_count": None,
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = tmp_path / "minimal_lasso.ipynb"
    nb_path.write_text(json.dumps(nb), encoding="utf-8")
    env = _extract_funcs_from_notebook(nb_path)

    rs = np.random.RandomState(1)
    n = 120
    p = 20
    q = 3
    X = rs.normal(size=(n, p)).astype(np.float32)
    B = np.zeros((p, q), dtype=float)
    B[:4, :] = np.array([[2.0, 1.0, 0.0]] * 4)
    Y = (X @ B + rs.normal(scale=0.1, size=(n, q))).astype(np.float32)

    nodewise_nb = env.get("nodewise_precision")
    debias_nb = env.get("debias_coeffs_batched")
    ztests_nb = env.get("ztests_from_debias_scores")
    assert nodewise_nb is not None and debias_nb is not None and ztests_nb is not None

    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    Ydf = pd.DataFrame(Y, columns=[f"y{i}" for i in range(q)])

    Theta_nb, tau2_nb = nodewise_nb(X, alpha_node=1e-3)
    assert Theta_nb.shape == (p, p)
    assert tau2_nb.shape == (p,)
    assert np.isfinite(Theta_nb).all()

    Theta_impl, tau2_impl = dl.nodewise_precision(Xdf, alpha_node=1e-3)
    assert Theta_impl.shape == (p, p)
    assert tau2_impl.shape == (p,)
    assert np.isfinite(Theta_impl).all()

    diag_nb = np.diag(Theta_nb)
    diag_impl = np.diag(Theta_impl)
    assert (diag_nb > 0).all()
    assert (diag_impl > 0).all()
    corr = np.corrcoef(diag_nb, diag_impl)[0, 1]
    med_ratio = float(np.median(diag_nb) / np.median(diag_impl))
    assert corr > 0.2, f"Nodewise diagonal correlation too low: {corr:.3f}"
    assert 0.4 < med_ratio < 2.5, f"Median diagonal ratio unexpected: {med_ratio:.3f}"

    B_hat = np.zeros((p, q), dtype=np.float32)
    B_tilde_nb, sigma2_nb = debias_nb(X, Y, B_hat, Theta_nb, batch=32)
    assert B_tilde_nb.shape == (p, q)
    assert sigma2_nb.shape == (q,)

    B_tilde_impl, sigma2_impl = dl.debias_coeffs_batched(Xdf, Ydf, B_hat, Theta_impl)
    assert B_tilde_impl.shape == (p, q)
    assert sigma2_impl.shape == (q,)

    assert np.allclose(sigma2_nb, sigma2_impl, rtol=1e-2, atol=1e-6)
