import numpy as np
import pandas as pd

from bsm_rfm.manuscript_stages import _build_hc3_inferential_filter_tables


def test_hc3_retains_strong_predictor():
    rs = np.random.RandomState(0)
    n = 200
    p = 3
    X = pd.DataFrame(rs.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    # make f0 a strong predictor for y0
    s2 = 5.0 * X["f0"] + rs.normal(scale=0.5, size=n)
    Y = pd.DataFrame({"y0": s2}).astype(float)

    intervals, summary = _build_hc3_inferential_filter_tables(
        X,
        Y,
        alpha=0.05,
        interval_method=("hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs"),
    )

    # ensure feature f0 is retained by HC3 filter
    retained = summary.set_index("feature_name")["hc3_retained_after_filter"].astype(bool)
    assert retained.loc["f0"]

    # ensure CI for f0,y0 excludes zero
    mask = (intervals["feature_name"] == "f0") & (intervals["output_name"] == "y0")
    assert mask.any(), "No interval found for f0,y0"
    row = intervals.loc[mask].iloc[0]
    assert bool(row["excludes_zero"]) is True
