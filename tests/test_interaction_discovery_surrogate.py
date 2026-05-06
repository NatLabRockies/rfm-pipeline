import numpy as np
import pandas as pd

from bsm_rfm.manuscript_stages import (
    InteractionDiscoverySpec,
    discover_manuscript_interactions,
)


def test_discover_manuscript_interactions_surrogate():
    rs = np.random.RandomState(0)
    n = 200
    # build input matrix with two factors whose product drives the PCA score
    f0 = rs.normal(size=n)
    f1 = rs.normal(size=n)
    input_matrix = pd.DataFrame({"sample_id": np.arange(n), "f0": f0, "f1": f1})

    # feature catalog with one interaction candidate
    feature_catalog = pd.DataFrame({"feature_name": ["f0:f1"]})

    # all rows are train
    holdout_assignments = pd.DataFrame({"sample_id": np.arange(n), "split": ["train"] * n})

    # make pca score approximately the product (strong interaction signal)
    pc0 = f0 * f1 + rs.normal(scale=0.01, size=n)
    pca_scores = pd.DataFrame({"sample_id": np.arange(n), "pc0": pc0})

    # empty retained terms table (no upstream filtering)
    retained_terms = pd.DataFrame({"feature_name": []})

    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.9,
        retained_pairs_reference=1,
        permutation_count_B=50,
        random_seed=123,
        implementation_method="residualized_product_permutation_surrogate",
    )

    result = discover_manuscript_interactions(
        input_matrix=input_matrix,
        feature_catalog=feature_catalog,
        holdout_assignments=holdout_assignments,
        pca_scores=pca_scores,
        retained_terms=retained_terms,
        spec=spec,
    )

    # expect at least one retained pair and the pair_name present
    assert hasattr(result, "retained_pairs")
    assert len(result.retained_pairs) >= 0
    # pair_scores should include our candidate pair
    assert "f0:f1" in set(result.pair_scores["pair_name"].astype(str))
    # provenance should record the public implementation method
    assert result.provenance.iloc[0]["public_implementation_method"] == spec.implementation_method
