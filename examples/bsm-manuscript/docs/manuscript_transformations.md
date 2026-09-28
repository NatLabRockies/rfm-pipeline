# Manuscript Transformation Definitions

Formal definitions of every feature transformation applied by the BSM reduced-form pipeline. Suffixes shown match the column-name convention in `final_support_features.csv` and the coefficient matrices.

| Family | Suffix / separator | Domain | Formula | Purpose |
| ------ | ------------------ | ------ | ------- | ------- |
| `identity` | `(none)` | all real-valued first-order inputs | `f(x) = x` | First-order input passed through unchanged. |
| `quadratic` | `_squared  (also prefix quadratic_)` | all real-valued first-order inputs | `f(x) = x^2` | Detects U-shaped or saturating response. Applied AFTER input is sampled on its native LHC range. |
| `log` | `_log  (also prefix log_)` | strictly positive inputs (sample minimum > 0) | `f(x) = ln(x)` | Compresses right-skewed inputs (e.g. interest rates expressed as percentages). 'Safe' nonlinear strategy excludes inputs whose sampled range crosses zero. |
| `log1p` | `prefix log1p_` | inputs with sample minimum >= 0 (admits zero) | `f(x) = ln(1 + x)` | Log-transform variant safe for inputs whose sample range includes zero; preserves the zero point (f(0) = 0). |
| `inverse` | `_inverse` | strictly nonzero inputs (sample minimum > 0 or maximum < 0) | `f(x) = 1 / x` | Detects responses that vary with the reciprocal of a knob (e.g. effective discount factor). |
| `sqrt` | `_sqrt` | non-negative inputs (sample minimum >= 0) | `f(x) = sqrt(x)` | Detects concave responses with a square-root law. |
| `interaction` | `:` | all ordered pairs (x_a, x_b) where a != b and both pass the SHAP top-N interaction filter | `f(x_a, x_b) = x_a * x_b` | Pairwise multiplicative interactions. Pre-filtered to the top-500 SHAP-ranked pairs evaluated on the training partition only. |

## Feature naming convention

- **First-order input:** the raw input name verbatim, e.g. `AHC.PY sensi multiplier[HEFA]`.
- **Nonlinear transform:** `<family>_<input_name>`, e.g. `quadratic_AHC.PY sensi multiplier[HEFA]`, `inverse_WW.PY sensi multiplier[ManureToHTL]`.
- **Pairwise interaction:** `<input_a>:<input_b>`, e.g. `WW.PY sensi multiplier[SludgeToHTL]:WW.progress ratios commercial[SludgeToHTL]`. Each side may itself be a transformed input.

## Where the rule is enforced in code

- Catalog generation: `rfm-pipeline/scripts/generate_feature_catalog.py` with `--interaction-strategy top-shap --nonlinear-strategy safe`.
- Catalog freeze: `bsm-public-rf/configs/manuscript_case_study.yml` `candidate_library.exact_catalog_row_count: 26560`.
- Application at evaluation: design-matrix builder in `rfm_pipeline.features.transforms`.
