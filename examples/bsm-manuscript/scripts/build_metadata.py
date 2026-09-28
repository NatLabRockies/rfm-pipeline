"""Build comprehensive input/output metadata for the BSM reduced-form model.

Sources of truth:
  1. Technical report docx (BSM Reduced Form FY25Q4 Report_20250930.docx),
     Appendix A (Table A1 — inputs) and Appendix B (Table B1 — outputs).
  2. Final model artifacts in bsm-public-rf/artifacts/final_model/:
       - final_support_features.csv  (canonical column order of the
         132-feature reduced-form coefficient matrix)
       - coefficient_matrix_raw_scale.csv  (the coefficient table itself)
       - x_standardization.csv, y_standardization.csv

Produced (machine-readable YAML + human-readable Markdown):
  bsm-public-rf/configs/manuscript_input_metadata.yml
  bsm-public-rf/configs/manuscript_output_metadata.yml
  bsm-public-rf/configs/manuscript_transformations.yml
  bsm-public-rf/configs/manuscript_module_abbreviations.yml
  bsm-public-rf/docs/manuscript_input_metadata.md
  bsm-public-rf/docs/manuscript_output_metadata.md
  bsm-public-rf/docs/manuscript_transformations.md
  bsm-public-rf/docs/manuscript_coefficient_column_order.md

Re-run with:
  pixi run python scripts/build_metadata.py
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

_DOCX_RELPATH = "docs/manuscripts/BSM Reduced Form FY25Q4 Report_20250930.docx"

# Repo root is derived from this file's location so the script works from any
# checkout. The technical-report docx lives outside this repository; point at
# it with BSM_SOURCE_DOCX, or at its containing repo with RFM_PIPELINE_ROOT.
BSM_REPO = Path(__file__).resolve().parents[1]
RFM_ROOT = Path(
    os.environ.get("RFM_PIPELINE_ROOT", BSM_REPO.parent / "rfm-pipeline")
).expanduser()
DOCX = Path(
    os.environ.get("BSM_SOURCE_DOCX", RFM_ROOT / _DOCX_RELPATH)
).expanduser()
FINAL = BSM_REPO / "artifacts/final_model"
CONFIGS = BSM_REPO / "configs"
DOCS = BSM_REPO / "docs"

# Module abbreviations — established from technical report appendix A and
# input naming convention. Appendix C in the docx is a title only; this
# table reconstructs the abbreviations referenced throughout the document.
MODULE_ABBREV = {
    "AHC": {
        "name": "Algal Hydrocarbons",
        "description": "Conversion pathways producing hydrocarbon fuels from algal feedstocks (HEFA, HTL).",
    },
    "CHC": {
        "name": "Cellulosic Hydrocarbons",
        "description": "Conversion pathways producing hydrocarbon fuels from cellulosic feedstocks (Biochem, Thermochem, Brownfield).",
    },
    "OHC": {
        "name": "Oil Hydrocarbons",
        "description": "Conversion pathways producing hydrocarbon fuels from oilcrop and FOG (fats/oils/greases) feedstocks (HEFA, HEFA-Brownfield).",
    },
    "WW": {
        "name": "Wet Waste Hydrocarbons",
        "description": "Conversion pathways producing hydrocarbon fuels from wet-waste feedstocks (ManureToHTL, SludgeToHTL).",
    },
    "SE": {
        "name": "Starch Ethanol",
        "description": "Conversion pathways producing ethanol and ethanol-to-jet fuel from starch feedstocks.",
    },
    "SC": {
        "name": "Starch / Cellulosic Ethanol-to-Jet",
        "description": "Pathway-specific knobs for starch and cellulosic ethanol-to-jet conversion.",
    },
    "OI": {
        "name": "Output Indicators",
        "description": "Aggregate output variables (production, MFSP, carbon intensity) reported by the BSM across all conversion platforms.",
    },
    "FM": {
        "name": "Feedstock Module",
        "description": "Cross-cutting feedstock allocation and policy knobs (binary scenario switches).",
    },
}

# Transformation definitions — formal math for the four nonlinear families
# plus pairwise interactions. Names match the suffix convention used in the
# feature catalog and final_support_features.csv.
TRANSFORMATIONS = {
    "identity": {
        "suffix": "",
        "domain": "all real-valued first-order inputs",
        "formula": "f(x) = x",
        "purpose": "First-order input passed through unchanged.",
    },
    "quadratic": {
        "suffix": "_squared  (also prefix quadratic_)",
        "domain": "all real-valued first-order inputs",
        "formula": "f(x) = x^2",
        "purpose": "Detects U-shaped or saturating response. Applied AFTER input is sampled on its native LHC range.",
    },
    "log": {
        "suffix": "_log  (also prefix log_)",
        "domain": "strictly positive inputs (sample minimum > 0)",
        "formula": "f(x) = ln(x)",
        "purpose": "Compresses right-skewed inputs (e.g. interest rates expressed as percentages). 'Safe' nonlinear strategy excludes inputs whose sampled range crosses zero.",
    },
    "log1p": {
        "suffix": "prefix log1p_",
        "domain": "inputs with sample minimum >= 0 (admits zero)",
        "formula": "f(x) = ln(1 + x)",
        "purpose": "Log-transform variant safe for inputs whose sample range includes zero; preserves the zero point (f(0) = 0).",
    },
    "inverse": {
        "suffix": "_inverse",
        "domain": "strictly nonzero inputs (sample minimum > 0 or maximum < 0)",
        "formula": "f(x) = 1 / x",
        "purpose": "Detects responses that vary with the reciprocal of a knob (e.g. effective discount factor).",
    },
    "sqrt": {
        "suffix": "_sqrt",
        "domain": "non-negative inputs (sample minimum >= 0)",
        "formula": "f(x) = sqrt(x)",
        "purpose": "Detects concave responses with a square-root law.",
    },
    "interaction": {
        "separator": ":",
        "domain": "all ordered pairs (x_a, x_b) where a != b and both pass the SHAP top-N interaction filter",
        "formula": "f(x_a, x_b) = x_a * x_b",
        "purpose": "Pairwise multiplicative interactions. Pre-filtered to the top-500 SHAP-ranked pairs evaluated on the training partition only.",
    },
}


def extract_docx_text(docx: Path) -> str:
    """Pull paragraph text out of a .docx WITHOUT pandoc.

    Returns one paragraph per line, in document order.
    """
    z = zipfile.ZipFile(docx)
    xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    text = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<w:tab/>", "\t", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n+", "\n", text).strip()
    return text


INPUT_NAME_RE = re.compile(r"^([A-Z]{2,3})\.\S")
NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?(e[-+]?\d+)?$")
UNIT_HINTS = ("unitless", "%/yr", "$/", "USD", "Gal", "g/MJ", "yr", "-", "fraction")


def parse_inputs(report_text: str) -> list[dict]:
    """Parse Appendix A Table A1 into structured input records.

    Each input is a multi-line block:
      line 1: input name (matches MODULE.something[...] pattern)
      lines 2..: description (optional, single line),
                 pathway impacted, min, max, units (optional - inherits)
    Continuation rows (same parameter, different pathway tag) omit
    description and sometimes units; values inherit from the most recent
    fully-specified row whose name matches up to the bracketed pathway tag.
    """
    lines = report_text.split("\n")
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "Units") + 1
        end = next(i for i, ln in enumerate(lines) if ln.startswith("Table A1."))
    except StopIteration as e:
        raise RuntimeError("Could not locate Appendix A boundaries") from e

    block = [ln.rstrip() for ln in lines[start:end] if ln.strip()]

    # Walk: a new input begins at a line matching the module-prefix regex.
    # Capture everything until the next match.
    records: list[dict] = []
    i = 0
    last_full: dict | None = None  # for inheritance of description/units
    while i < len(block):
        if not INPUT_NAME_RE.match(block[i]):
            i += 1
            continue
        name = block[i]
        # Collect this block: lines until next input name (exclusive)
        j = i + 1
        while j < len(block) and not INPUT_NAME_RE.match(block[j]):
            j += 1
        body = block[i + 1 : j]
        i = j

        rec = parse_input_body(name, body, last_full)
        records.append(rec)
        if rec.get("description") and rec.get("units"):
            last_full = rec
    return records


def parse_input_body(name: str, body: list[str], inherit_from: dict | None) -> dict:
    """Parse the body lines after an input name.

    Possible shapes (in order of appearance in the docx):
      5 fields: description, pathway, min, max, units
      4 fields: pathway, min, max, units   (description inherited)
      4 fields: description, pathway, min, max  (units inherited)
      3 fields: pathway, min, max  (description AND units inherited)
    """
    # Find the two numeric fields = min, max
    nums: list[tuple[int, float]] = [
        (k, float(v)) for k, v in enumerate(body) if NUMERIC_RE.match(v)
    ]
    rec: dict = {"name": name, "module": name.split(".", 1)[0]}
    if len(nums) >= 2:
        min_idx, min_val = nums[0]
        max_idx, max_val = nums[1]
        rec["min_sample_value"] = min_val
        rec["max_sample_value"] = max_val
        # Everything before min_idx = description and/or pathway
        head = body[:min_idx]
        # Pathway is always the LAST non-numeric line before min
        rec["pathway_impacted"] = head[-1] if head else None
        # Description is anything before that
        desc_parts = head[:-1]
        rec["description"] = " ".join(desc_parts).strip() or (
            inherit_from["description"] if inherit_from else None
        )
        # Units (if present) is the line(s) after max_idx
        tail = body[max_idx + 1 :]
        if tail:
            rec["units"] = " ".join(tail).strip()
        else:
            rec["units"] = inherit_from["units"] if inherit_from else None
    else:
        # No numeric values: binary scenario input (e.g. FM.Use Agnostic FS Conversion)
        # Body shape: description, pathway, units
        desc = body[0] if body else None
        pathway = body[1] if len(body) > 1 else None
        units = body[2] if len(body) > 2 else None
        rec.update(
            {
                "description": desc,
                "pathway_impacted": pathway,
                "units": units,
                "is_binary_scenario": True,
            }
        )
    return rec


def parse_outputs(report_text: str) -> list[dict]:
    """Parse Appendix B Table B1 into structured output records.

    Format:
      Variable
      Description
      Unit
      <name>
      <description (one paragraph)>
      <unit (one paragraph; may include arraying spec)>
      ...
    Ends at 'Table B1.' line.
    """
    lines = report_text.split("\n")
    try:
        start = (
            next(
                i
                for i, ln in enumerate(lines)
                if ln.strip() == "Appendix B: Reduced Form Model Target Outputs"
            )
            + 1
        )
        end = next(i for i, ln in enumerate(lines) if ln.startswith("Table B1."))
    except StopIteration as e:
        raise RuntimeError("Could not locate Appendix B boundaries") from e
    block = [ln.strip() for ln in lines[start:end] if ln.strip()]
    # Drop column-header rows
    while block and block[0] in {"Variable", "Description", "Unit"}:
        block.pop(0)
    # Each output = exactly 3 lines (name, desc, unit)
    records: list[dict] = []
    for i in range(0, len(block) - 2, 3):
        name = block[i]
        desc = block[i + 1]
        unit = block[i + 2]
        if not re.match(r"^[A-Z]{2,3}\.", name):
            continue
        # Parse arraying spec from unit string
        # e.g. "Gal/yr. Arrayed by conversion pathway, region, and product (i.e., gasoline, diesel, jet)"
        unit_clean, array_spec = _split_unit_and_array(unit)
        records.append(
            {
                "name": name,
                "module": name.split(".", 1)[0],
                "description": desc,
                "unit": unit_clean,
                "array_specification": array_spec,
            }
        )
    return records


def _split_unit_and_array(unit_field: str) -> tuple[str, str | None]:
    parts = unit_field.split(". ", 1)
    if len(parts) == 1:
        return unit_field.rstrip("."), None
    return parts[0].rstrip("."), parts[1].rstrip(".")


def load_coefficient_column_order() -> pd.DataFrame:
    """Read the canonical 132-feature column order from final_support_features.csv."""
    df = pd.read_csv(FINAL / "final_support_features.csv")
    return df.sort_values("final_support_position").reset_index(drop=True)


def cross_reference_inputs(coef_features: pd.DataFrame, inputs: list[dict]) -> pd.DataFrame:
    """Map each coefficient column back to its base input(s) + transformation."""
    input_by_name = {i["name"]: i for i in inputs}
    prefixes = (
        ("quadratic_", "quadratic"),
        ("log1p_", "log1p"),
        ("log_", "log"),
        ("inverse_", "inverse"),
        ("sqrt_", "sqrt"),
    )
    suffixes = (
        ("_squared", "quadratic"),
        ("_quadratic", "quadratic"),
        ("_log1p", "log1p"),
        ("_sqrt", "sqrt"),
        ("_log", "log"),
        ("_inv", "inverse"),
        ("_sq", "quadratic"),
    )
    rows = []
    for _, r in coef_features.iterrows():
        fname = r["feature_name"]
        ftype = r["feature_type"]
        bases: list[str] = []
        transform: str | None = None
        if ":" in fname:
            a, b = fname.split(":", 1)
            bases = [_strip_transform(a), _strip_transform(b)]
            transform = "interaction"
        else:
            for pref, label in prefixes:
                if fname.startswith(pref):
                    transform = label
                    bases = [fname[len(pref) :]]
                    break
            if transform is None:
                for suf, label in suffixes:
                    if fname.endswith(suf):
                        transform = label
                        bases = [fname[: -len(suf)]]
                        break
            if transform is None:
                transform = "identity"
                bases = [fname]
        rows.append(
            {
                "coefficient_column_index": int(r["final_support_position"]),
                "feature_name": fname,
                "feature_type": ftype,
                "transformation": transform,
                "base_input_1": bases[0] if len(bases) > 0 else None,
                "base_input_1_unit": (input_by_name.get(bases[0]) or {}).get("units"),
                "base_input_1_min": (input_by_name.get(bases[0]) or {}).get("min_sample_value"),
                "base_input_1_max": (input_by_name.get(bases[0]) or {}).get("max_sample_value"),
                "base_input_1_pathway": (input_by_name.get(bases[0]) or {}).get("pathway_impacted"),
                "base_input_1_description": (input_by_name.get(bases[0]) or {}).get("description"),
                "base_input_2": bases[1] if len(bases) > 1 else None,
                "base_input_2_unit": (input_by_name.get(bases[1]) or {}).get("units")
                if len(bases) > 1
                else None,
                "n_outputs_nonzero": int(r["n_outputs_excluding_zero"]),
                "stability_selection_frequency": float(r["stability_selection_frequency"]),
                "hc3_retained": bool(r["hc3_retained_after_filter"]),
                "delta_nrmse_if_removed": float(r["delta_nrmse_when_feature_removed"]),
            }
        )
    return pd.DataFrame(rows)


def _strip_transform(fname: str) -> str:
    for p in ("quadratic_", "log1p_", "log_", "inverse_", "sqrt_"):
        if fname.startswith(p):
            return fname[len(p) :]
    for s in ("_squared", "_quadratic", "_log1p", "_sqrt", "_log", "_inv", "_sq"):
        if fname.endswith(s):
            return fname[: -len(s)]
    return fname


def write_yaml(path: Path, obj: dict | list, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# {comment}\n")
        f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("# Source: scripts/build_metadata.py (rfm-pipeline)\n#\n")
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True, width=120)


def render_inputs_md(inputs: list[dict]) -> str:
    by_module: dict[str, list[dict]] = {}
    for r in inputs:
        by_module.setdefault(r["module"], []).append(r)
    out = ["# Manuscript Input Metadata", ""]
    out.append(
        "Comprehensive metadata for every first-order input to the BSM reduced-form model, "
        "derived from Appendix A (Table A1) of the FY25Q4 technical report."
    )
    out.append("")
    out.append(
        f"**Inputs in this catalog:** {len(inputs)} ({sum(1 for r in inputs if r.get('is_binary_scenario'))} binary scenario switches + {sum(1 for r in inputs if not r.get('is_binary_scenario'))} continuous LHC-sampled inputs)."
    )
    out.append("")
    out.append(
        "**Source of truth.** Variable names, descriptions, pathway tags, sample ranges, and units come directly from Appendix A Table A1. Module abbreviations come from `manuscript_module_abbreviations.yml`. Inputs are grouped by module prefix in document order."
    )
    out.append("")
    for mod, recs in by_module.items():
        meta = MODULE_ABBREV.get(mod, {})
        out.append(f"## Module `{mod}` — {meta.get('name', '(unknown module)')}")
        out.append("")
        if meta.get("description"):
            out.append(meta["description"])
            out.append("")
        out.append("| # | Input name | Description | Pathway | Min | Max | Units |")
        out.append("| - | ---------- | ----------- | ------- | --- | --- | ----- |")
        for i, r in enumerate(recs, 1):
            desc = (r.get("description") or "").replace("|", "\\|").replace("\n", " ")
            out.append(
                f"| {i} | `{r['name']}` | {desc} | "
                f"{(r.get('pathway_impacted') or '').replace('|', '\\|')} | "
                f"{r.get('min_sample_value', '—')} | "
                f"{r.get('max_sample_value', '—')} | "
                f"{r.get('units') or '—'} |"
            )
        out.append("")
    return "\n".join(out)


def render_outputs_md(outputs: list[dict]) -> str:
    out = ["# Manuscript Output Metadata", ""]
    out.append(
        "Comprehensive metadata for every base output variable predicted by the BSM "
        "reduced-form model, derived from Appendix B (Table B1) of the FY25Q4 "
        "technical report. The 17 base outputs expand into 23,495 scalar time-series "
        "outputs after arraying and annualization (2015-2051)."
    )
    out.append("")
    out.append("| # | Variable | Description | Unit | Array specification |")
    out.append("| - | -------- | ----------- | ---- | ------------------- |")
    for i, r in enumerate(outputs, 1):
        d = r["description"].replace("|", "\\|")
        out.append(
            f"| {i} | `{r['name']}` | {d} | {r['unit'].replace('|', '\\|')} | "
            f"{(r.get('array_specification') or '—').replace('|', '\\|')} |"
        )
    out.append("")
    out.append("## Output expansion convention")
    out.append("")
    out.append(
        "Each base output is arrayed across one or more discrete dimensions "
        "(conversion pathway, region, product). When the arrayed scalars are "
        "evaluated for each of the 37 annual values (2015 – 2051), the total "
        "output space is **23,495 scalars**. Of these, **9,954** exhibit "
        "non-negligible variance under the LHC input sweep and are scored by "
        "the manuscript's reported nRMSE; the remaining outputs are predicted "
        "with near-zero coefficients (constant or near-constant series)."
    )
    return "\n".join(out)


def render_transformations_md() -> str:
    out = ["# Manuscript Transformation Definitions", ""]
    out.append(
        "Formal definitions of every feature transformation applied by the BSM "
        "reduced-form pipeline. Suffixes shown match the column-name convention "
        "in `final_support_features.csv` and the coefficient matrices."
    )
    out.append("")
    out.append("| Family | Suffix / separator | Domain | Formula | Purpose |")
    out.append("| ------ | ------------------ | ------ | ------- | ------- |")
    for k, t in TRANSFORMATIONS.items():
        marker = t.get("suffix", t.get("separator", ""))
        out.append(
            f"| `{k}` | `{marker or '(none)'}` | {t['domain']} | `{t['formula']}` | {t['purpose']} |"
        )
    out.append("")
    out.append("## Feature naming convention")
    out.append("")
    out.append(
        "- **First-order input:** the raw input name verbatim, e.g. "
        "`AHC.PY sensi multiplier[HEFA]`.\n"
        "- **Nonlinear transform:** `<family>_<input_name>`, e.g. "
        "`quadratic_AHC.PY sensi multiplier[HEFA]`, `inverse_WW.PY sensi multiplier[ManureToHTL]`.\n"
        "- **Pairwise interaction:** `<input_a>:<input_b>`, e.g. "
        "`WW.PY sensi multiplier[SludgeToHTL]:WW.progress ratios commercial[SludgeToHTL]`. "
        "Each side may itself be a transformed input.\n"
    )
    out.append("## Where the rule is enforced in code")
    out.append("")
    out.append(
        "- Catalog generation: `rfm-pipeline/scripts/generate_feature_catalog.py` "
        "with `--interaction-strategy top-shap --nonlinear-strategy safe`.\n"
        "- Catalog freeze: `bsm-public-rf/configs/manuscript_case_study.yml` "
        "`candidate_library.exact_catalog_row_count: 26560`.\n"
        "- Application at evaluation: design-matrix builder in "
        "`rfm_pipeline.features.transforms`.\n"
    )
    return "\n".join(out)


def render_coefficient_column_order_md(xref: pd.DataFrame) -> str:
    out = ["# Coefficient Matrix Column Order", ""]
    out.append(
        "Canonical column order for the 132-feature reduced-form coefficient "
        "matrix (`artifacts/final_model/coefficient_matrix_raw_scale.csv` and "
        "`coefficient_matrix_standardized.csv`). Each row pairs the column "
        "position with its feature name, transformation, base input(s), "
        "units, sample range, and the feature's contribution metrics."
    )
    out.append("")
    out.append(
        f"**Total columns:** {len(xref)}. **Source of truth for column "
        f"order:** `artifacts/final_model/final_support_features.csv` "
        f"(field `final_support_position`)."
    )
    out.append("")
    out.append(
        "| Idx | Feature name | Transform | Base input 1 | Unit | Min | Max | Pathway | n_out≠0 | Δ nRMSE if removed |"
    )
    out.append(
        "| --- | ------------ | --------- | ------------ | ---- | --- | --- | ------- | ------- | ------------------ |"
    )
    for _, r in xref.iterrows():
        out.append(
            f"| {r['coefficient_column_index']} | "
            f"`{r['feature_name']}` | {r['transformation']} | "
            f"`{r['base_input_1']}` | {r['base_input_1_unit'] or '—'} | "
            f"{r['base_input_1_min'] if pd.notna(r['base_input_1_min']) else '—'} | "
            f"{r['base_input_1_max'] if pd.notna(r['base_input_1_max']) else '—'} | "
            f"{r['base_input_1_pathway'] or '—'} | "
            f"{r['n_outputs_nonzero']} | "
            f"{r['delta_nrmse_if_removed']:.4f} |"
        )
    out.append("")
    out.append("## How to use this table")
    out.append("")
    out.append(
        "1. **Reading the coefficient matrix:** column at index `i` of the "
        "coefficient matrix corresponds to the feature at row `i` of this table.\n"
        "2. **Reading a coefficient value:** the raw-scale coefficient maps to "
        "the base input in its native units; the standardized coefficient maps "
        "to the base input rescaled by its training-partition mean and standard "
        "deviation (see `x_standardization.csv`, `y_standardization.csv`).\n"
        "3. **Interaction columns:** the product `x_a * x_b` is computed AFTER "
        "any per-side transformation (e.g. `quadratic_X:Y` = `X^2 * Y`).\n"
        "4. **Unit conventions:** any unitless input has `units = 'unitless'`. "
        "Interactions and transformed inputs inherit the product/composition "
        "of their constituent units; the raw coefficient value carries the "
        "inverse of those units to recover the output unit.\n"
    )
    return "\n".join(out)


def main() -> None:
    if not BSM_REPO.exists():
        raise SystemExit(f"bsm-public-rf repo not found at {BSM_REPO}")
    if not DOCX.is_file():
        raise SystemExit(
            f"Source technical report not found at {DOCX}.\n"
            "This document is not distributed with this repository. Set "
            "BSM_SOURCE_DOCX to its path, or RFM_PIPELINE_ROOT to the root of "
            "a checkout containing it."
        )
    print(f"Extracting text from {DOCX.name} ...")
    text = extract_docx_text(DOCX)
    inputs = parse_inputs(text)
    outputs = parse_outputs(text)
    print(f"  parsed {len(inputs)} inputs, {len(outputs)} outputs")

    print("Loading coefficient column order ...")
    coef = load_coefficient_column_order()
    print(f"  {len(coef)} coefficient columns")

    print("Cross-referencing features → base inputs ...")
    xref = cross_reference_inputs(coef, inputs)
    # Coverage check
    unmatched = xref[xref["base_input_1_unit"].isna()]
    print(
        f"  {len(unmatched)}/{len(xref)} features have no input-unit match (expected: 0 for clean alignment)"
    )
    if len(unmatched):
        print(
            unmatched[["coefficient_column_index", "feature_name", "base_input_1"]]
            .head(10)
            .to_string(index=False)
        )

    print("Writing YAML metadata ...")
    write_yaml(
        CONFIGS / "manuscript_input_metadata.yml",
        {
            "n_inputs": len(inputs),
            "source": "Appendix A Table A1, BSM FY25Q4 Report",
            "inputs": inputs,
        },
        "Manuscript input metadata. Source: docs/manuscripts/BSM Reduced Form FY25Q4 Report_20250930.docx Appendix A.",
    )
    write_yaml(
        CONFIGS / "manuscript_output_metadata.yml",
        {
            "n_outputs_base": len(outputs),
            "n_outputs_scalar": 23495,
            "n_outputs_scored": 9954,
            "source": "Appendix B Table B1, BSM FY25Q4 Report",
            "outputs": outputs,
        },
        "Manuscript output metadata. Source: docs/manuscripts/BSM Reduced Form FY25Q4 Report_20250930.docx Appendix B.",
    )
    write_yaml(
        CONFIGS / "manuscript_transformations.yml",
        {"transformations": TRANSFORMATIONS},
        "Formal feature-transformation definitions. Match the suffix convention used in final_support_features.csv.",
    )
    write_yaml(
        CONFIGS / "manuscript_module_abbreviations.yml",
        {"modules": MODULE_ABBREV},
        "Module abbreviations referenced throughout the BSM FY25Q4 technical report.",
    )

    print("Writing Markdown deliverables ...")
    (DOCS / "manuscript_input_metadata.md").write_text(render_inputs_md(inputs))
    (DOCS / "manuscript_output_metadata.md").write_text(render_outputs_md(outputs))
    (DOCS / "manuscript_transformations.md").write_text(render_transformations_md())
    (DOCS / "manuscript_coefficient_column_order.md").write_text(
        render_coefficient_column_order_md(xref)
    )

    # Also dump the cross-reference as a machine-readable CSV
    xref.to_csv(FINAL / "coefficient_column_metadata.csv", index=False)

    # Summary report
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_docx": str(DOCX),
        "outputs": {
            "yaml": [
                "configs/manuscript_input_metadata.yml",
                "configs/manuscript_output_metadata.yml",
                "configs/manuscript_transformations.yml",
                "configs/manuscript_module_abbreviations.yml",
            ],
            "markdown": [
                "docs/manuscript_input_metadata.md",
                "docs/manuscript_output_metadata.md",
                "docs/manuscript_transformations.md",
                "docs/manuscript_coefficient_column_order.md",
            ],
            "csv": [
                "artifacts/final_model/coefficient_column_metadata.csv",
            ],
        },
        "counts": {
            "inputs_parsed": len(inputs),
            "inputs_binary": sum(1 for r in inputs if r.get("is_binary_scenario")),
            "inputs_continuous": sum(1 for r in inputs if not r.get("is_binary_scenario")),
            "outputs_parsed": len(outputs),
            "coefficient_columns": len(coef),
            "coefficient_columns_with_input_match": int(xref["base_input_1_unit"].notna().sum()),
        },
    }
    (DOCS / "manuscript_metadata_build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary: {json.dumps(summary['counts'], indent=2)}")


if __name__ == "__main__":
    main()
