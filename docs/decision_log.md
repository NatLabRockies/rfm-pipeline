# Decision Log

Record architecture, API, schema, workflow, validation, and dependency decisions that affect future development.

## Template

### YYYY-MM-DD — Decision title

- Context:
- Decision:
- Alternatives considered:
- Consequences:
- Tests/docs updated:

### 2026-07-18 — Generalize scenario/categorical handling; remove BSM AFSC/UAEORO from generic src

- Context: The generic rfm-pipeline hardcoded the BSM case study's two-binary
  scenario scheme (`AFSC`, `UAEORO`) across data.py (public scenario helpers +
  fixed-offset label parser), features.py (module special-case), manuscript_stages.py
  (`_legacy_normalize_factor_token`), and manuscript_runtime.py (demo fixtures).
  These are case-study literals in a repo that must stay 100% generic. Callers of
  the affected public symbols exist only inside data.py itself and the installed
  copy vendored into bsm-public-rf's env (not that repo's own source), so the
  public-API change is safe.
- Decision: Route scenario handling through generic, parameter/config-driven
  categorical mechanisms and remove all AFSC/UAEORO literals + the BSM label parser.
  - REMOVE `add_scenario_flags` + `_parse_on_off_flag` (BSM composite-label ingest;
    belongs in the case-study repo, not the generic pipeline).
  - Generalize `make_boolean_combination_labels` -> `combination_labels(frame, *, columns: Sequence[str], output_column="combination")` and
    `stratified_subset_by_boolean_combination` -> `stratified_subset_by_combination( frame, *, columns, n_per_combination, random_state=123, require_all_combinations=True)` (no hardcoded four-combination list).
  - Generalize `stratified_holdout_split(..., afsc_column, uaeoro_column)` ->
    `stratified_holdout_split(..., stratify_columns: Sequence[str] = ())`.
  - Drop the `{"AFSC","UAEORO"} -> "Scenario"` special-case in
    `canonical_module_from_factor_name` and the AFSC/UAEORO branches of
    `_legacy_normalize_factor_token`.
  - Rename manuscript demo fixtures from AFSC/UAEORO to generic categorical names.
  - Update `__init__.__all__` and config docstrings; the P0-S13/R3-S03 denylist now
    ENFORCES the absence of AFSC/UAEORO in src (removed from documented exceptions).
- Alternatives considered: (a) keep compatibility aliases — rejected (no shims per
  policy, and case-study repo source doesn't import them); (b) do it in 2 slices —
  rejected (public-API + reproduction-path risk across 6 src + 5 test files warrants
  finer slices).
- Consequences: Public API changes (function renames/removals + `__all__` edits).
  The deeper BSM fuel-pathway module taxonomy in `_legacy_module_from_factor_name`
  (AHC/CHC/OHC/OI/SE/WW/FM, "Algal Hydrocarbons", "Use AEO Reference Oil", ...) is a
  DISTINCT case-study leak, out of scope here, tracked as the next generalization
  milestone in docs/scope_backlog.md.
- Tests/docs updated: PHASE G slices in docs/WORKFLOW_MANUSCRIPT_ALIGNMENT_PLAN.md;
  test_data.py/test_features.py/test_pipeline_smoke.py/test_feature_expansion.py
  migrated to the generic API; new tests/alignment/test_G1_S0x\_\*.py per slice.
