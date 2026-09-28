# Migration record

## Source

The example was migrated from `NatLabRockies/bsm-public-rf` local `main` at
commit `3776ac5fd48b1ce0c03febbffb5f945de4fd8d64` on 2026-09-28.

The migration copied the tracked `configs/`, `docs/`, `scripts/`, `tests/`,
`tools/`, and `artifacts/` trees plus the pinned Pixi and Python manifests. It
also captured the locally regenerated PDF and SVG files from `figures/`. No
manuscript-repository working files were modified.

## Ownership mapping

| Surface                                                                   | Canonical owner after migration        |
| ------------------------------------------------------------------------- | -------------------------------------- |
| Generic reduced-form workflow code                                        | `rfm-pipeline`                         |
| BSM publication configs, scripts, tests, tables, diagnostics, and figures | `rfm-pipeline/examples/bsm-manuscript` |
| Consumable BSM coefficient bundle and inference API                       | `bsm-public-rf`                        |
| Article, cover, bibliography, and submission package                      | `bsm-public-rf-manuscript`             |

The coefficient bundle is intentionally present in both places: here as the
output of the reproducible workflow example, and in `bsm-public-rf` as the
public model release. A future metadata-to-release-matrix conversion can
replace the release copy without changing ownership of the workflow.

## Branch integration audit

Before cleanup, every published development branch in `bsm-public-rf` was an
ancestor of local `main`. The remaining non-ancestor branch tips were reviewed
against the reconciliation record in `docs/review_register.md`:

- `g11-g0-s1-contract`: earlier campaign-contract draft, explicitly
  superseded by the canonical G11/G12 chain.
- `g11-hpc-s1-hpc-package`: stale pre-submission manifest, explicitly rejected
  because it pins displaced commit identities.
- `p9-c-s1-bsm-calibration`: retired P9 calibration path, superseded by the
  G11/G12 campaign and preserved in the audit history.
- `reconcile/base-worktree-snapshot-20260907-060905`: safety snapshot of the
  pre-reconciliation tree; reviewed changes were already resolved onto main.
- `reconcile/g10-repair-wip-20260907`,
  `reconcile/g11-hpc-s1-wip-20260907`, and
  `reconcile/g12-publication-wip-20260907`: preservation-only WIP snapshots.
  Their production-relevant successors are on main; the two absent early HPC
  prototype files were superseded by the final campaign workflow and were not
  promoted as canonical code.

No unmerged branch supplied a newer public model artifact than the bundle
migrated from `main`.
