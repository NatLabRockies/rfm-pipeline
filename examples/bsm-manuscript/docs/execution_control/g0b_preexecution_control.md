# BSM G0/B pre-execution control snapshot

**Snapshot status:** immutable implementation control
**Gate status:** G0 `OPEN`; A `OPEN`; B `OPEN`
**Phase:** development only

This tracked snapshot binds the BSM-side repair to the “G0 amendment,” “Gate B
implementation requirements,” “Preflight implementation record,” “Mandatory
clean restart,” and “Clean-restart implementation response” sections of the
current `ANALYSIS_HANDOFF.md`. It is the control source for
`configs/method_contract.yaml`, `configs/g0_seed_ledger.yaml`, the recovery
driver, and pre-execution manifests.

## Prohibitions

- Do not use rejected worktrees, cached scores, generated recovery artifacts,
  inspected seeds, or prior acceptance decisions.
- Do not start calibration, scheduler, HPC, production, holdout, or
  artifact-generation work.
- Do not overwrite an output root or create a result-looking log without
  terminal records.
- G0/A/B remain open. Development controls are not calibration evidence.

## Frozen BSM-side controls

1. Every scenario uses one typed 160-column predictor schema: 158 continuous
   fields and the two binary scenario fields. Both binary fields are candidates,
   are preserved through the driver boundary, and realize all four joint cells
   in training and independent evaluation data.
2. The contract is the only scientific-settings source. It supplies phase,
   screen and interaction draw counts, q, alpha, confidence rule, scenario
   definitions, DGP controls, failure policy, and the seed-ledger identity.
   Duplicate, absent, or unknown contract fields fail before scoring.
3. The canonical selection label is `max_stat_adjusted_p_mc`. Its adjusted-p
   rule uses a shared response-row draw schedule, `>=` ties, and alpha 0.05.
   Screening and interaction use distinct declared draw counts.
4. Null outcomes distinguish an empty candidate family from a stage failure.
   An empty family is analysis-complete, contributes zero to the false-pair
   numerator and one to the locked null denominator, and is marked
   nondegenerate=false. A one-pair family is valid and reaches the reducer.
   Any actual stage failure is terminal `FAILED` and fails the ledger.
5. The DGP is a typed truth ledger. It records canonical main, transformation,
   and interaction identifiers; coefficients; output loadings; in-library
   status; and a shared train/evaluation response surface. Heteroscedasticity
   is a bounded row-level noise-scale function of named predictors.
6. Scenario/replicate/stage seeds derive from the contract identity and named
   keys, not iteration order. The tracked ledger lists every development
   replicate. All records preserve development, exploratory, or confirmatory
   phase labels.
7. Every planned replicate has exactly one terminal record. Terminal records
   bind contract and snapshot hashes, seed, schedule identity, candidate counts,
   typed truth, retained pairs, false-pair count, status, exception, runtime,
   and resource use. Aggregates are derived exclusively from this ledger.
8. Pre-execution manifests state `NOT_EXECUTED`; they contain no calibration
   estimate, passed gate, result count, or free-text validation attestation.

## Required pre-execution tests

- snapshot and contract/ledger identity verification;
- strict contract parsing and seed order invariance;
- full-schema binary-cell coverage, ranges, correlations, row-level variance,
  typed truth, and shared train/evaluation surface checks;
- empty-family, one-pair, duplicate, missing, and failed-terminal ledger
  semantics;
- truthful manifest and legacy-claim scans.

## Promotion boundary

Only a clean, mutually pinned implementation with independent G0/A review may
start the later scheduler pilot. Confirmation remains prohibited until its
separate ledger and acceptance evidence are frozen and reviewed.
