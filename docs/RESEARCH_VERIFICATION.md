# Research (ML / Conditional-Edge) Verification

A separate verification layer for future research engines. It is isolated from the frozen NQ
strategy verifier: it does not read strategy profiles, does not import `nq_frozen`, and does not
modify `verifier/causality.py` or `verifier/static_scan.py`. It reuses their primitives
(`mutate_future`, `generate_cutoffs`, `_equal`, `scan_source`) and the repository's `Finding`,
`VerificationReport` and verdict gate.

This layer is **not** an edge engine. It only decides whether an untrusted research engine is
causal and leakage-free under the tests below.

## What it proves (when a check is PASS)

| Family | Evidence |
|---|---|
| `research_contract` | Schema, IDs, timezone-awareness, `feature_asof_time <= event_time`, `event_time <= target_start <= target_end`, target resolved within data, determinism across two identical runs, feature/target firewall. |
| `research_causality` | At every cutoff, events, features and fully resolved targets knowable before the cutoff are identical across full, truncated and future-mutated bars. |
| `walkforward` | Verifier-built folds are chronological, expanding, purged by `target_end`, disjoint, complete; every OOF prediction maps to exactly one validation observation. |
| `ml_leakage` | Validation predictions do not move when labels/features the model must not use are poisoned; positive controls show poisoning reaches the model. |
| `lockbox` | No event at/after the lockbox, and no label whose window crosses it, is passed to `fit_predict_fold` (tables and access log). |
| `ml_static_scan` | Heuristic only. WARN means manual review (verdict INCOMPLETE). Never FAIL. |

The CLI also requires the existing `repository_health`, `data_audit` and `contract_provenance`
families, exactly as the frozen strategy verifier does.

## What it does NOT prove

- It does not prove the absence of every leakage mechanism. It proves invariance at the tested
  cutoffs, folds and poisonings. A passing static scan proves nothing.
- **Output-invariant leaks are invisible to output tests.** Example: full-sample standardization
  followed by OLS with an intercept is exactly affine-invariant, so predictions cannot change
  (`tests/research_toys.py::FullSampleScalingOLS`). Only the heuristic `ml_static_scan` points at it.
- **Adaptive adversaries.** A candidate that detects poisoned values (for example, by magnitude)
  and behaves differently can evade dynamic tests. Poisoned values are drastic by design.
- **Out-of-band data.** A candidate that reads files, the network or disk caches outside the
  tables it is given is not observed. Each call gets a fresh module instance, which defeats
  module-level caches only.
- **Bar-timestamp semantics.** A bar stamped `t` is treated as known at `t`, identical to the
  existing harness. If the source stamps bars at their open, `event_time` must be the bar close
  and the verifier cannot tell the difference.
- Contract-roll correctness, data quality beyond `audit_dataset`, and statistical validity of any
  edge are out of scope. Historical PF/IC is never evidence of correctness.

## Adapter contract

```python
def events(bars) -> DataFrame        # event_id, event_time, direction  (one row per event)
def features(bars, events) -> DataFrame  # event_id, feature_asof_time, <features...>
def targets(bars, events) -> DataFrame   # event_id, target_start, target_end, <labels...>
def fit_predict_fold(features, targets, train_ids, validation_ids, target_name) -> DataFrame
    # event_id, prediction: exactly one finite prediction per validation_id  (optional)
```

- `event_time` is the earliest timestamp at which the event is fully knowable. `event_id` is
  unique and deterministic (stable under truncation). Events are returned in chronological order.
- All timestamps are timezone-aware datetime dtypes.
- Every event has exactly one feature row. Set the module attribute
  `FEATURES_ALLOW_MISSING_EVENTS = True` to document intentional omissions; they are reported
  and excluded from modelling.
- Targets are **wide** (one row per event, label columns) or **long** (`target_name` +
  `target_value`, key `(event_id, target_name)`). Unresolved targets are omitted, never clamped.
- Missing `fit_predict_fold` → `ml_leakage` and OOF alignment are UNVERIFIED, never PASS.
- A clean template is at `templates/research/adapter.py`. It is a verifier fixture, not research.

## Why targets are treated differently from features

A feature must be computable from data up to `feature_asof_time`; any dependence on later data
is a failure. A target is future information by definition: it may depend on data after the
event, up to `target_end`. Once `target_end` has passed, the target is fixed history, so a target
with `target_end < cutoff` must be invariant. Targets crossing a cutoff are not compared.

The feature/target firewall rejects label columns in the feature frame. Reserved-name matching is
centralized in `verifier/research_contracts.py::reserved_feature_reason`. It matches whole tokens
(split on separators, camelCase and digits): `target(s)`, `label(s)`, `future`, `forward`, `fwd`,
`mfe`, `mae`, and phrases `barrier_result/hit/outcome`, `next_return/ret`. `futures_basis`,
`maestro` and `targeted_vol` are allowed. A feature named like a target column, or numerically
identical to one, FAILs. A near-linear copy (|corr| ≥ 0.9999) WARNs. Conservative consequence:
known-at-event values (e.g. a reference price) must not be placed in the targets table.

## Future mutation and truncation

For each cutoff `T`, the full pipeline runs on (1) full bars, (2) bars `< T` and (3) bars with
every row `>= T` drastically mutated by `verifier.causality.mutate_future`. Compared rows:
events with `event_time < T`, their feature rows, feature rows with
`feature_asof_time < T <= event_time` (mutation only), and targets with `target_end < T`.

Cutoffs are `verifier.causality.generate_cutoffs` (distributed, warm-up, session/date transitions,
fixed-seed random) plus the first bar strictly after sampled `event_time`, `feature_asof_time`
and `target_end` values (fast/standard/strong: 2/4/6 per kind). These are the tightest cutoffs
at which a row first becomes comparable, so a one-bar peek is exposed. A candidate that raises
on a prefix yields UNVERIFIED for that cutoff. A schema change under truncation/mutation FAILs.
A stage with zero comparable rows is UNVERIFIED.

## Purged expanding walk-forward

The verifier, not the candidate, builds folds: UTC calendar-year validation blocks, expanding
training history, no shuffling. A development observation trains fold `[vs, ve)` only if

```
event_time < vs   AND   target_end < vs
```

Rows with `event_time < vs <= target_end` are **purged**: their labels contain validation-period
prices. `audit_folds` independently proves disjointness, chronology, purge, completeness of each
validation block, strictly increasing non-overlapping blocks, and lockbox exclusion. It accepts
arbitrary folds so malicious splits can be tested (`tests/test_walkforward.py`).

## Poisoning (ml_leakage)

The model receives all development observations (lockbox removed) plus explicit
`train_ids`/`validation_ids`, so misuse is detectable. Per audited fold (strong: all folds;
standard: ≤6; fast: ≤3):

| Test | Poisoned | Must hold |
|---|---|---|
| determinism | nothing (identical re-run) | identical predictions |
| `validation_label_poisoning` | all label columns and target windows of the same validation fold | validation predictions unchanged |
| `future_label_poisoning` | labels of rows after the validation block | unchanged |
| `purged_label_poisoning` | labels of purged rows | unchanged |
| `future_feature_poisoning` | features of rows after the block (not train, not validation) | unchanged |
| `purged_feature_poisoning` | features of purged rows | unchanged |
| `intra_validation_feature_poisoning` | features of later validation rows | earlier validation predictions unchanged |
| `control_train_*_sensitivity` | training labels / training features | predictions **change** (else WARN: LOW POWER) |
| `split_compliance` | summary of the forbidden-row tests | no forbidden row influenced output |

Tolerance: `|a-b| <= 1e-9 + 1e-9·|b|`. A candidate that raises only when forbidden rows are
poisoned FAILs, because its output depends on them. Validation feature values are never poisoned
for the validation tests: predictions may legitimately depend on validation X. Future/validation
label poisoning with no applicable rows in any fold is UNVERIFIED. Every call receives deep
copies, so in-place input mutation is reported as INFO.

## Lockbox

`--lockbox-start YYYY-MM-DD` (UTC midnight). In development verification:

- events with `event_time >= lockbox_start` are withheld;
- **conservative:** events before the lockbox whose target window ends at/after it are also
  withheld, because their labels use lockbox-period prices;
- a hard guard (`assert_model_tables_exclude_lockbox`) runs before **every** model call,
  including poisoned calls (poisoned target windows are kept strictly below the boundary), and
  raises `LockboxViolation` instead of warning;
- the report records `lockbox_start`, development, withheld and boundary-withheld counts, and an
  access log of every event_id given to `fit_predict_fold`.

Event and feature generation may run on the full bar history: research causality proves that
historical rows are invariant to later data. There is no lockbox performance report and no API
that returns lockbox rows for modelling. No `--lockbox-start` means `lockbox` is UNVERIFIED.

## CLI

```bash
python scripts/verify_research.py \
    --adapter candidate_engines/research_adapter.py \
    --data D:\market-data\NQ.parquet \
    --timestamp-col timestamp \
    --target forward_return_60m \
    --lockbox-start 2025-01-01 \
    --mode strong
```

Options: `--source` (scan/hash target, default: adapter), `--contract-col`,
`--expected-interval`, `--min-train-events` (default 20), `--seed` (1729), `--skip-tests`,
`--report-prefix` (default `reports/research_<adapter>`). The command writes JSON and Markdown
and prints PASS/FAIL/UNVERIFIED per family. Exit codes: `0=VERIFIED`, `1=FAILED`,
`2=INCOMPLETE / UNVERIFIED`. Exit 2 is not a pass. On continuous-contract data without
provenance, `contract_provenance` keeps the verdict at 2 even when every research family passes.

## Self-tests

`tests/research_toys.py` holds a clean candidate and deliberate cheats. Each cheat must fail for
its own reason: full-sample-mean feature, one-bar event lookahead, named and disguised target
copies, target peeking past `target_end`, feature depending on data after its declared as-of
time, as-of after event, validation-label calibration, full-sample scaling, full-sample labels,
self-selected unpurged training, random splits, nondeterminism and misaligned predictions.
Hand-built unpurged, shuffled, overlapping and lockbox-touching folds are also tested.
