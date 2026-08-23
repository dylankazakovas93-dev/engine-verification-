# Backtest Verification Lab — NQ Frozen v1

A reusable local verification repository for trading engines.

The repository is designed so a future Claude/Codex/ChatGPT agent can bring a candidate engine, adapt its output into a canonical trade ledger, and run a fixed battery of tests before anyone looks at profit factor.

## What this catches

### Generic, strategy-independent checks
- malformed / non-chronological / duplicated OHLCV
- silent missing-data synthesis risk
- impossible OHLC geometry
- nonfinite or negative values
- future-dependence through truncation invariance
- future-mutation leakage
- more than one global open position
- exit before entry
- trades past their declared deadline
- noncausal ATR/source timestamps
- nonpositive initial risk
- inconsistent gross/cost/net R arithmetic
- suspicious source-code patterns such as negative shifts, backfill, centered rolling windows, forward `merge_asof`, and explicit future slices
- missing contract provenance and suspicious contract switches when `contract_id` exists

### NQ Frozen v1 strategy checks
The original frozen strategy remains in:
- `strategies/nq_frozen_v1/SPECIFICATION.md`
- `strategies/nq_frozen_v1/REQUIREMENTS.md`
- `strategies/nq_frozen_v1/ORACLE.md`
- the deterministic pytest suite under `tests/`
- the clean candidate/reference implementations under `nq_frozen/`

## Important limitation

A continuous OHLCV series **cannot prove its own rollover construction**. If there is no `contract_id`, selection ledger, or raw per-contract source, the harness reports rollover provenance as **UNVERIFIED**, not PASS.

## Install

```bash
python -m pip install -e '.[test]'
```

## One-command health check of this repository

```bash
python scripts/run_all.py
```

## Audit an arbitrary candidate engine

1. Put the engine anywhere under `candidate_engines/`.
2. Create a small adapter exposing `run(bars) -> trade ledger`.
3. Run:

```bash
python scripts/audit_candidate.py   --data /path/to/1Min_NQ.csv   --adapter /path/to/adapter.py   --source /path/to/engine.py
```

For a strategy with a reference engine, also run its strategy-specific reconciliation.

## Audit data only

```bash
python scripts/audit_data.py /path/to/1Min_NQ.csv
```

If the file has a contract column:

```bash
python scripts/audit_data.py /path/to/file.csv --contract-col contract_id
```

## Current frozen NQ candidate

```bash
pytest -q
python verify_real_data.py /path/to/1Min_NQ.csv --timestamp-col timestamp
```

Do not trust performance until the tests, generic audit, causality checks, and reference reconciliation all pass.

## Change control

A post-baseline engine fix requires:
1. a reproducible failing case,
2. a new regression test that fails before the fix,
3. the smallest patch,
4. the new test passing,
5. all old tests still passing,
6. causality checks passing,
7. real-data reconciliation passing,
8. trade-ledger hash changes explained.
