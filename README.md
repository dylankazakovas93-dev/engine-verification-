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

## Repository health

```bash
python scripts/run_all.py
```

## One-command engine verification

```bash
python scripts/verify.py \
  --strategy nq_frozen_v1 \
  --candidate strategies/nq_frozen_v1/adapter.py \
  --source nq_frozen \
  --data "D:\\market-data\\1Min_NQ.csv" \
  --timestamp-col timestamp \
  --high-res-data "C:\\Users\\me\\Downloads\\NQ.parquet" \
  --high-res-timestamp-col ts_event \
  --high-res-contract-col instrument_id \
  --high-res-symbol-col symbol
```

The command runs configured health/data/source/ledger/causality/reference/provenance gates
and writes durable JSON and Markdown reports under `reports/`. Exit codes are:

- `0`: `VERIFIED`
- `1`: `FAILED` (a reproducible mandatory violation)
- `2`: `INCOMPLETE / UNVERIFIED` (required evidence is missing)

An LLM must never translate exit code 2 into “everything passed.”

## Research (ML / conditional-edge) verification

Research engines use a separate, isolated verifier: adapter contract
`events` / `features` / `targets` / `fit_predict_fold`, future-mutation causality, verifier-owned
purged walk-forward folds, ML leakage poisoning and lockbox isolation. See
`docs/RESEARCH_VERIFICATION.md` and `templates/research/adapter.py`.

```bash
python scripts/verify_research.py --adapter candidate_engines/research_adapter.py \
  --data /path/to/bars.parquet --timestamp-col timestamp --target forward_return_60m \
  --lockbox-start 2025-01-01 --mode strong
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

CSV and Parquet are supported. A persistent streaming manifest can be written without modifying data:

```bash
python scripts/audit_data.py C:\\data\\NQ.parquet \
  --timestamp-col ts_event --expected-interval 1s \
  --contract-col instrument_id --symbol-col symbol \
  --allow-nonpositive-prices --manifest-out reports/nq_1s_manifest.json
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

## Causality adapter stages

`run(bars)` is required. Adapters should expose as many independent stages as genuinely available:
`features`, `signals`, `eligible_signals`, `proposed_entries`, `accepted_entries`, and `trades`.
`audit_stages(bars)` can return all stages in one engine invocation. Strong mode uses distributed,
warmup, event-adjacent, session-boundary, and fixed-seed random cutoffs for truncation and future mutation.

A passing static AST/regex scan is only heuristic evidence; it does not prove no leakage.

## New strategy workflow

```text
idea → frozen SPECIFICATION → numbered REQUIREMENTS → reviewed ORACLE
→ deterministic tests → production engine → independent reference where required
→ generic/strategy/data/causality reconciliation → machine verdict → performance
```

Create an isolated scaffold with `python scripts/new_strategy.py my_strategy`. Do not implement
the engine until its specification and oracle are frozen.

## Existing engine workflow

Preserve the engine under `candidate_engines/`, hash and scan it, write only an adapter, audit
the exact data, validate ledger/state/causality, then reconcile against a strategy oracle/reference.
Every proven bug becomes a failing regression before the smallest patch.

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
