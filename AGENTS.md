# AGENTS.md — Backtest Verification Repository

This repository is a verification harness first and a strategy repository second.

## Non-negotiable workflow

1. Treat each strategy's `SPECIFICATION.md`, `REQUIREMENTS.md`, and `ORACLE.md` as frozen.
2. Never change a frozen rule or expected oracle result merely to make an engine pass.
3. Treat every candidate engine as untrusted input.
4. Prefer an adapter around a candidate engine instead of rewriting the engine.
5. Run data audit before historical performance.
6. Run deterministic oracle/regression tests before real-data reconciliation.
7. Run causality/truncation and future-mutation tests before trusting PF/WR.
8. Run ledger/invariant checks, including global-position overlap.
9. If contract provenance is unavailable, report it as UNVERIFIED. Never infer that a continuous contract is correctly rolled from OHLC alone.
10. A bug fix after the baseline is frozen must begin with a failing regression test that demonstrates the violation, followed by the smallest patch.
11. Never delete or weaken a failing test without an explicit strategy-spec change.
12. Do not optimize/refactor strategy semantics during an audit.

## Required output from an audit

Produce:
- PASS / FAIL / UNVERIFIED per check family
- exact failing rows/timestamps when possible
- whether a finding is proven, heuristic, or unverifiable
- before/after hashes if a patch is made
- full pytest result
- causality result
- overlap result
- data-quality result
- contract-provenance result
- candidate/reference reconciliation result when a reference exists

## Candidate adapter contract

For a new engine, create an adapter without changing the engine. The adapter must expose:

```python
def run(bars: pandas.DataFrame) -> pandas.DataFrame:
    # Return one row per accepted trade.
```

Canonical trade columns should be mapped where available:
`signal_time, entry_time, exit_time, entry_price, exit_price, exit_reason,
initial_risk, stop_price, target_price, deadline, atr_source_end, gross_R,
cost_R, net_R, direction`.

Optional:

```python
def signals(bars: pandas.DataFrame) -> pandas.DataFrame:
    # one row per raw/eligible signal, with signal_time
```

The generic causality harness is stronger when `signals()` is supplied.

## Interpretation boundary

Generic checks can prove structural errors. Strategy-specific correctness requires a strategy profile and oracle. Contract-roll correctness requires contract-selection provenance or raw contract evidence plus an explicit roll policy.
