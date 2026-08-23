# Local Codex Handoff

You are operating locally on the user's computer with filesystem and terminal access.

## Objective

Finish and maintain this repository as the permanent backtest-verification lab. Do not treat the existing NQ engine as trusted merely because it exists.

## First actions

1. Read `AGENTS.md`.
2. Read:
   - `strategies/nq_frozen_v1/SPECIFICATION.md`
   - `strategies/nq_frozen_v1/REQUIREMENTS.md`
   - `strategies/nq_frozen_v1/ORACLE.md`
   - `docs/FUTURE_ENGINE_WORKFLOW.md`
   - `docs/CONTRACT_PROVENANCE.md`
3. Run:
   ```bash
   python -m pip install -e '.[test]'
   python -m pytest -q
   ```
4. Confirm all repository tests pass before changing anything.
5. Locate the user's canonical `1Min_NQ.csv` locally. Do not edit, sort, deduplicate, fill, overwrite, convert, or move it.
6. Inspect only the header/first few rows to identify its timestamp column and exact schema.
7. Hash it:
   ```bash
   python scripts/hash_data.py "/actual/path/1Min_NQ.csv" --out reports/nq_1min_manifest.json
   ```
8. Run data audit with the actual timestamp column:
   ```bash
   python scripts/audit_data.py "/actual/path/1Min_NQ.csv" --timestamp-col <actual_column>
   ```
9. If there is no contract identifier/provenance, state `ROLL PROVENANCE UNVERIFIED`; do not call it clean.

## Current frozen NQ candidate

The existing candidate is `nq_frozen/`; its independent reference is `nq_frozen/reference.py`.

Before historical performance:
1. verify deterministic pytest suite;
2. run generic candidate audit using `strategies/nq_frozen_v1/adapter.py`;
3. run production-vs-reference reconciliation;
4. inspect failures without changing frozen rules;
5. if a real bug exists, create a failing regression test first, then make the smallest patch.

Example generic command:

```bash
python scripts/audit_candidate.py   --data "/actual/path/1Min_NQ.csv"   --timestamp-col <actual_column>   --adapter strategies/nq_frozen_v1/adapter.py   --source nq_frozen   --tick-size 0.25
```

The generic causality audit intentionally reruns engines on truncated and future-mutated data, so it can be expensive on the full 16-year dataset. If resource constraints are real, first demonstrate it on several representative multi-month/year slices, then perform the strongest full-history checks the machine can support. Never silently skip the check: report what was and was not run.

## GitHub

The user has already created an empty repository. If the current local folder is not a git repo:
1. initialize git;
2. set branch `main`;
3. add the user's empty GitHub repository as `origin`;
4. commit this verification baseline;
5. push it.

Do not commit the 278 MB market-data CSV. `.gitignore` intentionally excludes market data.

If Git credentials or the exact remote URL are unavailable, do not guess. Leave the repo intact and report the single command or credential step the user must supply.

## Future engines

When another LLM brings a new engine:
- do not replace this harness;
- create `candidate_engines/<engine_name>/`;
- create/adapt a strategy directory under `strategies/<strategy_name>/`;
- preserve old frozen strategies and their tests;
- wrap the candidate with an adapter;
- run generic checks plus strategy-specific oracle tests;
- preserve every proven bug as a permanent regression test.

The verification lab grows; it is not rewritten from scratch for each engine.
