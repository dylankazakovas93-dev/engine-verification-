# Workflow for Every Future Engine

1. Create/freeze strategy spec under `strategies/<name>/SPECIFICATION.md`.
2. Convert material rules into numbered `REQUIREMENTS.md`.
3. Write human-verifiable `ORACLE.md` cases before trusting implementation.
4. Put candidate engine under `candidate_engines/<name>/`.
5. Write only an adapter to canonicalize the candidate's trade ledger.
6. Run repository tests.
7. Hash and audit the exact data file.
8. Run generic candidate audit:
   - schema
   - ledger invariants
   - global overlap
   - causality/truncation
   - future mutation
   - source risk scan
9. Run strategy-specific deterministic tests.
10. If available, run independent reference reconciliation.
11. Audit contract provenance separately.
12. Only then inspect PF/WR/equity curve.
13. Any newly found bug becomes a permanent failing regression before patching.
14. Run `python scripts/verify.py ...` and retain its JSON/Markdown artifacts.
15. Inspect PF/WR only if every pre-performance mandatory gate is `PASS` and the machine verdict is `VERIFIED`.

## Adapter stage interface

`run(bars)` is mandatory. Expose `features`, `signals`, `eligible_signals`,
`proposed_entries`, `accepted_entries`, and/or `audit_stages` when the candidate makes them
available. Do not fabricate intermediate stages. Missing mandatory evidence is `UNVERIFIED`.

## Causality modes

- `fast`: bounded smoke coverage.
- `standard`: distributed/warmup/session/event/random coverage.
- `strong` (default): 28 deterministic cutoffs at every exposed stage.

Every report records the exact cutoff timestamps, stages, and comparison counts.

## Verdict language

Use only:
- PASS: the check actually demonstrated the property
- FAIL: a reproducible violation was found
- WARN: heuristic/suspicious pattern requiring review
- UNVERIFIED: required evidence does not exist

Never upgrade WARN/UNVERIFIED to PASS because results look plausible.

Top-level exit codes: `0=VERIFIED`, `1=FAILED`, `2=INCOMPLETE / UNVERIFIED`.
