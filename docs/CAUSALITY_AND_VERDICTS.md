# Causality Coverage and Verdict Gate

For cutoff `T`, the verifier compares full-history output with both a truncated input containing rows
strictly before `T` and a full input whose rows at/after `T` have drastically mutated but structurally
valid OHLC. Only decisions whose canonical information timestamp is before `T` are reconciled.

Cutoffs combine distributed positions, indicator warmups, UTC/session transitions, real stage events,
and fixed-seed random positions. Every exposed stage is compared independently. This can prove a
specific observed future dependency; it cannot prove the absence of every possible leakage mechanism.

Feature-level checks require an adapter feature stage. If absent and mandatory, the verdict is
`INCOMPLETE / UNVERIFIED`, with `FEATURE-LEVEL CAUSALITY UNVERIFIED` in the findings.

Profiles define mandatory check families. The verdict gate is:

- `VERIFIED`: every mandatory family supplied evidence and no mandatory failure/warning/unverified item remains.
- `FAILED`: at least one reproducible check fails.
- `INCOMPLETE / UNVERIFIED`: no failure is proven, but mandatory evidence is absent or awaiting manual review.

A passing static scan is heuristic only. Continuous OHLC without contract-selection provenance cannot
prove rollover. Historical performance never upgrades a verdict.
