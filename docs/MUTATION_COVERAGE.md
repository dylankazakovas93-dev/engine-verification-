# Verifier Mutation Coverage

The verifier is tested against deliberately broken toy candidates. A mutation test passes only
when the intended verifier family rejects or flags the defect; this is not a claim that every
conceivable bug can be mechanically proven.

| Deliberate mutation | Expected detector |
|---|---|
| one-bar lookahead | stage-level truncation/future-mutation causality |
| `.shift(-1)` leakage | AST static risk scan and causality when behavior is exposed |
| future/full-sample normalization | feature-level causality |
| overlapping positions | ledger `one_global_position` |
| same-timestamp reentry under strict policy | ledger `one_global_position` |
| signal-bar entry | ledger `entry_after_signal` |
| wrong same-bar TP/SL ordering | trade-by-trade reference reconciliation |
| target-gap price improvement | trade-by-trade reference reconciliation |
| stop gap capped at `-1R` | trade-by-trade reference reconciliation |
| deadline overrun | ledger deadline invariant |
| unfinished ATR/source | ledger ATR causality invariant |
| incorrect tick rounding | ledger tick alignment |
| incorrect transaction cost/R arithmetic | ledger arithmetic and reference reconciliation |
| backfilled market data | AST static risk scan |
| silently sorted input | AST static risk scan plus frozen input tests |
| silently deduplicated input | AST static risk scan plus frozen input tests |
| IS/OOS labels affecting decisions | reference trade-count reconciliation |
| queued signal during an open position | overlap and reference trade-count reconciliation |
| wrong direction in a long-only profile | ledger direction invariant |

Static matches remain heuristic unless classification is `PROVEN FAILURE`. A passing static scan
does not prove that a candidate has no leakage.
