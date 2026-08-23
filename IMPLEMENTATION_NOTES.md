# Frozen implementation notes

These notes make edge-case interpretations explicit. They are not intended to change the strategy specification.

1. **Fractional-difference orientation**: `w[0]` multiplies the current stored close, `w[1]` the immediately preceding stored close, and so on.
2. **Stored rows, not synthetic minutes**: fractional difference, z-score, EMA, and rolling mean-range operate on stored rows/valid observations. Missing clock minutes are never synthesized.
3. **Four-hour previous close**: when an entire 4h bucket is empty, it has no OHLC. The next nonempty 4h bar uses the close of the previous nonempty 4h bar as `previous_close` for true-range calculation.
4. **Four-hour availability**: a nonempty bucket `[B, B+4h)` becomes causally available at exactly `B+4h`. An entry at that exact timestamp may use it. Empty buckets do not erase the last available completed nonempty TR.
5. **Concurrent entry at an existing trade's exit bar**: position state is checked immediately before processing the proposed entry row. If the previous trade is still open then, the new signal is discarded permanently, even if the prior trade exits at that same row's open.
6. **Year attribution**: transaction-cost year and IS/OOS reporting year use the entry timestamp converted to `America/New_York`.
7. **First nonempty 4h bar**: with no previous nonempty 4h close, true range is `high-low`.
8. **Profit-factor degenerate cases**: no trades -> NaN; positive net R and no negative net R -> +inf; no positive net R and at least one negative net R -> 0.0; all-zero net R -> NaN.

If any of these are not intended, change the specification and tests together before trusting historical results.
