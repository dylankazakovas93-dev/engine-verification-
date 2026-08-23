# Frozen NQ Strategy Specification

This document defines the strategy independently of any Python, Pine, or execution implementation. Changing any item below creates a different strategy or test.

## 1. Instrument and input bars

- Instrument: NQ futures.
- Direction: long only. Shorts are never entered.
- Decision and execution data: one-minute OHLCV bars.
- Bar timestamps denote the beginning of each one-minute interval.
- Bars are chronological and expressed internally in UTC; session rules are converted to the named US timezone.
- The research input is the corrected volume-selected continuous NQ contract series. Contract selection and roll construction happen upstream; the strategy itself does not choose contracts.
- The source can omit a minute when no trade prints. A missing row is not synthesized.

## 2. Fractional-difference signal

- Input series: natural logarithm of one-minute close.
- Fractional-difference order: `d = 0.45`.
- Maximum weight count: `100`.
- Weights are recursively generated from `w[0] = 1` and `w[k] = -w[k-1] * (d-k+1) / k`.
- Weight generation stops before the first weight whose absolute value is below `0.001`; otherwise it stops at 100 weights.
- The retained weights are applied causally. No future close is used.
- A rolling z-score is calculated over the most recent 100 valid fractional-difference observations, including the current signal bar.
- Rolling standard deviation uses population convention, `ddof = 0`.
- A raw long signal occurs only on an upward crossing of `+5.0`: current z-score is at least 5.0 and the immediately preceding stored row's z-score is below 5.0.
- Remaining above 5.0 does not create repeated signals. A new signal requires another upward crossing.

## 3. Keltner filter

- Keltner center: EMA of one-minute close with span 60, `adjust = false`.
- Keltner range unit: simple rolling mean of `high-low` over 60 one-minute rows, requiring all 60 observations.
- Keltner value: `(close - EMA60) / mean_range60`.
- The Keltner value is evaluated on the completed signal bar, never on the entry bar.
- A signal passes only when the Keltner value is finite and no greater than `11.462043356166953`.

## 4. Session and London exclusion

- London classification is defined in America/Chicago time.
- `globex_asia`: 17:00 CT through 01:59 CT.
- `london`: 02:00 CT through 08:29 CT.
- `ny_am`: 08:30 CT through 11:59 CT.
- `ny_pm`: 12:00 CT through 14:59 CT.
- `ny_late`: 15:00 CT through 15:59 CT.
- `maintenance`: 16:00 CT through 16:59 CT.
- Both the signal timestamp and the proposed entry timestamp must not be classified as London.
- Both timestamps must also be inside the permitted ET trading interval: 18:00 ET through 14:59 ET.
- Timestamps from 15:00 ET through 17:59 ET are prohibited.
- Signal and entry must belong to the same trading session, defined by sharing the same next 15:00 ET deadline. Signals are never carried across the daily cutoff, a weekend, holiday closure, or another session.

## 5. Entry

- Entry is considered only after the signal bar is complete.
- Proposed entry is the open of the next stored one-minute row.
- Because no-trade minutes can be absent, that next row may be more than one clock minute later, but it must remain eligible and within the same trading session as the signal.
- No interpolation, synthetic price, signal-bar close fill, or intrabar signal fill is used.
- At most one global position may be open at any time.
- If a proposed entry occurs while the previous trade remains open, that signal is discarded permanently; it is not queued.

## 6. Four-hour ATR

- Stop distance is based on one four-hour true range, described as `4h_atr1`.
- Four-hour bars are aligned to fixed UTC epoch boundaries: 00:00, 04:00, 08:00, 12:00, 16:00, and 20:00 UTC.
- Each four-hour bar aggregates all printed one-minute rows in `[boundary, boundary+4h)`.
- A four-hour period with at least one printed source row is valid. It is not required to contain 240 rows.
- A completely empty four-hour period supplies no OHLC or true range.
- True range is the maximum of: `high-low`, `abs(high-previous_close)`, and `abs(low-previous_close)`.
- ATR rolling window is one completed four-hour true range; there is no Wilder smoothing or multi-bar averaging.
- The ATR is shifted by one completed four-hour bucket before being made available. The source bucket must end no later than entry time.
- After an empty four-hour period, the latest causally available completed nonempty four-hour value is carried forward.
- A trade is skipped if its available ATR is missing, nonfinite, or nonpositive.

## 7. Initial bracket geometry

- NQ tick size: `0.25` index point.
- Unrounded stop: `entry - ATR`.
- Stop price is rounded upward to the next valid tick using a ceiling operation.
- Unrounded target: `entry + 0.5 * ATR`.
- Target price is also rounded upward to the next valid tick using a ceiling operation.
- Initial risk in points is `entry - rounded_stop`.
- A trade is skipped if rounded initial risk is not positive.
- The stop and target remain fixed for the life of the trade.
- There is no breakeven rule, trailing stop, time exit other than the session flat, partial exit, scaling, pyramiding, or re-entry attached to the same signal.

## 8. Bar-by-bar execution ordering

- Simulation begins on the entry bar and proceeds chronologically through printed one-minute rows.
- Opening events are resolved before intrabar high/low events.
- If bar open is at or below the stop, exit as SL at the bar open. This permits a loss worse than `-1R` on a downward gap.
- Otherwise, if bar open is at or above the target, exit as TP at the target price. Positive target gaps receive no price improvement.
- Otherwise, if the bar low touches or crosses the stop, exit as SL at the stop.
- Otherwise, if the bar high touches or crosses the target, exit as TP at the target.
- Therefore, when both stop and target are touched within one bar without an opening gap resolving the trade, the stop wins. This is the frozen conservative same-bar ambiguity rule.
- No high, low, or close occurring after a modeled opening exit is used.

## 9. Mandatory session flat

- Every accepted trade receives its own hard deadline of 15:00 ET.
- An entry from 18:00 ET onward uses the following calendar day's 15:00 ET deadline. An entry before 15:00 ET uses that calendar day's deadline.
- If an exact 15:00 ET row exists, an open trade is flattened at that row's open before its high, low, or close is examined.
- If no exact 15:00 ET row exists, the trade can use the complete range of the final printed bar before 15:00 and, if still open, is flattened at that bar's close.
- A trade may never consume a row timestamped after its own deadline.
- A trade may never roll forward to another day's 15:00 row because its own boundary row is absent.
- This 15:00 ET rule is stricter than a prop-firm 16:59 ET mandatory-flat deadline.

## 10. R accounting and transaction costs

- Gross R uses the rounded initial risk in points as denominator.
- TP accounting is frozen at nominal `+0.5R`, even when upward tick rounding puts the actual target slightly farther than 0.5R.
- Normal stop accounting is `-1R`.
- A stop opening gap uses `(actual exit price-entry) / initial risk` and can be below `-1R`.
- A forced flat uses `(flat price-entry) / initial risk` and can be positive, negative, or zero.
- Round-trip transaction cost is zero for entries dated before 2023.
- Round-trip transaction cost is 1.0 NQ index point for entries dated 2023 or later.
- Cost in R is `cost_points / initial risk`.
- Net R is `gross_R - cost_R`.
- Profit factor in R is the sum of positive net R divided by the absolute sum of negative net R.
- Win rate is the percentage of trades with net R greater than zero. A positive forced flat counts as a win; a negative or zero flat does not.

## 11. Sample labels

- IS years: 2010, 2013, 2017, 2020, 2022, 2024, 2026.
- OOS years: every other included year: 2011, 2012, 2014, 2015, 2016, 2018, 2019, 2021, 2023, 2025.
- Sample labels affect reporting only. They do not alter signals, filters, entries, stops, targets, costs, or exits.

## 12. Explicitly absent features

- No shorts.
- No Kaufman efficiency filter or entry confirmation.
- No volume, relative-volume, volatility, return-state, Kalman, HMM, Hurst, rough-volatility, Donchian, or additional Keltner bucket filter.
- No London trades.
- No breakeven-at-90-percent rule.
- No 120-minute holding-period exit.
- No 16:00 ET flat and no 16:59 ET flat; the frozen strategy uses 15:00 ET.
- No discretionary trade deletion and no special handling by year.
- No daily one-trade limit. Multiple sequential trades may occur on the same ET calendar date, but never concurrently.

## 13. Data assumptions that are not strategy parameters

- Accuracy depends on the upstream volume-selected continuous-contract construction being correct and free of duplicate or misassigned contract rows.
- One-minute OHLC cannot reveal the true order of an intrabar high and low. The stop-first convention is an explicit conservative modeling rule, not observed tick-level ordering.
- When an exact deadline row is absent, the last printed predeadline close is the last observable bar price known before the deadline; it is not a quote-level guaranteed fill.
- The backtest uses trade-bar OHLCV, not bid/ask quotes or an order-book queue model.
