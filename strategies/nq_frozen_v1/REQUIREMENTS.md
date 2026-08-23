# Frozen NQ v1 — Numbered Requirements

> Generated directly from the frozen prose specification. The prose specification remains authoritative; these IDs exist so tests and audit findings can point to one rule.


## 1. Instrument and input bars

- **R001** — Instrument: NQ futures.
- **R002** — Direction: long only. Shorts are never entered.
- **R003** — Decision and execution data: one-minute OHLCV bars.
- **R004** — Bar timestamps denote the beginning of each one-minute interval.
- **R005** — Bars are chronological and expressed internally in UTC; session rules are converted to the named US timezone.
- **R006** — The research input is the corrected volume-selected continuous NQ contract series. Contract selection and roll construction happen upstream; the strategy itself does not choose contracts.
- **R007** — The source can omit a minute when no trade prints. A missing row is not synthesized.

## 2. Fractional-difference signal

- **R008** — Input series: natural logarithm of one-minute close.
- **R009** — Fractional-difference order: `d = 0.45`.
- **R010** — Maximum weight count: `100`.
- **R011** — Weights are recursively generated from `w[0] = 1` and `w[k] = -w[k-1] * (d-k+1) / k`.
- **R012** — Weight generation stops before the first weight whose absolute value is below `0.001`; otherwise it stops at 100 weights.
- **R013** — The retained weights are applied causally. No future close is used.
- **R014** — A rolling z-score is calculated over the most recent 100 valid fractional-difference observations, including the current signal bar.
- **R015** — Rolling standard deviation uses population convention, `ddof = 0`.
- **R016** — A raw long signal occurs only on an upward crossing of `+5.0`: current z-score is at least 5.0 and the immediately preceding stored row's z-score is below 5.0.
- **R017** — Remaining above 5.0 does not create repeated signals. A new signal requires another upward crossing.

## 3. Keltner filter

- **R018** — Keltner center: EMA of one-minute close with span 60, `adjust = false`.
- **R019** — Keltner range unit: simple rolling mean of `high-low` over 60 one-minute rows, requiring all 60 observations.
- **R020** — Keltner value: `(close - EMA60) / mean_range60`.
- **R021** — The Keltner value is evaluated on the completed signal bar, never on the entry bar.
- **R022** — A signal passes only when the Keltner value is finite and no greater than `11.462043356166953`.

## 4. Session and London exclusion

- **R023** — London classification is defined in America/Chicago time.
- **R024** — `globex_asia`: 17:00 CT through 01:59 CT.
- **R025** — `london`: 02:00 CT through 08:29 CT.
- **R026** — `ny_am`: 08:30 CT through 11:59 CT.
- **R027** — `ny_pm`: 12:00 CT through 14:59 CT.
- **R028** — `ny_late`: 15:00 CT through 15:59 CT.
- **R029** — `maintenance`: 16:00 CT through 16:59 CT.
- **R030** — Both the signal timestamp and the proposed entry timestamp must not be classified as London.
- **R031** — Both timestamps must also be inside the permitted ET trading interval: 18:00 ET through 14:59 ET.
- **R032** — Timestamps from 15:00 ET through 17:59 ET are prohibited.
- **R033** — Signal and entry must belong to the same trading session, defined by sharing the same next 15:00 ET deadline. Signals are never carried across the daily cutoff, a weekend, holiday closure, or another session.

## 5. Entry

- **R034** — Entry is considered only after the signal bar is complete.
- **R035** — Proposed entry is the open of the next stored one-minute row.
- **R036** — Because no-trade minutes can be absent, that next row may be more than one clock minute later, but it must remain eligible and within the same trading session as the signal.
- **R037** — No interpolation, synthetic price, signal-bar close fill, or intrabar signal fill is used.
- **R038** — At most one global position may be open at any time.
- **R039** — If a proposed entry occurs while the previous trade remains open, that signal is discarded permanently; it is not queued.

## 6. Four-hour ATR

- **R040** — Stop distance is based on one four-hour true range, described as `4h_atr1`.
- **R041** — Four-hour bars are aligned to fixed UTC epoch boundaries: 00:00, 04:00, 08:00, 12:00, 16:00, and 20:00 UTC.
- **R042** — Each four-hour bar aggregates all printed one-minute rows in `[boundary, boundary+4h)`.
- **R043** — A four-hour period with at least one printed source row is valid. It is not required to contain 240 rows.
- **R044** — A completely empty four-hour period supplies no OHLC or true range.
- **R045** — True range is the maximum of: `high-low`, `abs(high-previous_close)`, and `abs(low-previous_close)`.
- **R046** — ATR rolling window is one completed four-hour true range; there is no Wilder smoothing or multi-bar averaging.
- **R047** — The ATR is shifted by one completed four-hour bucket before being made available. The source bucket must end no later than entry time.
- **R048** — After an empty four-hour period, the latest causally available completed nonempty four-hour value is carried forward.
- **R049** — A trade is skipped if its available ATR is missing, nonfinite, or nonpositive.

## 7. Initial bracket geometry

- **R050** — NQ tick size: `0.25` index point.
- **R051** — Unrounded stop: `entry - ATR`.
- **R052** — Stop price is rounded upward to the next valid tick using a ceiling operation.
- **R053** — Unrounded target: `entry + 0.5 * ATR`.
- **R054** — Target price is also rounded upward to the next valid tick using a ceiling operation.
- **R055** — Initial risk in points is `entry - rounded_stop`.
- **R056** — A trade is skipped if rounded initial risk is not positive.
- **R057** — The stop and target remain fixed for the life of the trade.
- **R058** — There is no breakeven rule, trailing stop, time exit other than the session flat, partial exit, scaling, pyramiding, or re-entry attached to the same signal.

## 8. Bar-by-bar execution ordering

- **R059** — Simulation begins on the entry bar and proceeds chronologically through printed one-minute rows.
- **R060** — Opening events are resolved before intrabar high/low events.
- **R061** — If bar open is at or below the stop, exit as SL at the bar open. This permits a loss worse than `-1R` on a downward gap.
- **R062** — Otherwise, if bar open is at or above the target, exit as TP at the target price. Positive target gaps receive no price improvement.
- **R063** — Otherwise, if the bar low touches or crosses the stop, exit as SL at the stop.
- **R064** — Otherwise, if the bar high touches or crosses the target, exit as TP at the target.
- **R065** — Therefore, when both stop and target are touched within one bar without an opening gap resolving the trade, the stop wins. This is the frozen conservative same-bar ambiguity rule.
- **R066** — No high, low, or close occurring after a modeled opening exit is used.

## 9. Mandatory session flat

- **R067** — Every accepted trade receives its own hard deadline of 15:00 ET.
- **R068** — An entry from 18:00 ET onward uses the following calendar day's 15:00 ET deadline. An entry before 15:00 ET uses that calendar day's deadline.
- **R069** — If an exact 15:00 ET row exists, an open trade is flattened at that row's open before its high, low, or close is examined.
- **R070** — If no exact 15:00 ET row exists, the trade can use the complete range of the final printed bar before 15:00 and, if still open, is flattened at that bar's close.
- **R071** — A trade may never consume a row timestamped after its own deadline.
- **R072** — A trade may never roll forward to another day's 15:00 row because its own boundary row is absent.
- **R073** — This 15:00 ET rule is stricter than a prop-firm 16:59 ET mandatory-flat deadline.

## 10. R accounting and transaction costs

- **R074** — Gross R uses the rounded initial risk in points as denominator.
- **R075** — TP accounting is frozen at nominal `+0.5R`, even when upward tick rounding puts the actual target slightly farther than 0.5R.
- **R076** — Normal stop accounting is `-1R`.
- **R077** — A stop opening gap uses `(actual exit price-entry) / initial risk` and can be below `-1R`.
- **R078** — A forced flat uses `(flat price-entry) / initial risk` and can be positive, negative, or zero.
- **R079** — Round-trip transaction cost is zero for entries dated before 2023.
- **R080** — Round-trip transaction cost is 1.0 NQ index point for entries dated 2023 or later.
- **R081** — Cost in R is `cost_points / initial risk`.
- **R082** — Net R is `gross_R - cost_R`.
- **R083** — Profit factor in R is the sum of positive net R divided by the absolute sum of negative net R.
- **R084** — Win rate is the percentage of trades with net R greater than zero. A positive forced flat counts as a win; a negative or zero flat does not.

## 11. Sample labels

- **R085** — IS years: 2010, 2013, 2017, 2020, 2022, 2024, 2026.
- **R086** — OOS years: every other included year: 2011, 2012, 2014, 2015, 2016, 2018, 2019, 2021, 2023, 2025.
- **R087** — Sample labels affect reporting only. They do not alter signals, filters, entries, stops, targets, costs, or exits.

## 12. Explicitly absent features

- **R088** — No shorts.
- **R089** — No Kaufman efficiency filter or entry confirmation.
- **R090** — No volume, relative-volume, volatility, return-state, Kalman, HMM, Hurst, rough-volatility, Donchian, or additional Keltner bucket filter.
- **R091** — No London trades.
- **R092** — No breakeven-at-90-percent rule.
- **R093** — No 120-minute holding-period exit.
- **R094** — No 16:00 ET flat and no 16:59 ET flat; the frozen strategy uses 15:00 ET.
- **R095** — No discretionary trade deletion and no special handling by year.
- **R096** — No daily one-trade limit. Multiple sequential trades may occur on the same ET calendar date, but never concurrently.

## 13. Data assumptions that are not strategy parameters

- **R097** — Accuracy depends on the upstream volume-selected continuous-contract construction being correct and free of duplicate or misassigned contract rows.
- **R098** — One-minute OHLC cannot reveal the true order of an intrabar high and low. The stop-first convention is an explicit conservative modeling rule, not observed tick-level ordering.
- **R099** — When an exact deadline row is absent, the last printed predeadline close is the last observable bar price known before the deadline; it is not a quote-level guaranteed fill.
- **R100** — The backtest uses trade-bar OHLCV, not bid/ask quotes or an order-book queue model.
