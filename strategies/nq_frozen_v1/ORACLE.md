# Frozen NQ v1 — Human-Verifiable Oracle

This file is the human-readable expected-behavior oracle. It is intentionally written without reference to implementation code. If an engine disagrees with one of these cases, the engine is wrong unless the frozen specification itself is explicitly versioned and changed.

## A. Input/data behavior

| Case | Setup | Expected |
|---|---|---|
| O001 Missing minute | Stored rows at 10:00 and 10:02; no 10:01 | Keep exactly two rows. Do not synthesize 10:01. |
| O002 Duplicate timestamp | Two stored rows have the same UTC timestamp | Reject input; do not silently deduplicate. |
| O003 Unsorted rows | 10:02 appears before 10:01 | Reject input; do not silently sort. |
| O004 Naive timestamp | Timestamp has no timezone | Reject input. |
| O005 Invalid OHLC | `high < low`, or open/close outside `[low, high]` | Reject input. |

## B. Fractional difference and z-score

| Case | Setup | Expected |
|---|---|---|
| O006 First weight | `w[0] = 1` | Exactly 1.0. |
| O007 Recursive next weight | `d=.45`; `w[k] = -w[k-1]*(d-k+1)/k` | Use the recurrence exactly. |
| O008 Weight cutoff | First candidate weight has `abs(weight) < .001` | Stop **before** retaining that weight. |
| O009 Weight cap | No cutoff before 100 weights | Retain no more than 100. |
| O010 Orientation | Three stored log closes and three weights | `w[0]` multiplies current row, `w[1]` previous stored row, etc. |
| O011 Z window | Fewer than 100 valid fracdiff observations | z-score unavailable. |
| O012 Population std | Exactly 100 valid fracdiff values | Use `ddof=0`, including the current signal row. |
| O013 Crossing | Previous stored z `4.99`, current z `5.00` | Raw long signal = true. |
| O014 Persistent high z | Previous z `5.10`, current z `5.30` | No new signal. |
| O015 Exact previous threshold | Previous z `5.00`, current z `5.20` | No new signal because previous must be **below** 5.0. |

## C. Keltner

| Case | Setup | Expected |
|---|---|---|
| O016 EMA recursion | span=60, `adjust=false` | Match causal recursive EMA; no future row. |
| O017 Mean range warmup | 59 stored rows | Keltner unavailable. |
| O018 Mean range ready | 60 stored rows | Use simple mean of `high-low` over those 60 stored rows. |
| O019 Threshold equality | Keltner exactly `11.462043356166953` | Signal passes. |
| O020 Above threshold | Keltner slightly greater than threshold | Signal fails. |
| O021 Entry-bar movement | Signal bar passes Keltner; next-row entry bar moves wildly | Entry-bar movement must not alter signal-bar Keltner decision. |

## D. Session / time boundaries

| Case | Timestamp classification | Expected |
|---|---|---|
| O022 01:59 CT | `globex_asia` | Not London. |
| O023 02:00 CT | `london` | Prohibited. |
| O024 08:29 CT | `london` | Prohibited. |
| O025 08:30 CT | `ny_am` | Not London. |
| O026 14:59 ET | permitted interval | Eligible if other rules pass. |
| O027 15:00 ET | prohibited interval | No signal/entry. |
| O028 17:59 ET | prohibited interval | No signal/entry. |
| O029 18:00 ET | permitted interval | Eligible if other rules pass. |
| O030 Same deadline | signal and proposed entry share same next 15:00 ET deadline | Session test passes. |
| O031 Different deadline | signal and proposed entry map to different 15:00 ET deadlines | Discard signal. |
| O032 DST | Session spans DST transition | Deadline is derived from local 15:00 ET calendar time, not by adding a fixed 24 hours. |

## E. Entry / global position

| Case | Setup | Expected |
|---|---|---|
| O033 Missing clock minute | Signal row 10:00, next stored row 10:03 | Proposed entry is 10:03 open. |
| O034 Cutoff crossing | Signal before cutoff, next stored row belongs to next session | Discard; do not carry signal. |
| O035 Existing position | Candidate entry occurs while prior position remains open | Discard candidate permanently; never queue. |
| O036 Same-row prior exit | Prior trade is still open immediately before candidate entry row but exits on that row open | Current frozen implementation note: reject new candidate. |

## F. 4h ATR

| Case | Setup | Expected |
|---|---|---|
| O037 Fixed buckets | Rows at 08:01 and 11:59 UTC | Both aggregate into `[08:00,12:00)`. |
| O038 Partial bucket | Only one printed minute inside a 4h bucket | Bucket is valid. |
| O039 Empty bucket | No printed rows in an entire 4h interval | No OHLC/TR is created for it. |
| O040 Before completion | Entry at 11:59 UTC | `[08:00,12:00)` TR is unavailable. |
| O041 Exact completion | Entry at 12:00 UTC | `[08:00,12:00)` TR may be used. |
| O042 Empty carry | Latest completed bucket is empty | Carry latest causally available nonempty TR. |
| O043 TR after gap | New nonempty bucket follows an empty bucket | Previous close is the prior **nonempty** 4h close. |
| O044 Invalid ATR | ATR missing, nonfinite, zero or negative | Skip trade. |

## G. Bracket geometry

| Case | Setup | Expected |
|---|---|---|
| O045 Exact tick | Price already on .25 tick | Ceiling leaves it unchanged. |
| O046 Fractional stop | Unrounded stop `19950.01` | Ceiling to `19950.25`. |
| O047 Fractional target | Unrounded target `20025.01` | Ceiling to `20025.25`. |
| O048 Zero rounded risk | Rounded stop equals entry | Skip trade. |
| O049 Fixed bracket | Trade opens successfully | Stop/target never trail or move. |

## H. Execution ordering

| Case | Setup | Expected |
|---|---|---|
| O050 Opening stop gap | entry=20000, stop=19950; bar O=19940 H=20050 L=19930 | Exit SL at **19940 open**. Ignore later H/L/C. |
| O051 Opening target gap | target=20025; bar O=20040 | Exit TP at **20025 target**, no improvement. |
| O052 Intrabar stop only | Open between levels, low touches stop | Exit SL at stop. |
| O053 Intrabar target only | Open between levels, low stays above stop, high touches target | Exit TP at target. |
| O054 Same-bar both | Open between levels, low touches stop and high touches target | Stop wins. |
| O055 Exact touch stop | low equals stop | Stop is hit. |
| O056 Exact touch target | high equals target and stop not touched | Target is hit. |

## I. Mandatory 15:00 ET flat

| Case | Setup | Expected |
|---|---|---|
| O057 Exact deadline row | 15:00 ET row O=102 H=200 L=50 C=150 | Flatten at **102 open**, ignore H/L/C. |
| O058 No deadline row | Last predeadline row is 14:59; next printed row 15:02 | Use full 14:59 range; if still open, flatten at 14:59 close. Never inspect 15:02. |
| O059 Last predeadline target | No 15:00 row; 14:59 high hits target | TP occurs before forced close. |
| O060 Own deadline only | No exact deadline row exists | Never roll forward to another day's 15:00 row. |

## J. R and costs

| Case | Setup | Expected |
|---|---|---|
| O061 Normal TP | target hit normally | Gross R = exactly `+0.5`, irrespective of tick-rounding extra distance. |
| O062 Normal SL | stop hit at stop | Gross R = exactly `-1.0`. |
| O063 Stop opening gap | entry=100, risk=10, exit open=87 | Gross R = `(87-100)/10 = -1.3`. |
| O064 Forced flat | entry=100, risk=10, flat=103 | Gross R = `+0.3`. |
| O065 Pre-2023 cost | ET entry year 2022 | cost points = 0. |
| O066 2023+ cost | ET entry year 2023 | cost points = 1.0 NQ point. |
| O067 Cost R | initial risk 4 points, cost 1 point | cost R = `0.25`. |
| O068 Net R | gross R .5, cost R .25 | net R = `.25`. |
| O069 Win definition | net R > 0 | win. |
| O070 Zero net | net R == 0 | not a win. |

## K. Global invariants / leakage

| Case | Deliberate defect | Expected verifier result |
|---|---|---|
| O071 Overlap | Trade B entry occurs before Trade A exit | `one_global_position = FAIL`. |
| O072 Same-open overlap | Strict profile; B entry timestamp equals A exit timestamp | FAIL under current frozen note. |
| O073 Future append | Append future rows after cutoff | All already-completed pre-cutoff decisions/trades remain identical. |
| O074 Future mutation | Change future OHLC massively after cutoff | Completed pre-cutoff ledger remains identical. |
| O075 Negative shift | Candidate source contains `.shift(-1)` | Static risk scan warns/highlights it; not automatically declared a proven leak without context. |
| O076 Backfill | Candidate source uses `.bfill()` | Static risk scan warns/highlights it. |
| O077 Centered rolling | Candidate uses `rolling(..., center=True)` | Static risk scan warns/highlights it. |
| O078 Forward as-of join | Candidate uses `merge_asof(... direction='forward')` | Static risk scan warns/highlights it. |

## L. Contract/roll provenance

| Case | Available evidence | Expected |
|---|---|---|
| O079 Continuous OHLCV only | No `contract_id`, no selection ledger | **UNVERIFIED**, never PASS. |
| O080 Continuous rows include contract_id | IDs present and non-null | Basic provenance/switch diagnostics can run, but roll-policy correctness is still not proven. |
| O081 Expected selection map supplied | timestamp → expected contract ID exists | Exact selected-vs-expected contract comparison can PASS/FAIL. |
| O082 Multiple selected IDs at same timestamp | Continuous file has >1 chosen contract ID for a timestamp | FAIL. |

## Oracle change rule

If a future audit discovers an uncovered, specification-proven defect:
1. add a new oracle case here;
2. add a regression test that fails on the buggy engine;
3. patch only after the failure is reproduced;
4. rerun the entire old suite;
5. never alter an old oracle answer solely to make a candidate pass.
