# Current Verification-Lab Status

Updated 2026-08-23.

- Frozen NQ prose specification, 100 numbered requirements, and 82 human-readable oracle cases remain preserved.
- Production and independent reference implementations remain a candidate/reference pair; neither frozen core was changed during the infrastructure pass.
- Full repository suite: **91 passed**.
- Dedicated causality/mutation regression pack: **20 passed**.
- Strong causality coverage for the frozen adapter on deterministic synthetic bars: 28 cutoffs across features, signals, eligible signals, proposed entries, accepted entries, and trades; 168 truncation comparisons plus 168 future-mutation comparisons; 0 failures.
- Machine verdicts are `VERIFIED` (exit 0), `FAILED` (exit 1), and `INCOMPLETE / UNVERIFIED` (exit 2). Missing mandatory evidence cannot produce `VERIFIED`; a mandatory failure takes precedence over missing evidence.
- The one-command runner is `python scripts/verify.py ...` and emits JSON plus Markdown reports.
- Auditor self-tests include deliberate lookahead, normalization, overlap, timestamp, fill-priority, gap, deadline, ATR, tick, cost/R, ordering, IS/OOS influence, and queued-signal mutations.
- Canonical one-minute dataset: not located; identity and historical verification remain **UNVERIFIED**.
- Supplied high-resolution dataset `C:\Users\mikek\Downloads\NQ.parquet`:
  - SHA-256 `f76cb2c5a1c9e4034fc1c8a1f23a490cc8a41c244d298bc2483a091fa8b924b0`
  - 2,099,660,927 bytes; 142,610,981 rows
  - UTC range 2010-07-07T00:00:00Z through 2026-08-06T23:59:58Z
  - chronological; 0 duplicate timestamp/instrument rows; 0 malformed rows; 0 nonfinite cells
  - immutable audit manifest: `reports/nq_1s_manifest.json`
- Raw contract IDs/symbols exist in the high-resolution dataset, but no frozen continuous-contract selector or exact selection map was supplied. **ROLL PROVENANCE UNVERIFIED**.
- One-second execution cross-check remains **UNVERIFIED** until canonical one-minute trades can be tied to exact raw contracts through verified roll provenance.
- Historical PF and win rate have not been inspected.

This status is not a claim that historical strategy performance or continuous-contract construction is valid. The frozen strategy verdict remains `INCOMPLETE / UNVERIFIED` until all mandatory real-data and provenance gates pass.
