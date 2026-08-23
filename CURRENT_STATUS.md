# Current Verification-Lab Status

- Frozen NQ prose specification preserved.
- 100 numbered requirements generated from the prose specification.
- 82 human-readable oracle cases frozen.
- Production and independent reference implementations retained as a candidate pair.
- Generic verifier added for:
  - OHLCV/data integrity
  - global position overlap
  - ledger timestamp/price invariants
  - ATR/source causality
  - R arithmetic
  - deadline violations
  - tick alignment
  - truncation invariance
  - future-data mutation
  - static source risk patterns
  - contract-provenance status
  - exact selected-vs-expected contract map comparison
- Auditor self-tests include deliberately broken examples.
- Repository test count at packaging: 52 passing.
- Smoke test of `audit_data.py` and `audit_candidate.py`: passed on synthetic timezone-aware OHLCV.
- Contract rollover remains correctly classified as UNVERIFIED when contract IDs/selection evidence are absent.

This status is not a claim that any historical performance is valid. Real-data verification still has to run locally against the canonical file.
