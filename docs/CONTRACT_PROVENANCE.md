# Contract and Rollover Verification

## What can be proven from a continuous OHLCV file?

Very little about the roll itself. Price/volume bars do not reveal which raw contract was selected unless a contract identifier is retained.

Therefore:

- continuous OHLCV without `contract_id` -> **UNVERIFIED**
- continuous OHLCV with `contract_id` -> basic switch/provenance diagnostics only
- continuous OHLCV + exact expected selection map -> selected contract can be checked exactly
- raw per-contract bars + a precisely specified roll/selection algorithm -> a separate oracle can reconstruct expected selections and compare them

## Recommended upstream artifacts

Keep these beside the canonical dataset, not inside the strategy engine:

1. `data_manifest.json`
   - SHA256 of the continuous file
   - byte size
   - provenance/source
   - creation date
2. `selection_map.csv`
   - `timestamp`
   - `expected_contract_id`
   - optional selection reason / volume comparison
3. a frozen upstream roll specification
   - exact candidate contracts
   - volume measurement window
   - tie behavior
   - roll timing
   - holidays/session treatment
   - whether switching back is permitted

Then verify:

```bash
python scripts/hash_data.py /path/to/1Min_NQ.csv --out data_manifest.json
python scripts/audit_roll.py --selected selected_with_contract_id.csv --expected selection_map.csv
```

Never let a strategy engine choose contracts if the research specification says contract construction is upstream.

Raw one-second files containing `instrument_id` and symbols are useful provenance evidence, but they
still do not prove which contract an upstream continuous series selected. Numeric IDs can be time-aware,
and symbols may include calendar spreads. Negative spread prices must not be auto-repaired.

The generic verifier supports an exact expected selection map, switch-policy callback, expiry metadata,
or an explicitly supplied independent selector over raw per-contract bars. It never invents the selector.
Without that policy/evidence, the result remains `ROLL PROVENANCE UNVERIFIED`.
