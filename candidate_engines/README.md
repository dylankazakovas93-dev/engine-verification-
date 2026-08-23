# Candidate engines

Place untrusted future engine code here or elsewhere in the repo.

Do not rewrite an engine merely to make it fit the harness. Write an adapter next to it that exposes:

```python
def run(bars):
    # invoke candidate engine
    # map its output to canonical ledger columns
    return trades
```

The candidate is not "clean" because its own tests pass. It must survive the independent repository checks.
