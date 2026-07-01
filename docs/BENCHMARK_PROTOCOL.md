# Stage 2 Benchmark Protocol

## Stop condition

Stage 2 is complete only when all conditions pass:

1. Stage 1 still passes:

```bash
python scripts/validation_score.py --fast
```

2. Stage 2 fast benchmark score passes:

```bash
python scripts/benchmark_score.py --fast
```

3. `docs/BENCHMARK_REPORT.md` exists and contains a benchmark table with:

- N
- chi
- seed
- exact energy when available
- variational final energy
- energy per bond
- absolute error when exact is available
- steps or runtime
- pass/fail
- known limitations

## Required implementation stages

### Stage 2.1: Dense observable references

Create `latticetn/observables.py` if it does not exist.

Implement dense-state reference functions:

- `dense_expect_local`
- `dense_expect_two_site`
- `dense_bond_energy_heisenberg`
- `dense_entanglement_entropy`

These should work for `N <= 8` on CPU.

### Stage 2.2: MPS observable API

Implement MPS observable functions:

- `mps_expect_local`
- `mps_expect_two_site`
- `mps_bond_energy_heisenberg`
- `mps_entanglement_entropy`

Stage 2 may use `mps.to_dense()` internally. Do not claim these are efficient large-N contractions unless implemented as such.

### Stage 2.3: Observable dense comparison tests

Add tests comparing MPS observable results to dense-state results for small random MPS.

Required tests:

```bash
pytest -q tests/test_observables_dense_compare.py
pytest -q tests/test_entanglement_entropy.py
```

### Stage 2.4: Benchmark runner

Implement or update:

```text
scripts/run_heisenberg_benchmark.py
```

It should support:

- `--preset tiny`
- `--preset fast`
- `--preset full`
- `--json-output PATH`
- `--markdown-output PATH`
- CPU-only default
- fixed seeds
- explicit N, chi, steps, lr in output

### Stage 2.5: Benchmark scoring

Implement or update:

```text
scripts/benchmark_score.py
```

It must:

1. Run Stage 1 fast validation.
2. Run Stage 2 tests.
3. Run a tiny benchmark smoke test.
4. Check that `docs/BENCHMARK_REPORT.md` can be generated or already exists.

### Stage 2.6: Report

Generate:

```text
docs/BENCHMARK_REPORT.md
```

The report must clearly distinguish:

- exact diagonalization checks for small N
- variational benchmark values
- trend comparison against `1/4 - ln(2)`
- limitations of the current implementation

## Pause conditions

Pause and report instead of forcing tests to pass if:

- variational energy falls below exact ground energy beyond tolerance
- physics convention is ambiguous
- tests require loosening tolerances without a numerical reason
- runtime becomes too high for CPU fast validation
- observable values disagree with dense references after two focused attempts
