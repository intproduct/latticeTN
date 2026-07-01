# Stage 3A Canonicalization Protocol

## Stop condition

Stage 3A is complete only when ALL hold:

1. `python scripts/validation_score.py --fast` exits 0.
2. `python scripts/benchmark_score.py --fast` exits 0.
3. `python scripts/canonical_score.py --fast` exits 0.
4. `docs/CANONICALIZATION_REPORT.md` contains:
   - left/right/mixed canonical test results,
   - compression test results,
   - entropy comparison (canonical vs dense SVD),
   - energy/fidelity errors,
   - pass/fail status,
   - known limitations.
5. `docs/CLAUDE_PROGRESS_CANONICAL.md` records changes, commands, results,
   and next-step suggestions.

GPU smoke is NOT a blocking item for this stage, but existing GPU-readiness
files must remain intact and the default test paths must not depend on GPU.

## Files

New:

- `latticetn/canonical.py` — canonicalization + compression core.
- `tests/test_mps_canonicalization.py`
- `tests/test_mps_compression.py`
- `tests/test_canonical_entanglement.py`
- `tests/test_canonical_score.py`
- `scripts/canonical_score.py`
- `scripts/run_canonical_smoke.py`
- `docs/CANONICALIZATION_SPEC.md`
- `docs/CANONICALIZATION_PROTOCOL.md`
- `docs/CANONICALIZATION_REPORT.md` (generated)
- `docs/CLAUDE_PROGRESS_CANONICAL.md`

Modified (non-breaking):

- `latticetn/mps.py` — add `MPS.from_tensors` classmethod (wraps canonical-form
  tensors; `requires_grad=False` by default).

## Required implementation stages

### Stage 3A.1: core module

Implement `latticetn/canonical.py` with left/right/mixed canonical sweeps,
norm checks, SVD compression with truncation-error reporting, canonical
entanglement entropy, and `from_dense`.

### Stage 3A.2: tests

- `test_mps_canonicalization.py`: left/right/mixed orthonormality + fidelity + norm.
- `test_mps_compression.py`: bond-dim cap, no-truncation recovery, energy error
  control + variational bound.
- `test_canonical_entanglement.py`: canonical vs dense entropy, product-state
  zero, Bell-pair `ln(2)`.
- `test_canonical_score.py`: `canonical_score.py --list` smoke.

### Stage 3A.3: scoring + report

`scripts/canonical_score.py --fast` runs the canonical tests, then
`scripts/run_canonical_smoke.py` writes `docs/CANONICALIZATION_REPORT.md`.

## Pause conditions

Pause and report instead of forcing tests to pass if:

- canonicalization breaks the dense state beyond a global phase,
- compression energy undershoots the exact ground energy beyond tolerance,
- canonical entropy disagrees with the dense SVD reference after two focused
  attempts,
- a change would require touching the differentiable energy path with
  `detach()`/`.data`/unnecessary `.item()`,
- a new large dependency is needed.

## Hard constraints

- No DMRG / TEBD / GPU performance benchmark.
- No enlarged system sizes; no long optimization.
- No changes to Stage 1/2 physics conventions or thresholds.
- No `.detach()`/`.data`/unnecessary `.item()` in the differentiable energy path.
- No large new dependencies (torch/numpy only).
- No broad refactor of legacy prototype files.
- GPU tests stay out of the default test path.
