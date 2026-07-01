# Stage 4B Scalable DMRG Benchmark Protocol

## Stop condition

Stage 4B is complete only when ALL hold:

1. `python scripts/validation_score.py --fast` exits 0.
2. `python scripts/benchmark_score.py --fast` exits 0.
3. `python scripts/canonical_score.py --fast` exits 0.
4. `python scripts/contraction_score.py --fast` exits 0.
5. `python scripts/dmrg_score.py --fast` exits 0.
6. `python scripts/dmrg_benchmark_score.py --fast` exits 0.
7. `docs/DMRG_BENCHMARK_REPORT.md` contains:
   - dense-vs-matrix-free comparison,
   - Lanczos-vs-dense eigensolver comparison,
   - small-system exact comparison (N<=6, both solvers),
   - N=10/12 smoke (no dense ED),
   - chi sweep,
   - runtime, pass/fail, known limitations.
8. `docs/CLAUDE_PROGRESS_DMRG_BENCHMARK.md` records changes, commands,
   results, next-step suggestions.

GPU smoke is NOT a blocking item; existing GPU-readiness files must stay intact
and default test paths must not depend on GPU.

## Files

New:

- `latticetn/lanczos.py` — Lanczos lowest-eigenpair from a matrix-free apply.
- `tests/test_dmrg_matrix_free_heff.py`
- `tests/test_dmrg_lanczos_solver.py`
- `tests/test_dmrg_benchmark_smoke.py`
- `tests/test_dmrg_benchmark_score.py`
- `scripts/dmrg_benchmark_score.py`
- `scripts/run_dmrg_benchmark.py`
- `docs/DMRG_BENCHMARK_SPEC.md`
- `docs/DMRG_BENCHMARK_PROTOCOL.md`
- `docs/DMRG_BENCHMARK_REPORT.md` (generated)
- `docs/CLAUDE_PROGRESS_DMRG_BENCHMARK.md`

Modified (non-breaking): `latticetn/dmrg.py` — add `matrix_free_apply`, a
`solver` option to `two_site_sweep` / `run_dmrg`, and `energy_per_bond` +
`solver` fields in the result dict. The Stage 4A dense path is preserved
(`solver="dense"` default) and unchanged.

## Required implementation stages

### 4B.1 matrix-free apply + Lanczos

`dmrg.matrix_free_apply` (already uses the verified `apply_heff`); `lanczos.py`
with full-reorthogonalization Lanczos returning (E0, vec).

### 4B.2 solver option

`solver="dense"|"lanczos"` in `two_site_sweep` / `run_dmrg`. Dense unchanged;
lanczos branches into `lanczos_lowest_eigenpair` on the matrix-free apply.

### 4B.3 tests

- matrix-free vs dense apply + lowest-eigenvalue recovery.
- Lanczos vs eigh; Lanczos DMRG recovers exact N=4/6; matches dense DMRG.
- benchmark smoke: both solvers vs exact N<=6, not-below-ground, chi sweep
  non-worsening, N=10 smoke (finite/down/bond-cap/runtime), solver recorded.

### 4B.4 score + report

`scripts/dmrg_benchmark_score.py --fast` runs tests then
`scripts/run_dmrg_benchmark.py` writes `docs/DMRG_BENCHMARK_REPORT.md`.

## Pause conditions

Pause and report instead of forcing tests to pass if:

- matrix-free apply disagrees with dense H_eff after two focused attempts,
- Lanczos fails to match dense eigh to spec on small systems,
- DMRG (either solver) undershoots exact ground beyond tolerance,
- the autograd energy path is touched with `detach()/.data`/unnecessary `.item`,
- a new large dependency is needed,
- runtime becomes too high for CPU fast validation (cap N=12 / few sweeps;
  report if exceeded).

## Hard constraints

- No TEBD / TDVP / finite-T / GPU performance benchmark.
- No changes to Stage 1/2/3A/3B/4A physics conventions or thresholds.
- `energy_with_MPO` / `rayleigh_energy_native` / Stage 4A dense DMRG reference
  not broken.
- No `.detach()/.data`/unnecessary `.item()` in the autograd energy path.
- No large new dependencies; no long training.
- GPU tests stay out of the default test path.
