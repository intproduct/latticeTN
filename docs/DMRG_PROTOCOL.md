# Stage 4A Two-Site DMRG Protocol

## Stop condition

Stage 4A is complete only when ALL hold:

1. `python scripts/validation_score.py --fast` exits 0.
2. `python scripts/benchmark_score.py --fast` exits 0.
3. `python scripts/canonical_score.py --fast` exits 0.
4. `python scripts/contraction_score.py --fast` exits 0.
5. `python scripts/dmrg_score.py --fast` exits 0.
6. `docs/DMRG_REPORT.md` contains:
   - exact comparison (N<=6),
   - DMRG energy history,
   - chi, truncation errors, bond dims,
   - `H_eff` hermiticity,
   - pass/fail, known limitations.
7. `docs/CLAUDE_PROGRESS_DMRG.md` records changes, commands, results,
   next-step suggestions.

GPU smoke is NOT a blocking item; existing GPU-readiness files must stay intact
and default test paths must not depend on GPU.

## Files

New:

- `latticetn/dmrg.py` — two-site DMRG core.
- `tests/test_dmrg_environments.py`
- `tests/test_dmrg_effective_hamiltonian.py`
- `tests/test_dmrg_two_site_update.py`
- `tests/test_dmrg_sweep_smoke.py`
- `tests/test_dmrg_score.py`
- `scripts/dmrg_score.py`
- `scripts/run_dmrg_small.py`
- `docs/DMRG_SPEC.md`
- `docs/DMRG_PROTOCOL.md`
- `docs/DMRG_REPORT.md` (generated)
- `docs/CLAUDE_PROGRESS_DMRG.md`

Modified (optional, non-breaking): `latticetn/canonical.py`,
`latticetn/contractions.py`, `latticetn/mps.py`. This stage was additive; no
modification was strictly required.

## Required implementation stages

### 4A.1 core module

Implement `latticetn/dmrg.py`: two-site mixed-canonical form, MPO environments,
effective Hamiltonian, local eigh solve, SVD two-site update with truncation,
sweep directions, minimal DMRG driver using native Rayleigh energy.

### 4A.2 tests

- environments: shape + full-MPO-numerator recovery.
- effective Hamiltonian: Hermitian, local eigenvalue vs exact, local-vs-full
  alignment in the full-chi limit.
- two-site update: bond cap, left/right canonical sites, truncation error
  nonneg/finite/zero at full chi, state reconstruction at full chi.
- sweep smoke: N<=6 vs exact, not-below-ground, monotonic-ish; N=8 smoke
  (finite, energy down, bond cap, runtime); `dmrg_score.py --list`.

### 4A.3 score + report

`scripts/dmrg_score.py --fast` runs the tests, then `scripts/run_dmrg_small.py`
writes `docs/DMRG_REPORT.md`.

## Pause conditions

Pause and report instead of forcing tests to pass if:

- DMRG energy undershoots exact ground beyond tolerance,
- `H_eff` is not Hermitian after re-derivation,
- local vs full MPO energy cannot be aligned after two focused attempts,
- the autograd energy path is touched with `detach()/.data`/unnecessary `.item`,
- a new large dependency is needed,
- runtime becomes too high for CPU fast validation (N>10, many sweeps).
  (N<=6 with full chi converges in <=4 sweeps; N=8 smoke <=3 sweeps is plenty.)

## Hard constraints

- No TEBD / TDVP / GPU performance benchmark / large-N optimization.
- No changes to Stage 1/2/3A/3B physics conventions or thresholds.
- `energy_with_MPO` / `rayleigh_energy_native` not broken.
- No `.detach()/.data`/unnecessary `.item()` in the autograd energy path.
- No large new dependencies.
- No broad refactor of legacy files.
- GPU tests stay out of the default test path.
- No long training.
