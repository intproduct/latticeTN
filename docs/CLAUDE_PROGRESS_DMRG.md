# Claude Progress Log: Stage 4A Two-Site DMRG

## Checkpoint 2026-07-01 — Stage 4A complete

```text
Date/time: 2026-07-01
Stage: Stage 4A two-site DMRG primitives + minimal sweep (final)
Files changed (NEW):
  - latticetn/dmrg.py : two-site DMRG core (non-differentiable)
        mixed_canonical_two_site, mpo_left_env / mpo_right_env,
        apply_heff, effective_hamiltonian (dense Hermitian, basis-vector build),
        local_ground_state, two_site_update (SVD split + chi truncation +
        discarded-weight reporting + sweep-direction canonical absorption),
        two_site_sweep (right/left), run_dmrg driver (alternating sweeps,
        native Rayleigh energy via contractions.rayleigh_energy_native).
  - tests/test_dmrg_environments.py        : env shapes + full-MPO-numerator recovery
  - tests/test_dmrg_effective_hamiltonian.py : Hermitian, local vs exact, full-chi alignment
  - tests/test_dmrg_two_site_update.py     : bond cap, canonical sites, trunc err, state recovery
  - tests/test_dmrg_sweep_smoke.py         : N<=6 vs exact, not-below-ground, monotonic-ish; N=8 smoke
  - tests/test_dmrg_score.py               : dmrg_score.py --list smoke
  - scripts/dmrg_score.py                  : --fast / --list
  - scripts/run_dmrg_small.py              : writes docs/DMRG_REPORT.md
  - docs/DMRG_SPEC.md
  - docs/DMRG_PROTOCOL.md
  - docs/DMRG_REPORT.md                    (generated)
  - docs/CLAUDE_PROGRESS_DMRG.md           (this file)
Files changed (MODIFIED): none required (stage is additive; reused Stage 3A
  canonical.from_tensors + Stage 3B contractions.rayleigh_energy_native; mps.py
  already has from_tensors from Stage 3A).
Commands run (and results):
  - python scripts/validation_score.py --fast   -> exit 0 (PASS)
  - python scripts/benchmark_score.py --fast    -> exit 0 (PASS)
  - python scripts/canonical_score.py --fast    -> exit 0 (PASS)
  - python scripts/contraction_score.py --fast  -> exit 0 (PASS)
  - pytest -q (5 Stage 4A test files)           -> 19 passed
  - python scripts/dmrg_score.py --fast         -> exit 0 (DMRG score: PASS)
  - GPU smoke (matched GPU) re-run for regression -> exit 0 (intact)
Numerical highlights (from docs/DMRG_REPORT.md):
  - N=4 chi=8 (4 sweeps): DMRG final -1.6160254038 == exact (abs err 4.4e-16),
    below_ground False, H_eff herm err 0.0e+00, max bond 4.
  - N=6 chi=8 (4 sweeps): DMRG final -2.4935771339 == exact (abs err 4.4e-16),
    below_ground False, H_eff herm err 1.8e-16, max bond 8.
  - N=8 chi=8 (3 sweeps, no dense ED): energy decreases from -0.3629 to the
    DMRG-converged value, finite, max bond <= 8, runtime ~s level.
  - Eigenvalue alignment check: in the full-chi limit the local H_eff lowest
    eigenvalue equals the exact global ground energy (verified at N=4).
Differentiability / autograd note:
  - DMRG is NON-differentiable: all work under torch.no_grad on detached
    tensors. The autograd energy_with_MPO / rayleigh_energy_native paths are
    NOT modified. Reading the global DMRG energy as a float (for the report)
    happens on the current MPS via rayleigh_energy_native — a separate scalar
    evaluation, not inside a training graph.
  - One subtle bug during development: an initial right-MPO-environment einsum
    bound the LEFT mpo bond (free) instead of the RIGHT mpo bond (contracted),
    making H_eff non-Hermitian (err ~4.7) and the local eigenvalue undershoot
    the exact ground (-3.74 vs -1.616). Caught immediately by the Hermitian +
    below-ground guards; fixed by the Verified leg mapping
    (einsum 'abc,dga,ebfg,hfc->deh' for the right env).
Current failing item: none
Next action / suggestions:
  - Stage 4B (optional): efficiency — replace the basis-vector Heff build with
    a single reshape-and-einsum dense Heff (or an iterative eigensolver /
    Lanczos for the local solve) so N can grow beyond ~10. The current
    O(N * D^2) with D=l*d*d*r is fine for N<=8/10 but not large N.
  - Optional: noise / subspace expansion / mixed precision — out of scope here.
  - Optional: single-site DMRG with the center-of-orthogonality shift, reusing
    mixed_canonical_two_site's primitives.
  - Optional: integrate DMRG-converged states into the Stage 2 benchmark
    observables for a converged-vs-variational comparison.
Notes:
  - conventions unchanged: S=sigma/2, J=1.0, open boundary, complex128.
  - No TEBD / TDVP / GPU performance benchmark / large-N optimization.
  - GPU-readiness files untouched; DMRG tests are CPU-only and not in any
    default GPU / Stage1-2-3A-3B path.
```
