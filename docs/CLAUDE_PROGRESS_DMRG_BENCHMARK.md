# Claude Progress Log: Stage 4B Scalable DMRG Benchmark

## Checkpoint 2026-07-01 — Stage 4B complete

```text
Date/time: 2026-07-01
Stage: Stage 4B scalable DMRG solver + DMRG benchmark (final)
Files changed (NEW):
  - latticetn/lanczos.py : Lanczos lowest-eigenpair from a matrix-free apply
        (full reorthogonalization, multi-restart, tol/max_iter caps).
  - tests/test_dmrg_matrix_free_heff.py : apply vs dense; lowest-eigenvalue vs exact
  - tests/test_dmrg_lanczos_solver.py   : Lanczos vs eigh; Lanczos DMRG exact N4/N6; matches dense
  - tests/test_dmrg_benchmark_smoke.py  : both solvers vs exact N<=6, not-below-ground, chi sweep, N=10 smoke
  - tests/test_dmrg_benchmark_score.py  : dmrg_benchmark_score.py --list smoke
  - scripts/dmrg_benchmark_score.py     : --fast / --list
  - scripts/run_dmrg_benchmark.py       : CLI (N,chi,sweeps,seed,solver,dtype,device); writes report
  - docs/DMRG_BENCHMARK_SPEC.md
  - docs/DMRG_BENCHMARK_PROTOCOL.md
  - docs/DMRG_BENCHMARK_REPORT.md       (generated)
  - docs/CLAUDE_PROGRESS_DMRG_BENCHMARK.md (this file)
Files changed (MODIFIED, non-breaking):
  - latticetn/dmrg.py : added matrix_free_apply(...); solver='dense'|'lanczos'
        option in two_site_sweep / run_dmrg; energy_per_bond + solver fields
        in result dict. Stage 4A dense path preserved (default solver='dense',
        behavior unchanged when solver='dense').
Commands run (and results):
  - validation / benchmark / canonical / contraction / dmrg _score --fast
        -> all exit 0
  - pytest -q (4 Stage 4B test files) -> 15 passed
  - python scripts/dmrg_benchmark_score.py --fast -> exit 0 (DMRG benchmark score: PASS)
  - GPU smoke (matched GPU) re-run for regression -> exit 0 (intact)
Numerical highlights (from docs/DMRG_BENCHMARK_REPORT.md):
  - matrix-free vs dense apply: ||mf(x) - dense@x|| = 3.5e-16 (D=16, N=4 bond 1).
  - Lanczos E0 -1.616025403784 vs dense eigh -1.616025403784, diff 2.2e-16.
  - Exact compare N<=6, both solvers: final == exact (abs err ~4e-16), not below ground.
  - chi sweep N=6: chi=4 E=-2.49254 (trunc 3.5e-4); chi=8,16 exact -2.49358.
    Energy non-increasing with chi (within tol).
  - N=10 (lanczos, 3 sweeps, chi cap 16): 0.082 -> -4.25803520, max bond 16,
    runtime 5.8s, finite, energy decreased.
  - N=12 (lanczos, 3 sweeps, chi cap 16): -0.483 -> -5.14209057, max bond 16,
    runtime 95.7s, finite, energy decreased (under the 240s cap).
Differentiability / autograd note:
  - DMRG + Lanczos run under torch.no_grad on detached tensors (non-differentiable).
  - autograd energy_with_MPO / rayleigh_energy_native NOT modified; Stage 4A
    dense DMRG reference preserved (default solver='dense').
  - Reading the global DMRG energy as a float for the report uses
    rayleigh_energy_native (a separate scalar eval, not inside a training graph).
Reusability note:
  - The matrix-free apply reuses the Stage 4B/4A verified apply_heff einsum;
    no new leg mapping was needed, so the dense-vs-matrix-free agreement is
    essentially exact by construction (verified to ~1e-16).
Current failing item: none
Next action / suggestions:
  - Stage 4C (optional): single-site DMRG + center-of-orthogonality sweep +
    density-matrix perturbation; would tighten convergence at fixed chi.
  - Optional: replace simple Lanczos with a LOBPCG/block/Lanczos-with-restart
    for better convergence at large D; the matrix-free apply already supports it.
  - Optional: parallelize the per-bond env updates (carry L/R across bonds
    instead of re-canonicalizing each bond) for a real speedup at N>=12.
  - Optional: integrate the Stage 4B DMRG into the Stage 2/3 benchmark reports
    as a converged-state reference alongside the variational/dense references.
Notes:
  - conventions unchanged: S=sigma/2, J=1.0, open boundary, complex128.
  - No TEBD / TDVP / finite-T / GPU performance benchmark.
  - GPU-readiness files untouched; Stage 4B tests are CPU-only and not in any
    default GPU / Stage1-2-3A-3B-4A path.
```
