# Stage 4B Scalable DMRG Benchmark Specification

## Goal

Upgrade the Stage 4A two-site DMRG into a more scalable DMRG **benchmark tool**:
add a **matrix-free** `H_eff` apply and an optional **Lanczos** local solver so
larger bond dimension / system sizes are reachable, while keeping the Stage 4A
dense `H_eff` path as the reference.

No TEBD, TDVP, finite-temperature algorithms, or GPU performance benchmark.

## Physics conventions (unchanged)

- `H = J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})`
- `S = sigma / 2` (spin, NOT Pauli), `J = 1.0`, open boundary
- default dtype `torch.complex128`, CPU-only for tests

## Required capabilities

1. Matrix-free two-site `H_eff` apply: `dmrg.matrix_free_apply(mps, mpo, i)`
   returns `f(v) -> H_eff @ v` (no D x D materialization).
2. Dense `H_eff` vs matrix-free apply agree on small systems.
3. Optional Lanczos/local iterative eigensolver: `lanczos.lanczos_lowest_eigenpair`.
4. Lanczos lowest eigenvalue matches dense `torch.linalg.eigh` on small systems.
5. DMRG driver supports `solver="dense"` and `solver="lanczos"`
   (`dmrg.run_dmrg(..., solver=...)`, `dmrg.two_site_sweep(..., solver=...)`).
6. Benchmark script accepts `N, chi, sweeps, seed, solver, dtype, device`.
7. Benchmark report outputs: energy history, final energy, energy per bond,
   runtime, max bond dim, truncation errors, solver type.

## Test requirements

- N<=6: dense and Lanczos DMRG final energy vs exact (within tol).
- Final energy must not undershoot exact ground beyond tolerance.
- Matrix-free `H_eff @ x` vs dense `H_eff @ x` agree.
- Lanczos small-system lowest eigenvalue matches dense eigensolver.
- N=10 or N=12 CPU smoke (no dense ED): finite, energy decreases, bond dims
  <= chi, reasonable runtime.
- chi sweep: energy must not materially worsen as chi grows; report any
  numerical wiggles.
- Default tests CPU-only, small systems, fast.

## Constraints

- Conventions unchanged; Stage 1/2/3A/3B/4A thresholds not relaxed.
- `energy_with_MPO`, `rayleigh_energy_native`, and the Stage 4A dense DMRG
  reference must NOT be broken.
- DMRG/Lanczos may run under `torch.no_grad()`; no autograd-graph requirement.
- No `.detach()/.data`/unnecessary `.item()` added to the autograd energy path.
- No large dependencies; no long training; GPU tests out of the default path.
