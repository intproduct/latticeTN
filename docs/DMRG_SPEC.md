# Stage 4A Two-Site DMRG Specification

## Goal

Implement a small-system, correctness-focused **two-site DMRG** for the
open-boundary 1D spin-1/2 Heisenberg chain, building on the Stage 3A
canonicalization/compression and Stage 3B native contractions. Validate against
exact diagonalization for N<=6; a medium smoke (N=8) checks finiteness, energy
decrease, and bond-dim caps without dense ED.

No TEBD, TDVP, GPU performance benchmark, or large-N optimization.

## Physics conventions (unchanged)

- `H = J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})`
- `S = sigma / 2` (spin, NOT Pauli)
- `J = 1.0`, open boundary
- default dtype `torch.complex128`
- CPU-only for tests

## Tensor conventions

- MPS `A_i`: `(l, s, r)`. MPO `W_i`: `(l, r, s_in, s_out)`.
- Two-site block `Theta(l, s_i, s_{i+1}, r)` on bond `(i, i+1)`.

## Required capabilities (`latticetn/dmrg.py`)

1. MPO left/right environments (`mpo_left_env`, `mpo_right_env`), three-leg
   `(bra, mpo, ket)` environment tensors.
2. Two-site effective Hamiltonian `effective_hamiltonian(mps, mpo, i)` — dense
   Hermitian matrix acting on the flattened `Theta`; built by applying `H_eff`
   to basis vectors (D small). `apply_heff` exposes the core contraction.
3. Local two-site ground-state solve via `torch.linalg.eigh`
   (`local_ground_state`).
4. SVD two-site tensor split (`two_site_update`).
5. `chi` truncation with truncation-error (discarded-weight) reporting.
6. One-site sweep directions: left-to-right and right-to-left
   (`two_site_sweep(direction=...)`).
7. Minimal two-site DMRG driver (`run_dmrg`) alternating sweeps.
8. Global DMRG energy via the native MPO Rayleigh quotient
   (`contractions.rayleigh_energy_native`).
9. Generate `docs/DMRG_REPORT.md` (`scripts/run_dmrg_small.py`).

## Differentiability

DMRG is a NON-differentiable classical optimizer (operates under `torch.no_grad`
on detached tensors). The autograd energy path (`energy_with_MPO`,
`rayleigh_energy_native`) is NOT modified; no `.detach()`/`.data`/unnecessary
`.item()` added there. (Reading energy scalars as floats for the report is a
report-path, outside the training graph.)

## Test requirements

- N<=6: DMRG final energy vs exact (within tol).
- DMRG energy must not undershoot exact ground beyond tolerance.
- Sweep energy overall non-increasing; tiny numerical wiggles must be reported.
- `H_eff` Hermitian.
- Local `H_eff` energy aligned with full native MPO energy on small systems.
- After two-site update, bond dims <= chi.
- SVD truncation error non-negative and finite.
- N=8 (or N=10) smoke: finite, energy decreases, bond dims, runtime; no dense ED.
- Default tests CPU-only, small systems, fast.

## Constraints

- Conventions unchanged; Stage 1/2/3A/3B thresholds not relaxed.
- `energy_with_MPO` / `rayleigh_energy_native` not broken.
- No large dependencies; no broad refactor of legacy files.
- GPU tests stay out of the default path.
- No long training.
