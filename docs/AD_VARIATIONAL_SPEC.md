# Stage 4R AD-MPS Variational Solver Specification

## Goal

Realign latticeTN to its automatic-differentiation tensor-network mainline: a
**first-class AD-MPS variational solver** that optimizes the MPS tensors
directly via PyTorch autograd + a torch optimizer on the differentiable
Rayleigh quotient.

The classical Lanczos/DMRG code (Stage 4A/4B) is a **reference baseline only**
and is NOT used in the AD optimization path. This stage does not extend
Lanczos/DMRG.

## Physics conventions (unchanged)

- `H = J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})`
- `S = sigma / 2` (spin, NOT Pauli), `J = 1.0`, open boundary
- default dtype `torch.complex128`, CPU-only for tests

## Required capabilities (`latticetn/ad_variational.py`)

1. `ADVariationalMPS` — an MPS whose tensors are `torch.nn.Parameter`s
   (trainable, requires_grad=True), wrapping the Heisenberg MPO.
2. Differentiable loss `E = <psi|H|psi>/<psi|psi>` via
   `contractions.rayleigh_energy_native` (no `to_dense`).
3. Optimizer support: Adam (default) and optional LBFGS.
4. Fixed-seed initialization.
5. Training loop recording: energy history, gradient norm, MPS norm, max bond
   dimension.
6. Small-system (N=4/6) comparison against exact diagonalization.
7. Comparison against the classical DMRG reference (DMRG is a reference ONLY;
   never inside the AD path).
8. Generate `docs/AD_VARIATIONAL_REPORT.md`.

## Autograd rule (the mainline contract)

- The loss path uses NO `det()` / `.data` / unnecessary `.item()` /
  `torch.no_grad()`. No local eigensolver, no Lanczos, no DMRG sweep in the
  optimization path.
- Per-tensor L2 renormalization after each step is performed OUTSIDE the loss
  graph (under `no_grad`, mutating `.data`) — a scale-invariant stability
  projection (the Rayleigh quotient is unchanged), identical in spirit to the
  Stage 1 `_full_normalize` routine. It is NOT part of the differentiable
  energy computation.

## Test requirements

- Loss is a scalar, finite, `requires_grad=True`.
- `loss.backward()` -> all trainable MPS param grads non-None and finite.
- No `detach()`/`.data`/unnecessary `.item()`/`no_grad()` in the loss path
  (verified by AST source inspection).
- Adam short training lowers the energy.
- N=4/6 final energy not below exact ground beyond tolerance; tol recorded in
  the report (N=4 < 1e-6, N=6 < 1e-3 at 200 Adam steps).
- AD final energy close to the DMRG reference (within 1e-3).
- Default tests CPU-only, small systems, fast.
- Conventions unchanged.

## Constraints

- Stage 1/2/3A/3B/4A/4B thresholds not relaxed; existing interfaces not broken.
- `energy_with_MPO` / `rayleigh_energy_native` not modified.
- No large dependencies; no long training; GPU tests out of the default path.
