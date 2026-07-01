# Stage 3B Native Contraction Specification

## Goal

Reduce Stage 2 observables' dependence on `MPS.to_dense()` by implementing
native tensor-network contractions that scale polynomially in N and chi. The
scalable path covers norm, local/two-site observables, correlation, MPO
expectation, and Rayleigh energy. Small systems are validated against dense
references; a larger system (N=20, chi<=8) gets a finite/shape/execution smoke
without `to_dense` or exact diagonalization.

No DMRG, TEBD, or GPU performance benchmark.

## Physics conventions (unchanged)

- `H = J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})`
- `S = sigma / 2` (spin, NOT Pauli)
- `J = 1.0`, open boundary
- default dtype `torch.complex128`
- CPU-only for this stage's tests

## Tensor conventions

- MPS site tensor `A`: `(left_bond, phys, right_bond) = (l, s, r)`.
- MPO site tensor `W`: `(left_bond, right_bond, phys_in, phys_out) = (l, r, si, so)`.

Environment tensors sweep left-to-right (norm / local / two-site / MPO) holding
paired (bra, ket) legs so bonds contract correctly.

## Required capabilities (`latticetn/contractions.py`)

1. `native_norm_sq(mps)` / `native_norm(mps)` — norm contraction, no `to_dense`.
2. `native_local_expect(mps, op, site)` — `<O_i>`.
3. `native_two_site_expect(mps, op1, i, op2, j)` — `<op1_i op2_j>` (caller
   order respected; works for non-commuting ops, i>j).
4. `native_bond_energy_heisenberg(mps, i)` — `<S_i . S_{i+1}>` via SzSz +
   (1/2)(S+S- + S-S+).
5. `native_correlation(mps, op, i, j)` — e.g. `<Sz_i Sz_j>`.
6. `native_mpo_numerator(mps, mpo)` — `<psi|MPO|psi>` (raw numerator).
7. `native_mpo_expectation(mps, mpo)` / `rayleigh_energy_native(mps, mpo)` —
   `<psi|H|psi>/<psi|psi>` (native energy path).

## Differentiability rule

- The energy path (`native_mpo_numerator`, `rayleigh_energy_native`,
  `native_norm*`) is fully differentiable: NO `.detach()`/`.data`/unnecessary
  `.item()`/`torch.no_grad()`.
- Observable / report helpers stay differentiable but callers may wrap them in
  `torch.no_grad()`; the energy path is kept separate.

## Test requirements

1. Native norm == dense norm (N<=6 random MPS).
2. Native `<Sz_i>` == dense reference.
3. Native `<Sz_i Sz_j>` == dense reference (incl. i>j).
4. Native bond energy == dense reference.
5. Native MPO energy == dense-state energy and == Stage 1 `energy_with_MPO`.
6. Native energy `backward()` -> all MPS param grads non-None.
7. On canonical/compressed MPS, native observables == dense reference.
8. Scalability smoke (N=20, chi<=8): NO `to_dense`, NO exact diagonalization;
   only finite/shape/device/dtype/execution + cheap runtime.
9. All default tests CPU-only, small systems, fast.

## Constraints

- Conventions unchanged; Stage 1/2/3A thresholds not relaxed.
- Differentiable energy path not broken.
- No large dependencies (torch/numpy only).
- No broad refactor of legacy files.
- GPU tests stay out of the default path.
