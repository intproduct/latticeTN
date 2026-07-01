# Stage 3A Canonicalization Specification

## Goal

Advance latticeTN from a dense-reference correctness benchmark into a
genuinely scalable MPS canonical-form library: implement MPS canonicalization
(left / right / mixed) and SVD-based bond compression, with truncation-error
reporting and canonical-form entanglement entropy.

This stage does NOT implement DMRG, TEBD, or any GPU performance benchmark.

## Physics conventions (unchanged)

- `H = J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})`
- `S = sigma / 2` (spin convention, NOT Pauli)
- `J = 1.0`, open boundary
- default dtype `torch.complex128`
- CPU-only for this stage's tests

## MPS tensor convention

Each site tensor has shape `(left_bond, phys, right_bond) = (l, d, r)`, with
the left bond of site 0 and the right bond of site N-1 equal to 1 (open
boundary). Site 0 is the most-significant index in the dense state vector
(consistent with `MPS.to_dense` and `operators.heisenberg_dense`).

Canonical forms:

- **Left-canonical** site `A (l,d,r)`: reshape `(l*d, r)`, columns orthonormal
  (`A^H A = I_r`). Equivalent to an isometry pointing right.
- **Right-canonical** site `B (l,d,r)`: reshape `(l, d*r)`, rows orthonormal
  (`B B^H = I_l`). Isometry pointing left.
- **Mixed-canonical** with center `c`: sites `< c` left-canonical, sites
  `> c` right-canonical, site `c` carries the entanglement. The Schmidt
  coefficients across the cut `[0, c) | [c, N)` live on the left bond of site
  `c` and are read by an SVD of the center tensor reshaped `(l, d*r)`.

## Required capabilities (in `latticetn/canonical.py`)

1. `left_canonical(mps)` — exact QR left sweep.
2. `right_canonical(mps)` — exact LQ right sweep.
3. `mixed_canonical(mps, center)` — left sweep to `center`, right sweep from `N-1` to `center+1`.
4. `canonical_norm(mps)` and `center_frob_norm(mps, center)` — norm checks.
5. `svd_compress(mps, chi)` — SVD truncation capping every bond at `chi`, with
   per-bond discarded-weight truncation-error reporting.
6. `entanglement_entropy(mps, cut)` — canonical-form von Neumann entropy.
7. `from_dense(state, N, ...)` — dense vector to MPS via successive SVDs.

## Differentiability rule

Canonicalization/compression is NON-differentiable preprocessing/postprocessing
(operates under `torch.no_grad` on detached tensors; outputs wrapped via
`MPS.from_tensors(requires_grad=False)`). The existing differentiable
`energy_with_MPO` path is NOT modified and NO `.detach()`/`.data`/unnecessary
`.item()` is added there.

## Test requirements

1. Left-canonical preserves the dense state up to a global phase; sites are
   left-orthonormal.
2. Right-canonical ditto.
3. Mixed-canonical: left of center left-orthonormal, right of center
   right-orthonormal.
4. Canonical norm equals the dense norm.
5. `svd_compress` bond dimensions do not exceed the target `chi`.
6. Compression fidelity is high when `chi` is not truncating (exact recovery).
7. Compression does not spoil the Heisenberg energy beyond control; energy
   must satisfy the variational bound (>= exact ground - tol).
8. Canonical entanglement entropy matches the dense SVD reference.
9. All tests CPU-only, small systems, no long optimization.
10. Stage 1 / Stage 2 (and GPU-readiness files) are not broken.
