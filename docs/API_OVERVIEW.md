# API Overview

Module-by-module reference for `latticetn`. Conventions throughout:
`S = sigma/2`, open boundary, MPS site tensor shape `(left, phys, right)`,
MPO site tensor shape `(left, right, phys_in, phys_out)`, default dtype
`torch.complex128`, default device `cpu`.

> **Mainline vs. baseline.** `ad_variational` and `ad_local` are the **AD
> mainline** (differentiable Rayleigh quotient + torch optimizer). `dmrg` and
> `lanczos` are **classical reference baselines** — never imported by the AD
> modules, never in the loss path. `canonical` provides SVD/QR tools used only
> as optional stabilization/projection/compression, never as the optimizer.

---

## `latticetn.operators` — operators + dense references

Spin operators and dense (small-N) Hamiltonians used as golden references.

- `spin_operators(dtype, device)` → `{"Sx","Sy","Sz","S+","S-"}`, with
  `S = sigma/2`.
- `pauli_matrices(dtype, device)` → `{"X","Y","Z"}` (Pauli, NOT `S`).
- `heisenberg_dense(N, J=1.0, dtype, device)` → dense Heisenberg Hamiltonian
  `(2**N, 2**N)`.
- `tfi_dense(N, J=1.0, h=1.0, dtype, device)` → dense transverse-field Ising.
- `exact_ground_energy(H)` → `(E0, ground_state)` (reference / oracle).

## `latticetn.mps` — matrix-product state

- `MPS(N, dim, chi, dtype=complex128, device="cpu")` — random-init open-boundary
  MPS; tensors are `nn.Parameter` (autograd-friendly).
- `MPS.from_tensors(tensors, dtype, device, requires_grad=False)` — build from
  an explicit per-site list.
- `MPS.to_dense()` → state vector `(dim**N,)` (differentiable).
- `MPS.overlap(other)` → `<self|other>` full complex scalar (differentiable).
- `MPS.energy_with_MPO(mpo)` → differentiable Rayleigh quotient (dense-path
  convenience; the scalable path is `contractions.rayleigh_energy_native`).

## `latticetn.mpo` — matrix-product operator

- `MPO.from_bonds(N, dim, dtype, device)` → builder; chain
  `.generate_heisenberg(J=1.0)` / `.generate_tfi(...)` to populate tensors.
- `MPO.tensors` → list of site tensors `(left, right, phys_in, phys_out)`.

## `latticetn.contractions` — differentiable native contractions (THE LOSS PATH)

Scalable, **fully differentiable** MPS/MPO contractions (no `to_dense`,
polynomial in N and chi). This is where the AD loss lives.

- `native_norm_sq(mps)` / `native_norm(mps)` — `<psi|psi>` / sqrt.
- `native_local_expect(mps, op, site)` — `<psi|op_site|psi>` (differentiable).
- `native_two_site_expect(mps, op1, i, op2, j)` — two-point expectation.
- `native_bond_energy_heisenberg(mps, i)` — `<S_i·S_{i+1}>`.
- `native_correlation(mps, op, i, j)` — two-point correlation.
- `native_mpo_numerator(mps, mpo)` — `<psi|H|psi>` numerator.
- `native_mpo_expectation(mps, mpo)` — Rayleigh ratio.
- `rayleigh_energy_native(mps, mpo)` — **alias of `native_mpo_expectation`; the
  differentiable energy used as the AD loss.**

These contain **no** `eigh`/`svd`/`qr`, **no** `detach`/`.data`/`no_grad`/
unnecessary `.item()` — verified by AST tests.

## `latticetn.observables` — dense-reference observables

Small-N diagnostics (dense-path; pair with `exact_ground_energy` references).

- `dense_expect_local(state, op, site, N)`, `dense_expect_two_site(state, op1, i, op2, j, N)`.
- `dense_bond_energy_heisenberg(state, i, N)`.
- `dense_entanglement_entropy(state, cut, N)`.
- `mps_expect_local(mps, op, site)`, `mps_expect_two_site(...)`,
  `mps_bond_energy_heisenberg(mps, i)`, `mps_entanglement_entropy(mps, cut)`.

## `latticetn.canonical` — SVD/QR canonicalization + compression (Stage 3A)

**Non-differentiable** gauge/compression tools. Permitted roles: gauge fixing,
stabilization, projection, compression, diagnostics. **Never the optimizer,
never in the loss path.**

- `left_canonical(mps)`, `right_canonical(mps)`, `mixed_canonical(mps, center)`
  → new MPS in canonical form (exact QR/LQ sweep; state preserved up to phase).
- `left_orthonormal_error(A)`, `right_orthonormal_error(B)`,
  `left_orthonormal_all(mps)`, `right_orthonormal_all(mps)` — orthonormality
  diagnostics.
- `canonical_norm(mps)`, `center_frob_norm(mps, center)`.
- `svd_compress(mps, chi)` → `(compressed_mps, info)` with per-bond truncation
  errors and bond dims.
- `entanglement_entropy(mps, cut)` — von Neumann entropy across a cut (nats).
- `from_dense(state, N, dim, chi, dtype, device)` — dense → MPS via SVDs.

## `latticetn.ad_variational` — Stage 4R global AD-MPS  🔹 AD MAINLINE

Trains **all** MPS site tensors simultaneously on the differentiable Rayleigh
quotient.

- `ADVariationalMPS(mps, mpo)` — wraps MPS (tensors → trainable `nn.Parameter`)
  + MPO; `.energy()`/`.loss()` → `rayleigh_energy_native` (autograd-clean).
- `train_ad_mps(admps, num_steps, lr, optimizer="adam"|"lbfgs", projection=...)`
  → history dict (`energy_history`, `grad_norm_history`, `norm_history`,
  `canonical_error_history`, `final_energy`, ...).
- `projection="none"|"tensor_norm"|"canonical"` — post-step gauge projection
  (under `no_grad` mutating `.data`, outside the loss graph).

## `latticetn.ad_local` — Stage 5A AD local-tensor optimization  🔹 AD MAINLINE

Trains **one center tensor at a time** on the differentiable Rayleigh quotient,
sweeping the orthogonality center by QR. The autograd analogue of DMRG's local
update — **gradient descent, not a local eigensolver**.

- `ADLocalOptimizer(mps, mpo, center)` — mixed-canonical; only the center
  tensor is trainable. `.energy()`/`.loss()` → `rayleigh_energy_native`.
  `.move_center(new_center)` shifts the center by QR sweeps (center movement,
  not the optimizer).
- `train_ad_local(mps, mpo, num_sweeps, local_steps, lr, optimizer="lbfgs",
  stabilization="none"|"tensor_norm"|"qr"|"canonical")` → history dict +
  per-sweep records.
- `stabilization` is **optional post-step stabilization only**, under `no_grad`
  mutating `.data` — never the solver, never in the loss path.

## `latticetn.ad_two_site` — Stage 5B two-site AD local-tensor optimization  🔹 AD MAINLINE

Trains **one two-site block `Θ` at a time** on the differentiable local
Rayleigh quotient `E(Θ)=<Θ|H_eff|Θ>/<Θ|Θ>`, sweeping the active bond and
optionally growing / truncating the bond at the SVD split. The autograd
analogue of two-site DMRG — **gradient descent on `Θ`, not a local
eigensolver**.

- `ADTwoSiteOptimizer(mps, mpo, bond)` — two-site mixed-canonical; only `Θ` is
  trainable. `.energy()`/`.loss()` → the differentiable local Rayleigh quotient
  (pure einsum on `Θ` + frozen constant MPO environments).
  `.split(max_bond_dim, cutoff, direction)` → SVD split of `Θ` back into two
  site tensors with optional truncation (compression, **not** the solver).
- `train_ad_two_site(mps, mpo, num_sweeps, local_steps, lr, optimizer="lbfgs",
  max_bond_dim, cutoff, stabilization)` → history dict + per-sweep/per-bond
  records (`energy_history`, `bond_dim_history`, `truncation_error_history`).

## `latticetn.dmrg` — classical two-site DMRG (Stage 4A/4B)  ⚠️ REFERENCE BASELINE

Classical two-site DMRG with a dense or Lanczos local eigensolver. **Not**
imported by the AD modules; run only from `dmrg_score.py` /
`dmrg_benchmark_score.py` for baseline comparison.

- `run_dmrg(mps, mpo, chi, num_sweeps, solver="dense"|"lanczos")` → result
  dict (`final_energy`, `history`, `below_ground`, ...).
- internals: `two_site_sweep`, `effective_hamiltonian`, `two_site_update`,
  `local_ground_state`.

## `latticetn.lanczos` — Krylov local eigensolver (Stage 4B)  ⚠️ REFERENCE BASELINE

Matrix-free Krylov-subspace lowest-eigenpair solver, used by DMRG's `lanczos`
solver. **Not** imported by the AD modules.

- `lanczos_lowest_eigenpair(apply, dim, dtype, ...)` → lowest eigenpair.
- `ritz_quotient(apply, v)` → Ritz quotient diagnostic.

---

## Benchmark / score scripts (Stage 6A)

- `scripts/run_ad_gpu_benchmark.py` — CPU/GPU benchmark of the three AD
  mainline solvers (global AD-MPS, one-site AD, two-site AD) with exact / DMRG
  reference baselines. GPU is opt-in (`LATTICETN_RUN_GPU=1`, uses `cuda:0`;
  clean-skips if CUDA unavailable). Writes `docs/AD_GPU_BENCHMARK_REPORT.md`
  + JSON results. No AD loss path is modified; it only calls `train_ad_*` with
  device-placed MPS/MPO.
- `scripts/ad_gpu_benchmark_score.py` — Stage 6A score: runs the three
  benchmark test files, regenerates the report, checks required terms.

---

For usage flows, see [`USER_GUIDE.md`](USER_GUIDE.md). For the standing policy
on what is mainline vs. baseline, see [`AD_MAINLINE_POLICY.md`](AD_MAINLINE_POLICY.md).
