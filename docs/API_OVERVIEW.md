# API Overview

Module-by-module reference for `latticetn`. Conventions throughout:
`S = sigma/2`, open boundary, MPS site tensor shape `(left, phys, right)`,
MPO site tensor shape `(left, right, phys_in, phys_out)`, default dtype
`torch.complex128`, default device `cpu`.

> **Mainline vs. baseline.** `ad_variational` and `ad_local` are the **AD
> mainline** (differentiable Rayleigh quotient + torch optimizer). `dmrg` and
> `lanczos` are **classical reference baselines** — never imported by the AD
> modules, never in the loss path. Traditional `tdvp` is a classical real-time
> baseline, separate from the future AD-TDVP research path. `canonical`
> provides SVD/QR tools used only
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
- `spinless_fermion_dense(N, t=1.0, V=0.0, mu=0.0, dtype, device)` → dense
  open-boundary spinless fermion t-V chain (Stage 7A). Built with the explicit
  global Jordan-Wigner parity string `F = (-1)^n` — a genuine fermionic
  Hamiltonian, NOT a hard-core-boson one.
- `hubbard_dense(N, t=1.0, U=4.0, mu=0.0, h=0.0, dtype, device)` → dense
  open-boundary spinful Hubbard chain (Stage 7C), d=4. Built with the explicit
  global Jordan-Wigner parity (site-major 2N-mode ordering; per-site parity
  `P = F_up x F_down` on the left-factor site of each spin-resolved hop) — a
  genuine fermionic Hamiltonian, NOT a spin / hard-core-boson one.
- `exact_ground_energy(H)` → `(E0, ground_state)` (reference / oracle).

## `latticetn.fermion_operators` — fermion local operators (Stage 7A / 7C)

Local fermion operators. This is 1D Jordan-Wigner, NOT graded fermionic
tensors.

- `fermion_operators(dtype, device)` → `{"I","c","cdag","n","F","n_minus_half"}`
  (Stage 7A, spinless, 2x2) with `c = [[0,1],[0,0]]`, `c^d = [[0,0],[1,0]]`,
  `n = c^d c = diag(0,1)`, `F = (-1)^n = diag(1,-1)` (the JW parity string),
  `n-1/2 = diag(-1/2,1/2)`. Algebra: `{c,c^d}=I`, `c^2=(c^d)^2=0`, `n=c^d c`,
  `F^2=I`, `F c = -c F`.
- `hubbard_local_operators(dtype, device)` → `{"I","cup","cdagup","cdown",
  "cdagdown","nup","ndown","ntot","sz","double_occ","parity"}` (Stage 7C,
  spinful, 4x4) in the standard Hubbard basis `|0>,|up>,|down>,|up,down>`.
  On-site JW order is up-first (the down operators carry `F_up` internally);
  `parity = F_up x F_down = diag(1,-1,-1,1)`. Algebra: same-spin
  `{c_s,c^d_s}=I`, cross-spin `{c_up,c_down}=0`, etc.; `parity` anticommutes
  with `c`/`c^d`.

## `latticetn.model_builder` — general 1D model builder (Stage 7B)

A **model/MPO construction layer, NOT a new solver**. Describes a 1D
open-boundary chain as a `ModelSpec` (list of terms + explicit
`statistics` = "boson"|"fermion"), and builds dense/MPO from it. The AD
mainline is unchanged.

- `ModelSpec(N, dim, statistics, terms, name, dtype, device)` — the spec.
- Term types: `OnsiteTerm(op, coeff)`; `TwoSiteTerm(op_i, op_j, coeff)`
  (bosonic/spin, NO JW string); `FermionHopTerm(coeff)` (JW, carries the
  parity string — NOT hard-core-boson); `DensityDensityTerm(coeff, op)`
  (diagonal, no JW string).
- Presets: `heisenberg_model(N, J)` (S=sigma/2, boson);
  `spinless_fermion_tv_model(N, t, V, mu)` (Stage 7A t-V chain, fermion);
  `hubbard_model(N, t, U, mu, h)` (Stage 7C spinful Hubbard, fermion, d=4).
- `build_dense(spec)` → `(2**N, 2**N)` dense Hamiltonian (dispatches to the
  existing validated references; byte-identical to Stage 1/7A).
- `build_mpo(spec)` → `MPO` (dispatches to `MPO.generate_*`; `to_dense`
  matches `build_dense`). For fermions the JW parity string is carried by the
  MPO's parity-carrying virtual state.

## `latticetn.benchmarking` — unified CPU/GPU benchmark registry (Stage 7B)

A benchmark/recording layer (NOT a solver) that runs the AD mainline solver
on a model spec on CPU and (opt-in) a V100/TITAN V GPU, recording the
Stage-7A+ timing contract. Uses `scripts/gpu_selector.py` (V100/TITAN V;
clean-skip; no fallback).

- `RunRecord` — dataclass: model, N, chi, solver, device, device_name, dtype,
  runtime, speedup, final_energy, exact_error, below_ground, gpu_skip_reason.
- `benchmark_model(spec, chi, seed, steps)` → dict with `cpu`/`gpu` records,
  `speedup`, `exact_energy`, `device_info`, `gpu_skip_reason`, `gpu_ran`.
- The GPU is NOT required to be faster; speedup is recorded only.

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
  `.generate_heisenberg(J=1.0)` / `.generate_tfi(...)` /
  `.generate_spinless_fermion(t=1.0, V=0.0, mu=0.0)` (Stage 7A, bond dim 6,
  JW parity-carrying virtual state) /
  `.generate_hubbard(t=1.0, U=4.0, mu=0.0, h=0.0)` (Stage 7C, bond dim 6, d=4,
  no separate parity-carrying state — the inter-site parity cancels in the
  product; the surviving site-`i` parity is absorbed into the `@P`/`P@` left
  factors of each spin-resolved hop) to populate tensors.
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

### Spinless fermion observables (Stage 7A)

Dense-reference fermion observables for the t-V chain (1D JW; the NN hopping
observable carries the JW parity string so it is genuinely fermionic):

- `dense_fermion_local_density(state, site, N)` → `<n_site>`.
- `dense_fermion_density_density(state, i, j, N)` → `<n_i n_j>`.
- `dense_fermion_nn_hopping(state, i, N)` → `<c^d_i c_{i+1} + h.c.>`.
- `mps_fermion_local_density(mps, site)`,
  `mps_fermion_density_density(mps, i, j)`,
  `mps_fermion_nn_hopping(mps, i)` — MPS variants (dense-reference).

### Spinful Hubbard observables (Stage 7C)

Dense-reference Hubbard observables (1D JW; the spin-resolved NN hopping
carries the surviving per-site parity `P` at the left-factor site so it is
genuinely fermionic):

- `dense_hubbard_local_density(state, site, N, spin)` →
  `<n_{site, spin}>` (spin in "up"/"down"/"tot").
- `dense_hubbard_double_occ(state, site, N)` → `<n_up_site n_down_site>`.
- `dense_hubbard_local_sz(state, site, N)` → `<S^z_site>`.
- `dense_hubbard_nn_hopping(state, i, N, spin)` →
  `<c^d_{i,s} c_{i+1,s} + h.c.>` (spin in "up"/"down").
- `mps_hubbard_local_density(mps, site, spin=...)`,
  `mps_hubbard_double_occ(mps, site)`, `mps_hubbard_local_sz(mps, site)`,
  `mps_hubbard_nn_hopping(mps, i, spin=...)` — MPS variants (dense-reference).

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

## `latticetn.tdvp` — traditional real-time TDVP (Stage 12B)  ⚠️ CLASSICAL BASELINE

Matrix-free projector-splitting time evolution on the finite open-boundary MPS
manifold. This is deliberately not AD-TDVP.

- `TDVP(mps, mpo, dt, method="one_site"|"two_site", device=...,
  krylov_dim=..., max_bond_dim=..., truncation_tol=...)` — configure the
  time-independent Hamiltonian evolution.
- `TDVP.evolve(steps, observables={name: callback})` → `TDVPResult` with the
  evolved MPS, time/energy/norm histories, callback histories, and two-site
  truncation/bond diagnostics.
- `tdvp.effective_hamiltonian` — native one-site, zero-site, and two-site MPO
  environment actions; no dense many-body Hamiltonian.
- `tdvp.krylov.lanczos_expm_action` — Hermitian matrix-free action of
  `exp(-i dt H_eff)` with full reorthogonalization and CPU/CUDA device parity.
- One-site keeps bond dimensions fixed. Two-site uses adaptive SVD ranks up to
  `max_bond_dim` and records relative discarded weights.

See [`STAGE12B_TDVP_REPORT.md`](STAGE12B_TDVP_REPORT.md) and
`scripts/run_tdvp_heisenberg_quench.py`.

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

## Benchmark / score scripts (Stage 6A / 7A)

- `scripts/run_ad_gpu_benchmark.py` — CPU/GPU benchmark of the three AD
  mainline solvers (global AD-MPS, one-site AD, two-site AD) with exact / DMRG
  reference baselines. GPU is opt-in (`LATTICETN_RUN_GPU=1`, uses `cuda:0`;
  clean-skips if CUDA unavailable). Writes `docs/AD_GPU_BENCHMARK_REPORT.md`
  + JSON results. No AD loss path is modified; it only calls `train_ad_*` with
  device-placed MPS/MPO.
- `scripts/ad_gpu_benchmark_score.py` — Stage 6A score: runs the three
  benchmark test files, regenerates the report, checks required terms.
- `scripts/gpu_selector.py` — unified GPU selector (Stage 7A onward). Selects
  a GPU whose name contains `V100` or `TITAN V`/`Titan V`; clean-skips (no
  fallback) otherwise. Opt-in via `LATTICETN_RUN_GPU=1`.
- `scripts/run_spinless_fermion_benchmark.py` — Stage 7A CPU/GPU benchmark of
  the three AD mainline solvers on the spinless fermion t-V chain (JW), with
  ED reference. GPU opt-in via `LATTICETN_RUN_GPU=1` + unified selector.
  Writes `docs/FERMION_REPORT.md` + JSON results.
- `scripts/fermion_score.py` — Stage 7A score: runs the six fermion test
  files, regenerates the report, checks required terms.
- `scripts/run_model_builder_benchmark.py` — Stage 7B unified benchmark
  registry across registered model presets (Heisenberg, spinless fermion
  t-V) on CPU and (opt-in) a V100/TITAN V GPU. Writes
  `docs/MODEL_BUILDER_REPORT.md` + JSON results.
- `scripts/model_builder_score.py` — Stage 7B score: runs the five
  model-builder test files, regenerates the report, checks required terms.

---

For usage flows, see [`USER_GUIDE.md`](USER_GUIDE.md) (and the Chinese
[`USER_GUIDE.zh-CN.md`](USER_GUIDE.zh-CN.md)). For step-by-step runnable
walkthroughs, see the bilingual tutorials under
[`tutorials/`](tutorials/) / [`tutorials.zh-CN/`](tutorials.zh-CN/). For the
standing policy on what is mainline vs. baseline, see
[`AD_MAINLINE_POLICY.md`](AD_MAINLINE_POLICY.md).
