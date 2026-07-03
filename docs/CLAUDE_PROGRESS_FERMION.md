# Stage 7A Progress Log — Spinless Fermion t-V Chain (Jordan-Wigner)

## Goal

Add 1D open-boundary spinless fermion t-V chain support to `latticeTN` on top
of the unchanged AD mainline, and unify GPU timing (V100/TITAN V only) from
Stage 7A onward. No TDVP / finite-temperature / Hubbard / graded fermionic
tensors.

## Changes (files)

New files:
- `latticetn/fermion_operators.py` — `fermion_operators(dtype, device)` →
  `{I, c, cdag, n, F=(-1)^n, n_minus_half}`. JW parity string documented.
- `scripts/gpu_selector.py` — unified GPU selector. Picks a GPU whose name
  contains `V100` or `TITAN V`/`Titan V`; clean-skips otherwise (no fallback
  to `cuda:0`). Used by the fermion benchmark and tests.
- `scripts/run_spinless_fermion_benchmark.py` — CPU/GPU benchmark of the
  three AD mainline solvers on the fermion t-V chain, with ED reference.
- `scripts/fermion_score.py` — Stage 7A score: runs the 6 test files,
  regenerates `docs/FERMION_REPORT.md`, checks required terms.
- `tests/test_fermion_operators.py`
- `tests/test_spinless_fermion_dense.py`
- `tests/test_spinless_fermion_mpo_dense.py`
- `tests/test_spinless_fermion_native_energy.py`
- `tests/test_spinless_fermion_ad_solvers.py`
- `tests/test_fermion_gpu_timing.py`
- `docs/FERMION_SPEC.md`, `docs/FERMION_PROTOCOL.md`, `docs/FERMION_REPORT.md`,
  this file.

Modified files:
- `latticetn/operators.py` — added `spinless_fermion_dense(N,t,V,mu,dtype,
  device)` with the explicit global JW parity string; helpers `_global_single`,
  `_global_two_diag`.
- `latticetn/mpo.py` — added `MPO.generate_spinless_fermion(t,V,mu)`, bond
  dim 6, JW parity-carrying virtual state; `to_dense` matches the dense ref.
- `latticetn/observables.py` — fermion observables: `dense_fermion_local_density`,
  `dense_fermion_density_density`, `dense_fermion_nn_hopping`, and MPS
  variants.
- `latticetn/__init__.py` — export `fermion_operators`, `spinless_fermion_dense`.
- `docs/GPU_TESTING_PROTOCOL.md`, `docs/USER_GUIDE.md`, `docs/USER_GUIDE.zh-CN.md`,
  `docs/API_OVERVIEW.md`, `docs/INDEX.md`, `ROADMAP.md`, `REPO_STATUS.md`.

## Key design points

- JW parity string `F = (-1)^n = diag(1,-1)` is the key. The dense reference
  builds global `c_i = F^i x c x I...`; the MPO routes every hop through a
  parity-carrying virtual state so the same `F^i` string is emitted on sites
  `0..i-1`. The MPO bond dim is 6.
- Critical MPO detail: the parity-start transition `0->1: F` lives ONLY in
  the left-boundary (site-0) tensor. If it were in the bulk, a parity string
  could start at any site k, producing a spurious partial-parity hop
  `F_k..F_{i-1} c_i c_{i+1}` (missing `F_0..F_{k-1}`). Bond 0 starts its hop
  directly (`0->2: c^d`) because `F^0 = I`.
- The on-site chemical potential `-mu(n-1/2)` and the interaction
  `V(n-1/2)(n-1/2)` are diagonal, so no JW string is needed for them.
- AD loss path (`contractions.rayleigh_energy_native`) is UNCHANGED — it is
  operator-agnostic. Only the Hamiltonian/MPO/operator layer is new.
- Unified GPU selector: name-match `V100`/`TITAN V`/`Titan V`
  (case-insensitive), nvidia-smi preferred with torch fallback, re-matches
  the name to resolve the torch logical index. Never falls back to a
  non-matching GPU.

## Commands + results

Pre-Stage-7A core regressions (all PASS, CPU-only):
- `python scripts/validation_score.py --fast` → PASS
- `python scripts/benchmark_score.py --fast` → PASS
- `python scripts/contraction_score.py --fast` → PASS
- `python scripts/ad_variational_score.py --fast` → PASS
- `python scripts/ad_local_opt_score.py --fast` → PASS
- `python scripts/ad_two_site_score.py --fast` → PASS
- `python scripts/ad_gpu_benchmark_score.py --fast` → PASS (CPU-only)

Stage 7A:
- `python scripts/fermion_score.py --fast` → PASS (CPU-only; GPU clean-skip)
- `LATTICETN_RUN_GPU=1 python scripts/fermion_score.py --fast` → PASS
  (gpu_ran=True; Tesla V100-SXM2-16GB selected; CPU/GPU energy parity within
  tolerance; runtime/speedup recorded; not below ground)

Sanity (free fermion, V=0, mu=0): dense E0 matches the single-particle
open-chain formula `-2t cos(k pi/(N+1))` summed over negative modes for
N=2..6. Observables self-consistent: reconstructed E0 from `<hop>` +
`<n_i n_{i+1}>` matches ED E0 exactly.

## Items not run

- No long benchmarks (only N=4/6, chi=4/8 `--fast`).
- No TDVP / finite-temperature / Hubbard / graded fermionic tensors (out of
  scope by hard constraints).
- The Stage 6A AD GPU benchmark selector (`run_ad_gpu_benchmark.py`) is left
  as-is (it predates Stage 7A and uses `cuda:0`); the new unified selector
  lives in `scripts/gpu_selector.py` and is the canonical selector from
  Stage 7A onward.

## Next action

Stage 7A is complete. Suggested commit:
`Add spinless fermion support with GPU timing`
(Do NOT commit per the Stage 7A git constraints; report only.)
