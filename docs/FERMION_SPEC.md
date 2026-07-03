# Stage 7A — Spinless Fermion t-V Chain (Jordan-Wigner) Spec

## Scope

Stage 7A adds the open-boundary 1D **spinless fermion t-V chain** to
`latticeTN`, on top of the unchanged AD mainline. The model is:

```
H = -t * sum_i (c^d_i c_{i+1} + c^d_{i+1} c_i)
    + V * sum_i (n_i - 1/2)(n_{i+1} - 1/2)
    - mu * sum_i (n_i - 1/2)
```

with local basis `|0>` (empty), `|1>` (occupied), `d=2`, default
`torch.complex128`, open boundary.

**This is 1D Jordan-Wigner (JW) fermions, NOT a full graded fermionic tensor
network.** Fermionic operators are represented as tensor products of bosonic
2x2 matrices with an explicit JW parity string `F = (-1)^n`. The JW parity
string is the key object; without it the Hamiltonian would be a hard-core-boson
Hamiltonian (wrong signs for hopping beyond the first bond).

## What is added (and what is NOT)

Added (Stage 7A):
- `latticetn/fermion_operators.py` — local fermion operators `I, c, c^d, n,
  F=(-1)^n, n-1/2`.
- `latticetn/operators.py::spinless_fermion_dense` — dense reference
  Hamiltonian with the explicit global JW string.
- `latticetn/mpo.py::MPO.generate_spinless_fermion` — bond-dim-6 fermionic
  MPO with a JW parity-carrying virtual state; `to_dense` matches
  `spinless_fermion_dense`.
- `latticetn/observables.py` — fermion observables: local density `<n_i>`,
  density-density `<n_i n_j>`, NN hopping `<c^d_i c_{i+1}+h.c.>`.
- `scripts/gpu_selector.py` — unified GPU selector (V100/TITAN V only).
- `scripts/run_spinless_fermion_benchmark.py` + `scripts/fermion_score.py`.
- Tests + docs (this file, FERMION_PROTOCOL, FERMION_REPORT,
  CLAUDE_PROGRESS_FERMION, GPU_TESTING_PROTOCOL update).

NOT added (hard constraints):
- No TDVP, no finite-temperature, no Hubbard, no graded fermionic tensors.
- No change to the AD mainline / loss path (it is operator-agnostic; only the
  Hamiltonian/MPO/operator layer is new).
- No change to Heisenberg conventions or existing thresholds.
- No new large dependencies (torch/numpy/scipy/pytest only).

## Physics conventions

- Spinless fermion, `d=2`, `|0>`/`|1>`, `torch.complex128`, open boundary.
- `c |1> = |0>`, `c |0> = 0`; `c^d |0> = |1>`, `c^d |1> = 0`;
  `n = c^d c = diag(0,1)`; `F = (-1)^n = diag(1,-1)`.
- Global operators carry the JW parity string on all sites left of the
  operator: `c_i = F^{i} x c x I...`. Nearest-neighbor hopping
  `c^d_i c_{i+1}` reduces (after JW cancellation between the two sites) to
  `F^{i} x c^d x c x I...`.
- The dense reference and the MPO both build the global operators with this
  string; they agree to `1e-12` (see `test_spinless_fermion_mpo_dense.py`).
- Exact diagonalization (`numpy.linalg.eigh`) is the reference baseline;
  DMRG remains available as a classical baseline. Neither is in the AD path.

## AD mainline (unchanged)

The three AD mainline solvers run on the fermion MPO unchanged:
- global AD-MPS (`ad_variational.train_ad_mps`, Adam);
- one-site AD local (`ad_local.train_ad_local`, LBFGS);
- two-site AD local (`ad_two_site.train_ad_two_site`, LBFGS).

The differentiable loss is still `contractions.rayleigh_energy_native` —
operator-agnostic. Stage 7A only swaps the Hamiltonian/MPO.

## GPU rules (Stage 7A onward)

- A **unified GPU selector** (`scripts/gpu_selector.py`) selects a GPU whose
  name contains `V100` or `TITAN V`/`Titan V`. No other GPU is used; there is
  no fallback to `cuda:0`.
- GPU is opt-in via `LATTICETN_RUN_GPU=1`.
- If no matching GPU is present (or GPU not opted in), the GPU portion
  **clean-skips** (exit 0; CPU results still recorded).
- If a matching GPU is present, Stage 7A runs CPU and GPU small-system tests
  and records: device name, dtype, N, chi, solver, final energy, exact error,
  runtime, speedup, below_ground.
- dtype default `torch.complex128`. The GPU is NOT required to be faster;
  speedup is recorded only to observe AD-TN GPU acceleration trends.
