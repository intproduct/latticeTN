# Stage 7B — General 1D Model Builder + Unified Benchmark Registry Spec

## Scope

Stage 7B abstracts the existing Heisenberg and spinless-fermion t-V
Hamiltonians behind a **unified 1D model-construction layer**
(`latticetn/model_builder.py`) and adds a **unified CPU/GPU benchmark
registry** (`latticetn/benchmarking.py`). This is a **model/MPO construction
layer, NOT a new solver** — the AD mainline (differentiable Rayleigh quotient
+ autograd + torch optimizer) is unchanged; SVD/QR/canonicalization remain
auxiliary stabilization; exact/DMRG/Lanczos remain reference baselines. The
layer prepares the ground for future Hubbard / TDVP / finite-temperature work
without touching the loss path.

## What is added (and what is NOT)

Added (Stage 7B):
- `latticetn/model_builder.py` — `ModelSpec`, term types (`OnsiteTerm`,
  `TwoSiteTerm` for bosonic/spin, `FermionHopTerm` and `DensityDensityTerm`
  for fermionic/JW), presets (`heisenberg_model`, `spinless_fermion_tv_model`),
  and builders `build_dense(spec)` / `build_mpo(spec)`.
- `latticetn/benchmarking.py` — `benchmark_model(spec, chi, seed, steps)` →
  a unified record (model, N, chi, solver, device, device_name, dtype,
  runtime, speedup, final_energy, exact_error, below_ground, gpu_skip_reason)
  on CPU and (opt-in) a V100/TITAN V GPU via `scripts/gpu_selector.py`.
- `scripts/run_model_builder_benchmark.py` + `scripts/model_builder_score.py`.
- Tests + docs (this file, MODEL_BUILDER_PROTOCOL, MODEL_BUILDER_REPORT,
  CLAUDE_PROGRESS_MODEL_BUILDER, GPU_TESTING_PROTOCOL update).

NOT added (hard constraints):
- No TDVP, no finite-temperature, no Hubbard, no graded fermionic tensors.
- No new solver; no change to the AD loss path / Heisenberg / fermion
  conventions / existing thresholds. No new large dependencies.

## Model spec

A `ModelSpec` describes a 1D open-boundary chain as a list of terms with an
explicit boson/fermion `statistics` field (the two never mix in one spec):

- `OnsiteTerm(op, coeff)` — `coeff * sum_i op_i` (no JW string even for
  fermions; a single-site operator commutes with the parity string).
- `TwoSiteTerm(op_i, op_j, coeff)` — bosonic/spin
  `coeff * sum_i op_i op_{i+1}` (NO JW string).
- `FermionHopTerm(coeff)` — `-coeff * sum_i (c^d_i c_{i+1} + h.c.)`, JW
  fermionic; carries the parity string `F^i` on sites 0..i-1 (NOT
  hard-core-boson).
- `DensityDensityTerm(coeff, op)` — diagonal `coeff * sum_i op_i op_{i+1}`
  (default `op = n - 1/2`); no JW string.

Presets (Stage 7B ships two):
- `heisenberg_model(N, J)` — `H = J * sum_i S_i.S_{i+1}`, `S = sigma/2`,
  boson.
- `spinless_fermion_tv_model(N, t, V, mu)` — the Stage 7A t-V chain, fermion.

## Builders

- `build_dense(spec)` — dense `(2**N, 2**N)` Hamiltonian. Stage 7B dispatches
  to the existing validated references (`operators.heisenberg_dense` /
  `operators.spinless_fermion_dense`) so the physics is byte-identical to
  Stage 1/7A. A future stage may add a generic term-by-term assembler.
- `build_mpo(spec)` — MPO. Dispatches to `MPO.generate_heisenberg` /
  `MPO.generate_spinless_fermion`. `to_dense` matches `build_dense` (see
  `test_model_builder_mpo_dense`). For fermions the JW parity string is
  carried by the MPO's parity-carrying virtual state.

## Benchmark registry

`benchmark_model(spec, chi, seed, steps)` returns a record dict with the
Stage-7A+ timing contract on CPU and (opt-in) a V100/TITAN V GPU. It uses the
**unified GPU selector** (`scripts/gpu_selector.py`): name-match `V100`/
`TITAN V`/`Titan V`, no fallback, clean-skip otherwise. The GPU is NOT
required to be faster; speedup is recorded only to observe AD-TN GPU
acceleration trends.

## Conventions

Open boundary, `d=2`, default `torch.complex128`. Heisenberg uses
`S = sigma/2` (NOT Pauli); spinless fermion uses basis `|0>`/`|1>`, JW. The
two statistics never mix in one spec. ED is CPU-only reference baseline.
