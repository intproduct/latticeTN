# Stage 7B Model Builder Progress Log

## Goal

Abstract the existing Heisenberg and spinless-fermion t-V Hamiltonians behind
a unified 1D model-construction interface, and add a unified CPU/GPU
benchmark registry. AD mainline unchanged. No TDVP / finite-temperature /
Hubbard / graded fermionic tensors. Prepared for future Hubbard / TDVP /
finite-temperature work without touching the loss path.

## Changes (files)

New files:
- `latticetn/model_builder.py` — `ModelSpec`, term types (`OnsiteTerm`,
  `TwoSiteTerm` for bosonic/spin; `FermionHopTerm`, `DensityDensityTerm` for
  fermionic/JW), presets (`heisenberg_model`, `spinless_fermion_tv_model`),
  `build_dense(spec)` / `build_mpo(spec)`.
- `latticetn/benchmarking.py` — `benchmark_model(spec, chi, seed, steps)` →
  unified CPU/GPU record (model, N, chi, solver, device, device_name, dtype,
  runtime, speedup, final_energy, exact_error, below_ground, gpu_skip_reason)
  via `scripts/gpu_selector.py`.
- `scripts/run_model_builder_benchmark.py` — runs the registry over both
  presets, writes `docs/MODEL_BUILDER_REPORT.md` + JSON.
- `scripts/model_builder_score.py` — Stage 7B score: runs the 5 test files,
  regenerates the report, checks required terms.
- `tests/test_model_builder_heisenberg.py`
- `tests/test_model_builder_fermion.py`
- `tests/test_model_builder_mpo_dense.py`
- `tests/test_benchmark_registry.py`
- `tests/test_stage7b_score.py`
- `docs/MODEL_BUILDER_SPEC.md`, `docs/MODEL_BUILDER_PROTOCOL.md`,
  `docs/MODEL_BUILDER_REPORT.md`, this file.

Modified files:
- `latticetn/__init__.py` — export `model_builder`.
- `docs/GPU_TESTING_PROTOCOL.md`, `docs/USER_GUIDE.md`,
  `docs/USER_GUIDE.zh-CN.md`, `docs/API_OVERVIEW.md`, `docs/INDEX.md`,
  `ROADMAP.md`, `REPO_STATUS.md`.

## Key design points

- Stage 7B is a **model/MPO construction layer, NOT a new solver**. The AD
  mainline (`contractions.rayleigh_energy_native` + `train_ad_*`) is
  operator-agnostic and unchanged. Only the Hamiltonian/MPO construction is
  abstracted.
- `build_dense`/`build_mpo` **dispatch** to the existing validated generators
  (`operators.heisenberg_dense` / `operators.spinless_fermion_dense` /
  `MPO.generate_heisenberg` / `MPO.generate_spinless_fermion`), so the
  physics is byte-identical to Stage 1/7A. A future stage may add a generic
  term-by-term assembler; the dispatch keeps the JW parity string exact for
  fermions (no hard-core-boson degradation).
- The `ModelSpec.statistics` field ("boson"/"fermion") makes the
  boson/fermion distinction explicit; the two never mix in one spec. Fermion
  terms (`FermionHopTerm`) carry the JW string; boson terms (`TwoSiteTerm`)
  do not.
- The unified benchmark registry reuses the Stage 7A `scripts/gpu_selector.py`
  (V100/TITAN V; clean-skip; no fallback). It records the full Stage-7A+
  timing contract and is reusable for any future preset.

## Commands + results

Pre-Stage-7B core regressions (all PASS):
- `validation_score --fast`, `benchmark_score --fast`, `contraction_score --fast`,
  `ad_variational_score --fast`, `ad_local_opt_score --fast`,
  `ad_two_site_score --fast`, `fermion_score --fast` (CPU),
  `LATTICETN_RUN_GPU=1 fermion_score --fast` (GPU, V100) — all PASS.

Stage 7B:
- `python scripts/model_builder_score.py --fast` → PASS (CPU-only; GPU
  clean-skip).
- `LATTICETN_RUN_GPU=1 python scripts/model_builder_score.py --fast` → PASS
  (gpu_ran=True; Tesla V100-SXM2-16GB; CPU/GPU energy parity; runtime/speedup
  recorded; not below ground).

Alignment sanity: `build_dense(spec)` == existing dense reference (0 diff);
`build_mpo(spec).to_dense()` == `build_dense(spec)` (≤ 1e-12); native
Rayleigh energy == dense energy (≤ 1e-9) for both presets.

## Items not run

- No long benchmarks (only N=4/6, chi=4/8 `--fast`).
- No Hubbard / XXZ / TFI / long-range MPO (future work; the dispatch
  architecture supports adding presets without touching the AD loss path).
- No TDVP / finite-temperature / graded fermionic tensors (out of scope by
  hard constraints).

## Next action

Stage 7B is complete. Suggested commit:
`Add general 1D model builder and benchmark registry`
(Do NOT commit per the Stage 7B git constraints; report only.)
