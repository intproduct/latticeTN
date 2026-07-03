# Stage 7B — General 1D Model Builder + Benchmark Registry Protocol

> Operational protocol and validation steps for
> `scripts/model_builder_score.py` and
> `scripts/run_model_builder_benchmark.py`. Stage 7B is a **model/MPO
> construction layer, NOT a new solver**. The AD mainline is unchanged; exact
> /DMRG/Lanczos remain reference baselines.

## 1. Scope

Provide a unified 1D model-construction interface (`ModelSpec` + term types +
presets + `build_dense`/`build_mpo`) and a unified CPU/GPU benchmark registry
that records the Stage-7A+ timing contract across the registered presets.
Default runs are CPU-only, `torch.complex128`, small systems (N=4,6) for
`--fast`. GPU is opt-in via `LATTICETN_RUN_GPU=1` and uses the unified
`scripts/gpu_selector.py` (V100/TITAN V; clean-skip otherwise).

## 2. Required files

- Modules: `latticetn/model_builder.py`, `latticetn/benchmarking.py`
- Runner: `scripts/run_model_builder_benchmark.py`
- Score: `scripts/model_builder_score.py`
- GPU selector: `scripts/gpu_selector.py` (Stage 7A)
- Docs: `docs/MODEL_BUILDER_SPEC.md`, `docs/MODEL_BUILDER_PROTOCOL.md`,
  `docs/MODEL_BUILDER_REPORT.md`, `docs/CLAUDE_PROGRESS_MODEL_BUILDER.md`,
  `docs/GPU_TESTING_PROTOCOL.md` (updated)
- Tests:
  - `tests/test_model_builder_heisenberg.py`
  - `tests/test_model_builder_fermion.py`
  - `tests/test_model_builder_mpo_dense.py`
  - `tests/test_benchmark_registry.py`
  - `tests/test_stage7b_score.py`

## 3. Commands

```bash
# CPU-only (default; always works, no GPU required):
python scripts/model_builder_score.py --fast

# GPU benchmark (opt-in; runs only if LATTICETN_RUN_GPU=1 AND a V100/TITAN V
# is selected by the unified gpu_selector):
LATTICETN_RUN_GPU=1 python scripts/model_builder_score.py --fast

# List required files:
python scripts/model_builder_score.py --list
```

The score script:
1. runs the five required test files with `pytest -q` (CPU-only env by
   default; GPU visible when opted in);
2. runs `scripts/run_model_builder_benchmark.py --markdown-output
   docs/MODEL_BUILDER_REPORT.md --json-output <path>` to regenerate the
   report and JSON results;
3. checks the report contains all required sections/terms;
4. prints `Model builder score: PASS` and exits 0 on success.

A clean GPU skip is a **successful exit 0**: the CPU benchmark still runs and
the report records the skip reason.

## 4. Test coverage (what is checked)

- **Heisenberg preset** (`test_model_builder_heisenberg.py`): dense matches
  `heisenberg_dense`; J-scaling; statistics is boson; ground energy matches
  ED; terms decompose to `S.S = Sz Sz + (1/2)(S+S- + S-S+)`; native Rayleigh
  energy matches dense.
- **fermion preset** (`test_model_builder_fermion.py`): dense matches
  `spinless_fermion_dense` across N and (t,V,mu); statistics is fermion;
  terms include a `FermionHopTerm` (JW); ground energy matches ED; NOT
  hard-core-boson for N>=3; native Rayleigh matches dense.
- **MPO-dense** (`test_model_builder_mpo_dense.py`): `build_mpo(spec).to_dense`
  matches `build_dense(spec)` for both presets; shapes/bond dims (Heisenberg
  D=5, fermion D=6); Hermitian; unregistered presets raise
  `NotImplementedError`.
- **registry** (`test_benchmark_registry.py`): CPU record has all required
  fields; both presets run; CPU not below ground; GPU clean-skips when not
  opted in; when opted in + V100/TITAN V present, CPU/GPU parity + timing +
  below-ground guard, and the GPU name matches V100/TITAN V.
- **score** (`test_stage7b_score.py`): `--list` exits 0 and lists required
  files; all required files exist.

## 5. Report requirements (`docs/MODEL_BUILDER_REPORT.md`)

Must contain: mainline statement (AD mainline unchanged; Stage 7B is a
model/MPO construction layer, NOT a new solver; ED is reference baseline only;
SVD/QR/canonicalization remain auxiliary stabilization); device info;
CPU/GPU comparison per model (final energy, exact error, runtime, speedup,
below-ground); model-builder coverage table; overall pass/fail; known
limitations.

## 6. Tolerances

- CPU/GPU final-energy agreement: `|E_cpu - E_gpu| < 1e-6` for N=4 and
  `< 1e-5` for N=6 (existing AD tolerances; not widened).
- Final energy must not undershoot the exact ground by more than `1e-6`
  (`below_ground=False`).
- Runtime/speedup must be **recorded**; the GPU is **not** required to be
  faster.

## 7. Physics conventions

Unchanged: open boundary, `d=2`, `torch.complex128`; Heisenberg `S = sigma/2`;
spinless fermion JW. ED is CPU-only. The two statistics never mix in one
spec. No TDVP / finite-temperature / Hubbard / graded fermionic tensors.

## 8. Stop conditions

Stage 7B is complete when:
1. Core scores and `fermion_score.py --fast` still pass.
2. `python scripts/model_builder_score.py --fast` exits 0.
3. Heisenberg and spinless fermion model spec dense/MPO/native energies all
   align.
4. CPU/GPU timing registry works; V100/TITAN V present → GPU parity; else
   clean-skip.
5. Docs and report complete.
6. Final report lists modified files, API, commands, CPU/GPU timing, items
   not run, suggested commit command.

## 9. Hard constraints

- No TDVP, no finite-temperature, no Hubbard, no graded fermionic tensors.
- No new solver; no change to AD loss path / Heisenberg / fermion conventions.
- No widening of existing thresholds; no new large dependencies; no long
  benchmarks; no git mutation commands.
