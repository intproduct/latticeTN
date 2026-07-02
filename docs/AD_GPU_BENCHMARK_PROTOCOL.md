# Stage 6A — CPU/GPU AD Solver Benchmark Protocol

> Operational protocol and validation steps for `scripts/run_ad_gpu_benchmark.py`
> and `scripts/ad_gpu_benchmark_score.py`. The benchmark measures the **AD
> mainline solvers** on CPU and the single GPU. DMRG / Lanczos / exact
> diagonalization are **reference baselines only**.

## 1. Scope

Benchmark the three AD mainline solvers (global AD-MPS, one-site AD, two-site
AD) on the open-boundary 1D spin-1/2 Heisenberg chain, comparing CPU and the
machine's single GPU for numerical parity and runtime/speedup. All default
runs are CPU-only, `torch.complex128`, small systems (N=4,6) for `--fast`.
GPU is opt-in via `LATTICETN_RUN_GPU=1` and uses `cuda:0` when CUDA is
available.

## 2. Required files

- Runner: `scripts/run_ad_gpu_benchmark.py`
- Score: `scripts/ad_gpu_benchmark_score.py`
- Docs: `docs/AD_GPU_BENCHMARK_SPEC.md`, `docs/AD_GPU_BENCHMARK_PROTOCOL.md`,
  `docs/AD_GPU_BENCHMARK_REPORT.md`, `docs/CLAUDE_PROGRESS_AD_GPU_BENCHMARK.md`
- Tests:
  - `tests/test_ad_gpu_benchmark_config.py`
  - `tests/test_ad_gpu_benchmark_smoke.py`
  - `tests/test_ad_gpu_benchmark_report.py`

## 3. Commands

```bash
# CPU-only (default; always works, no GPU required):
python scripts/ad_gpu_benchmark_score.py --fast

# GPU benchmark (opt-in; runs only if LATTICETN_RUN_GPU=1 AND CUDA available):
LATTICETN_RUN_GPU=1 python scripts/ad_gpu_benchmark_score.py --fast

# List required files:
python scripts/ad_gpu_benchmark_score.py --list
```

The score script:
1. runs the three required test files with `pytest -q` (CPU-only env);
2. runs `scripts/run_ad_gpu_benchmark.py --markdown-output
   docs/AD_GPU_BENCHMARK_REPORT.md --json-output <path>` to regenerate the
   report and JSON results;
3. checks the report contains all required sections/terms;
4. prints `AD GPU benchmark score: PASS` and exits 0 on success.

A clean GPU skip (no `LATTICETN_RUN_GPU=1`, or CUDA unavailable) is a
**successful exit 0**: the CPU benchmark still runs and the report records the
skip reason.

## 4. Test coverage (what is checked)

- **config** (`test_ad_gpu_benchmark_config.py`, always runs CPU): the
  benchmark config dataclass parses `--fast` presets; device selection
  respects `LATTICETN_RUN_GPU` (CPU when unset, `cuda:0` when set + CUDA
  available, clean-skip marker when set but CUDA unavailable); the three AD
  solvers are the mainline and DMRG/ED are flagged reference-only.
- **smoke** (`test_ad_gpu_benchmark_smoke.py`): runs the **CPU** benchmark at
  the smallest preset and asserts each AD solver finishes with a finite final
  energy, a recorded runtime, a speedup field, and `below_ground=False`. GPU
  parity assertions clean-skip unless `LATTICETN_RUN_GPU=1` and CUDA is
  available; when they run, CPU/GPU final energies agree within tolerance and
  the GPU energy does not undershoot the exact ground beyond tolerance.
- **report** (`test_ad_gpu_benchmark_report.py`): `ad_gpu_benchmark_score.py
  --list` exits 0 and lists the required tests / runner / docs.

## 5. Report requirements (`docs/AD_GPU_BENCHMARK_REPORT.md`)

Must contain: mainline statement (AD mainline benchmarked; DMRG/Lanczos are
reference baselines only, not the solver); device info (GPU name, CUDA version,
PyTorch version, device, dtype); exact / DMRG reference; CPU/GPU comparison
table per solver (final energy, energy error, energy per bond, runtime,
speedup, below-ground flag); overall pass/fail; known limitations. Must
explicitly state that **the GPU benchmark evaluates the AD mainline solvers**
and that **DMRG/Lanczos are reference baselines only**.

## 6. Tolerances

- CPU/GPU final-energy agreement per AD solver: `|E_cpu - E_gpu| < 1e-6` for
  N=4 and `< 1e-5` for N=6 (matching the existing AD tolerances; not widened).
- GPU (and CPU) final energy must not undershoot the exact ground energy by
  more than `1e-6` (`below_ground=False`).
- Runtime and speedup must be **recorded**; the GPU is **not** required to be
  faster, because small systems are overhead-dominated.

## 7. Physics conventions

Unchanged: `H = J * sum_i S_i.S_{i+1}`, `S = sigma/2`, `J = 1.0`, open
boundary, `torch.complex128`. ED is CPU-only; the GPU runs only the AD solver
optimization. No silent switch between `S` and `sigma`.
