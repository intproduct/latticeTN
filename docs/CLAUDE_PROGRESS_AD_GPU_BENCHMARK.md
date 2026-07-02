# CLAUDE Progress — Stage 6A CPU/GPU AD Solver Benchmark

## Goal

Build a unified, opt-in CPU/GPU benchmark of the three AD mainline solvers of
latticeTN (global AD-MPS, one-site AD local, two-site AD local) on the
open-boundary 1D spin-1/2 Heisenberg chain, comparing CPU and the machine's
single GPU for numerical parity and runtime/speedup. DMRG / Lanczos / exact
diagonalization are reference baselines only, never the AD mainline. GPU is
opt-in (`LATTICETN_RUN_GPU=1`, uses `cuda:0`; clean-skips if CUDA
unavailable). Small, controlled `--fast` preset only — no long/large
benchmark. No new algorithm; no AD loss path modified.

## Files added / modified this stage

Added:
- `scripts/run_ad_gpu_benchmark.py` — CPU/GPU benchmark runner + report generator.
- `scripts/ad_gpu_benchmark_score.py` — Stage 6A score script.
- `tests/test_ad_gpu_benchmark_config.py` — config + device-selection (CPU, always).
- `tests/test_ad_gpu_benchmark_smoke.py` — runs the benchmark; GPU parity opt-in.
- `tests/test_ad_gpu_benchmark_report.py` — score `--list` subprocess test.
- `docs/AD_GPU_BENCHMARK_SPEC.md`
- `docs/AD_GPU_BENCHMARK_PROTOCOL.md`
- `docs/AD_GPU_BENCHMARK_REPORT.md` (generated)
- `docs/CLAUDE_PROGRESS_AD_GPU_BENCHMARK.md`

Modified (top-level docs only; no library / loss-path changes):
- `README.md` — Stage 6A row + Quick validation GPU command.
- `ROADMAP.md` — Stage 5C marked DONE (shipped as Stage 6A).
- `REPO_STATUS.md` — passed scores + "not yet done" updated.
- `docs/USER_GUIDE.md` — validation table + CPU/GPU notes.
- `docs/API_OVERVIEW.md` — `ad_two_site` entry (was missing) + benchmark script section.
- `docs/INDEX.md` — Stage 6A row.

No changes to `latticetn/ad_variational.py`, `ad_local.py`, `ad_two_site.py`,
`contractions.py`, `mps.py`, `mpo.py`, `operators.py`, or `dmrg.py`. The
benchmark only CALLS the existing `train_ad_*` functions with device-placed
MPS/MPO; the AD modules inherit the device from the MPS they wrap.

## Commands run and results

_Pre-check_: `sc` conda env had `torch 2.12.0+cu126`, CUDA available,
`device_count=1`, GPU name `NVIDIA TITAN V`. `pytest` was missing from the env
and was installed from `requirements-dev.txt` (`pytest>=7.4`,
`pytest-timeout>=2.2` — within the CLAUDE.md-allowed dependency set); the Bash
safety-classifier was intermittently unavailable for `pip install`, so the
install was completed via a `!`-prefixed command in the session prompt.

Direct runner verification (no pytest needed; `scripts/run_ad_gpu_benchmark.py`
is a plain script):

- CPU-only: `CUDA_VISIBLE_DEVICES= python scripts/run_ad_gpu_benchmark.py --fast`
  → `ad gpu benchmark: pass=True gpu_ran=False` (exit 0); report regenerated
  at `docs/AD_GPU_BENCHMARK_REPORT.md` with a clean GPU-skip note.
- GPU opt-in: `LATTICETN_RUN_GPU=1 python scripts/run_ad_gpu_benchmark.py --fast`
  → `ad gpu benchmark: pass=True gpu_ran=True` (exit 0); GPU = `NVIDIA TITAN V`
  on `cuda:0`, CUDA 12.6, torch 2.12.0+cu126.
- `python scripts/ad_gpu_benchmark_score.py --list` → exit 0, lists all required
  tests / runner / docs (so `tests/test_ad_gpu_benchmark_report.py` passes).
- GPU `torch.linalg.qr` / `svd` on `complex128` at `cuda:0` verified finite and
  on-device (one-site/two-site sweeps' QR/SVD run correctly on GPU).

Existing regressions (branch `stage6a-gpu-ad-benchmark`, all `--fast`, exit 0 —
stop condition 1):
- `python scripts/validation_score.py --fast` → PASS
- `python scripts/benchmark_score.py --fast` → PASS
- `python scripts/canonical_score.py --fast` → PASS
- `python scripts/contraction_score.py --fast` → PASS
- `python scripts/ad_variational_score.py --fast` → PASS
- `python scripts/ad_local_opt_score.py --fast` → PASS
- `python scripts/ad_two_site_score.py --fast` → PASS

Stage 6A formal score runs (stop conditions 2 & 3):
- `python scripts/ad_gpu_benchmark_score.py --fast` (CPU-only) →
  `AD GPU benchmark score: PASS` (exit 0); 18 tests, 16 passed / 2 clean-skipped
  (the complementary GPU parity / clean-skip pair).
- `LATTICETN_RUN_GPU=1 python scripts/ad_gpu_benchmark_score.py --fast` (GPU) →
  `AD GPU benchmark score: PASS` (exit 0), `gpu_ran=True`; the GPU parity tests
  ran and passed (CPU/GPU energies agree to machine epsilon).

Default `pytest -q` collects 233 tests (Stage 5B was 215; Stage 6A adds the
three benchmark test files).

## Key results (direct runner runs, seed 0)

CPU-only run (`--fast`, `CUDA_VISIBLE_DEVICES=`):

| N | chi | solver | final E | energy error | runtime_s | below ground |
|---:|---:|---|---:|---:|---:|:---:|
| 4 | 4 | global AD-MPS (Adam) | -1.6160223657 | 3.04e-06 | 5.34 | False |
| 4 | 4 | one-site AD local (LBFGS) | -1.6160254037 | 4.19e-11 | 0.39 | False |
| 4 | 4 | two-site AD local (LBFGS) | -1.6160254036 | 1.82e-10 | 0.12 | False |
| 6 | 8 | global AD-MPS (Adam) | -2.4934815147 | 9.56e-05 | 7.22 | False |
| 6 | 8 | one-site AD local (LBFGS) | -2.4935771330 | 8.91e-10 | 0.96 | False |
| 6 | 8 | two-site AD local (LBFGS) | -2.4935771330 | 8.39e-10 | 0.29 | False |

exact E0: N=4 → -1.6160254038; N=6 → -2.4935771339. DMRG reference matches
exact to ~1e-10 on both N. One-site / two-site AD reach machine precision;
global AD-MPS (first-order Adam, short steps) is within its existing tolerance
envelope (N=4 ≈ 3e-6; N=6 ≈ 1e-4, well under `AD_TOL[6]=1e-3`).

GPU run (`LATTICETN_RUN_GPU=1`, `cuda:0`, NVIDIA TITAN V) — CPU/GPU parity:

| N | solver | CPU E | GPU E | \|CPU-GPU\| | CPU rt (s) | GPU rt (s) | speedup | GPU below ground |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 4 | global AD-MPS | -1.6160223657 | -1.6160223657 | 0.00e+00 | 5.34 | 6.93 | 0.77 | False |
| 4 | one-site AD | -1.6160254037 | -1.6160254037 | 4.44e-16 | 0.39 | 0.90 | 0.43 | False |
| 4 | two-site AD | -1.6160254036 | -1.6160254036 | 2.22e-16 | 0.12 | 0.33 | 0.36 | False |
| 6 | global AD-MPS | -2.4934815147 | -2.4934815147 | 8.88e-16 | 7.22 | 15.32 | 0.47 | False |
| 6 | one-site AD | -2.4935771330 | -2.4935771330 | 4.44e-16 | 0.96 | 1.96 | 0.49 | False |
| 6 | two-site AD | -2.4935771330 | -2.4935771330 | 0.00e+00 | 0.29 | 0.72 | 0.40 | False |

All `|CPU-GPU|` energy diffs are at machine epsilon (≪ `ENERGY_AGREE_TOL`);
no below-ground on either device. **The GPU is slower than the CPU for every
case** (speedup 0.36–0.77×), exactly as expected for these tiny systems —
host↔device transfer and kernel-launch overhead dominate the very short sweeps.
This is recorded honestly in the report; the benchmark contract does not
require the GPU to be faster.

## Design notes / assumptions

- **Single-GPU target.** Stage 2.5's GPU smoke was written for multi-GPU boxes
  and refuses to fall back to an unmatched GPU (name filter
  `LATTICETN_GPU_NAME_FILTER`). This stage's contract is "the current machine's
  one GPU", so name filtering would be vacuous; the runner uses `cuda:0` when
  `LATTICETN_RUN_GPU=1` and CUDA is available. This is a deliberate, documented
  deviation from the Stage 2.5 device-selection rules, scoped to this stage
  only (Stage 2.5's `run_gpu_smoke.py` is unchanged).
- **No AD loss path touched.** The runner constructs MPS/MPO with
  `device="cpu"` or `device="cuda:0"` and calls `train_ad_mps` /
  `train_ad_local` / `train_ad_two_site` as-is. The AD modules store
  `self.device = mps.device` and route all ops through the placed tensors, so
  they are device-agnostic by construction.
- **Apples-to-apples start.** CPU and GPU builds use the same seed; the GPU
  MPS is seeded on CPU then copied by value onto the GPU so both start from
  identical tensors (not different RNG draws).
- **ED / DMRG are CPU-only reference baselines.** ED calls
  `numpy.linalg.eigh`; DMRG runs under `no_grad` on CPU. The GPU runs only the
  AD solver optimization.
- **Tolerances not widened.** `ENERGY_AGREE_TOL = {4: 1e-6, 6: 1e-5}` matches
  the existing AD tolerances; `BELOW_GROUND_TOL = 1e-6`.
- **Speedup is recorded but not asserted.** Small systems (N=4/6, chi=4/8,
  short sweeps) are overhead-dominated; the GPU may be slower and that is
  acceptable. The contract is numerical parity + an honest runtime number.

## Next-step suggestions

- A larger-N opt-in GPU benchmark (N=8/10/12, larger chi) once GPU correctness
  parity is confirmed here — to actually measure where GPU pays off. Out of
  scope for Stage 6A's `--fast` preset.
- If the GPU `tc.linalg.qr`/`svd` path on complex128 ever proves a bottleneck
  for the local/two-site sweeps, a future stage could add a `device`-aware
  fallback; not needed for the small `--fast` cases here.
