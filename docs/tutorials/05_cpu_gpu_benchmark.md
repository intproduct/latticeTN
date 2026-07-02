# 05 — CPU/GPU AD solver benchmark

Run the unified benchmark that compares the three AD mainline solvers (global
AD-MPS, one-site AD local, two-site AD local) on CPU and the machine's single
GPU, with exact diagonalization and DMRG as **reference baselines only**.

## Goal

- Run the benchmark on CPU (always) and on GPU (opt-in).
- Understand the GPU opt-in rule (`LATTICETN_RUN_GPU=1`, `cuda:0`).
- Read the CPU/GPU energy parity and the speedup column.
- Internalize the key caveat: **the GPU may be slower than CPU on small
  systems** — speedup is a trend observation, not a guarantee.

## Mainline positioning

The benchmark **evaluates the AD mainline solvers**. **DMRG / Lanczos / exact
diagonalization are classical reference baselines ONLY** and are not part of
the AD optimization path; they are reported for context and run on CPU. The
GPU runs only the AD solver optimization.

## The opt-in GPU rule

This machine has a single GPU, so there is **no name filtering** (unlike the
Stage 2.5 multi-GPU smoke). The rule is:

1. GPU is **opt-in**: runs only when `LATTICETN_RUN_GPU=1`.
2. When opted-in AND `torch.cuda.is_available()` AND `device_count() > 0`,
   use `cuda:0`.
3. When opted-in but no visible CUDA device → **clean-skip**: the report
   records the skip and exits 0; the CPU portion still runs.
4. Default runs are CPU-only and never require a GPU.

## Run commands

```bash
# CPU-only (default; always works, no GPU needed):
python scripts/ad_gpu_benchmark_score.py --fast

# GPU opt-in (clean-skips if CUDA unavailable or no visible device):
LATTICETN_RUN_GPU=1 python scripts/ad_gpu_benchmark_score.py --fast

# Just list the required files:
python scripts/ad_gpu_benchmark_score.py --list
```

The `--fast` preset is small on purpose: N=4 (χ=4) and N=6 (χ=8), short
steps/sweeps, so the whole score finishes in seconds on CPU.

## Expected output

The score prints (CPU-only):

```text
$ python -m pytest -q tests/test_ad_gpu_benchmark_config.py tests/test_ad_gpu_benchmark_smoke.py tests/test_ad_gpu_benchmark_report.py
... [ok]
$ python scripts/run_ad_gpu_benchmark.py --markdown-output docs/AD_GPU_BENCHMARK_REPORT.md --json-output docs/ad_gpu_benchmark_fast_results.json
ad gpu benchmark: pass=True gpu_ran=False report -> docs/AD_GPU_BENCHMARK_REPORT.md

AD GPU benchmark score: PASS
```

With `LATTICETN_RUN_GPU=1` on a CUDA machine, `gpu_ran=True` and the report's
CPU/GPU comparison table is populated. CPU reference numbers (from the
`--fast` preset, seed 0):

| N | chi | solver | final E | energy error | runtime_s | below ground |
|---:|---:|---|---:|---:|---:|:---:|
| 4 | 4 | global AD-MPS (Adam) | -1.6160223657 | 3.04e-06 | ~5.5 | False |
| 4 | 4 | one-site AD local (LBFGS) | -1.6160254037 | 4.19e-11 | ~0.3 | False |
| 4 | 4 | two-site AD local (LBFGS) | -1.6160254036 | 1.82e-10 | ~0.1 | False |
| 6 | 8 | global AD-MPS (Adam) | -2.4934815147 | 9.56e-05 | ~7.0 | False |
| 6 | 8 | one-site AD local (LBFGS) | -2.4935771330 | 8.91e-10 | ~0.9 | False |
| 6 | 8 | two-site AD local (LBFGS) | -2.4935771330 | 8.39e-10 | ~0.3 | False |

exact E0: N=4 → -1.6160254038; N=6 → -2.4935771339.

## The key caveat: GPU can be slower on small systems

> "Runtime/speedup are recorded but the GPU is NOT required to be faster: small
> systems are overhead-dominated (host↔device transfer, short sweeps)."

The `--fast` cases are tiny: the two-site N=4 sweep runs in **~0.1 s** on CPU,
well below the host↔device transfer + kernel-launch latency of a GPU. So a
**speedup < 1× (GPU slower than CPU) is expected and acceptable** here — the
benchmark contract is *numerical parity* + an honest runtime number, not a
performance win. Treat the speedup column as a **trend observation**: it
becomes meaningful only at larger N/χ (out of scope for `--fast`).

The hard correctness checks, always:

- CPU/GPU final energies agree within `ENERGY_AGREE_TOL = {4: 1e-6, 6: 1e-5}`
  (in practice they agree to machine epsilon, `~1e-16`).
- Neither device's energy undershoots the exact ground beyond `1e-6`
  (`below_ground=False`).

## Common errors

- **`gpu_ran=False` even with `LATTICETN_RUN_GPU=1`** — the score's CPU-only
  env hides the GPU when `LATTICETN_RUN_GPU` is unset; if it *is* set and you
  still see a skip, check `torch.cuda.device_count() > 0` (e.g. a stale
  `CUDA_VISIBLE_DEVICES=""` in your shell — unset it).
- **`RuntimeError` from `linalg.qr`/`svd` on GPU** — should not happen on
  complex128 with current PyTorch; the one-site/two-site sweeps use QR/SVD in
  post-step helpers. If it does, fall back to CPU (the CPU run is always the
  source of truth).
- **Mixing CPU and CUDA tensors** — the runner copies CPU tensors onto the GPU
  by value for an apples-to-apples start; do not hand-build a half-CPU/half-GPU
  MPS/MPO.
- **Expecting GPU to be faster** — see the caveat above. A speedup of 0.4× on
  these tiny cases is normal, not a bug.

## Where next

- A larger-N opt-in GPU run (N=8/10/12, bigger χ) to actually see GPU payoff is
  out of scope for `--fast`; it would be a future stage.
- API: `docs/API_OVERVIEW.md` → "Benchmark / score scripts (Stage 6A)".
