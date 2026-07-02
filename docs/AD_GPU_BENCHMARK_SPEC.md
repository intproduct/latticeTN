# Stage 6A — CPU/GPU AD Solver Benchmark Spec

> Status: implemented by `scripts/run_ad_gpu_benchmark.py` +
> `scripts/ad_gpu_benchmark_score.py`. This stage benchmarks the **AD mainline
> solvers** (global AD-MPS, one-site AD, two-site AD) on CPU and the machine's
> single GPU. DMRG / Lanczos / exact diagonalization are **reference baselines
> only**, never the AD mainline.

## 1. Goal

Build a **unified, opt-in CPU/GPU benchmark** that systematically compares the
three AD mainline solvers of latticeTN on the open-boundary 1D spin-1/2
Heisenberg chain, measuring **numerical agreement** and **runtime/speedup**
between CPU and the current machine's single GPU. No new algorithm is
introduced; no AD loss path is modified. The benchmark is a *measurement
harness* over the existing AD mainline, not a solver.

Scope is deliberately **small and controlled** (`--fast` defaults to N=4/6,
chi=4/8, short steps/sweeps): the goal is correctness parity + a honest
speedup number, **not** large-scale performance tuning.

## 2. Solvers under benchmark (AD mainline only)

1. **global AD-MPS** — `latticetn.ad_variational.train_ad_mps` (all MPS tensors
   trained simultaneously on the differentiable Rayleigh quotient; Adam).
2. **one-site AD local optimization** — `latticetn.ad_local.train_ad_local`
   (single-site orthogonality-center sweep; LBFGS).
3. **two-site AD local optimization** — `latticetn.ad_two_site.train_ad_two_site`
   (two-site Theta block sweep with optional bond growth/truncation; LBFGS).

Reference baselines (reported for context, **never** the AD mainline):

4. **classical DMRG** — `latticetn.dmrg.run_dmrg` (dense; reference baseline).
5. **exact diagonalization** — `latticetn.operators.exact_ground_energy` on
   `heisenberg_dense` (small-N golden reference; always CPU).

## 3. GPU selection rules (this machine has exactly one GPU)

Unlike Stage 2.5's name-matched GPU smoke, this stage targets the **single GPU
on the current machine** and therefore does **not** filter by GPU name:

1. GPU benchmark is **opt-in**: it runs only when `LATTICETN_RUN_GPU=1`.
2. When opted-in and `torch.cuda.is_available()` is `True`, use `cuda:0`
   (the single visible device).
3. When opted-in but CUDA is unavailable, the GPU portion **clean-skips**:
   the report records the skip and exits 0. The CPU portion still runs.
4. Default `pytest` and the default `--fast` score run are **CPU-only** and
   never require a GPU.
5. The report records GPU name, CUDA version, PyTorch version, device, dtype.

Rationale: Stage 2.5 was written for multi-GPU machines and refuses to fall
back to an unmatched GPU. This stage's contract is "the current machine's one
GPU", so name filtering would be vacuous; `cuda:0` is the correct, unambiguous
target when exactly one device is visible.

## 4. Benchmark requirements

1. `--fast` mode runs small cases (e.g. N=4 and N=6, chi=4/8, short
   steps/sweeps) so the whole score finishes quickly on CPU.
2. CPU and GPU use the **same seed, dtype, and solver config** (identical
   initial MPS tensors copied by value onto the GPU, identical optimizer /
   lr / steps / sweeps).
3. Each case records: `final_energy`, `energy_error` (vs exact),
   `energy_per_bond`, `runtime_s`, `speedup` (CPU runtime / GPU runtime),
   `device`, `dtype`, `solver`, `optimizer`, `N`, `chi`, `seed`,
   `below_ground` flag.
4. Small-N cases are compared against exact diagonalization.
5. DMRG is reported as a reference baseline; it is **never** the AD mainline.
6. Results are saved as JSON (machine-readable) under `docs/`.
7. `docs/AD_GPU_BENCHMARK_REPORT.md` is generated with a CPU/GPU comparison
   table and a conclusion.

## 5. What is NOT done here (hard constraints)

- No new algorithm. No modification to any AD loss path
  (`contractions.rayleigh_energy_native`, `ad_two_site.ADTwoSiteOptimizer.energy`,
  `ad_local`/`ad_variational` energy methods).
- DMRG / Lanczos / `eigh` are never the AD mainline; they are reference only.
- No existing tolerance is widened.
- No long / large-scale benchmark; `--fast` stays small and fast.
- No GPU run unless `LATTICETN_RUN_GPU=1` **and** CUDA is available.
- No caches / pyc / pytest cache / local private config / large temp files are
  committed.

## 6. Autograd rule (per `docs/AD_MAINLINE_POLICY.md`)

The AD solvers' loss paths stay autograd-clean: no `detach()`/`.data`/
`torch.no_grad()`/unnecessary `.item()` inside the differentiable energy.
Energy *comparison* values for the report are extracted **outside** the
gradient computation as plain Python floats, only after `backward()` — that is
the report path, not the differentiable energy path. The benchmark calls the
existing `train_ad_*` functions as-is; it does not touch their internals.

## 7. Physics conventions

Unchanged: `H = J * sum_i S_i.S_{i+1}`, `S = sigma/2`, `J = 1.0`, open
boundary, `torch.complex128`. No silent switch between `S` and `sigma`. ED is
always computed on CPU (it calls `numpy.linalg.eigh`); the GPU is used only
for the AD solver runs.

## 8. Comparisons (reference only)

- exact diagonalization (small N) — golden reference; CPU.
- classical DMRG (`dmrg.run_dmrg`, dense) — reference baseline; CPU.
- CPU vs GPU final energy for each AD solver — parity check (must agree within
  tolerance; the GPU result must not undershoot the exact ground energy beyond
  tolerance).
