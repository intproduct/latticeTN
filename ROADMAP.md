# Roadmap

This document describes **future directions only** — nothing here is implemented
yet, and no feature work is started without a new `/goal`. The inviolable rule
for every future stage: **the AD mainline stays the primary solver; classical
solvers stay reference baselines; the loss path stays autograd-clean.**

## Guiding principles (carry forward from `docs/AD_MAINLINE_POLICY.md`)

- Primary path: `MPS parameters -> differentiable Rayleigh quotient ->
  loss.backward() -> torch optimizer step`.
- SVD / QR / canonicalization / compression = optional post-step
  stabilization / projection / compression, **never** the optimizer.
- `dmrg` / `lanczos` / `eigh` / dense local eigensolvers = reference baselines /
  oracles only, never in the AD loss/optimization path.
- Each new stage ships its own `*_score.py` + report; default
  `validation_score`/`benchmark_score` lists are not modified to depend on GPU
  or classical-solver tests.

## Planned stages

### Stage 5B — Two-site AD local optimization (DONE)

Extend Stage 5A's AD local-tensor optimization to a **two-site** active window:
at each bond contract the two adjacent site tensors into a single trainable
two-site center tensor `Θ`, train it on the differentiable local Rayleigh
quotient `E(Θ)=<Θ|H_eff|Θ>/<Θ|Θ>` by autograd (`loss.backward()` + Adam/LBFGS),
then split `Θ` back into two site tensors by SVD with optional
`max_bond_dim`/`cutoff` truncation, and sweep left-to-right / right-to-left.
This is the AD analogue of two-site DMRG, with **gradient descent replacing the
local eigensolver**. SVD/QR here are post-step split / compression / inter-bond
gauge fixing, **not** the solver. Implemented by `latticetn/ad_two_site.py`;
scored by `scripts/ad_two_site_score.py --fast`; documented in
`docs/AD_TWO_SITE_SPEC.md`, `docs/AD_TWO_SITE_PROTOCOL.md`,
`docs/AD_TWO_SITE_REPORT.md`.

### Stage 5C — GPU AD benchmark (DONE, shipped as Stage 6A)

An **opt-in** GPU benchmark of the global / local AD-MPS solvers at larger N and
chi, respecting the project GPU policy. Default tests stay CPU-only. This was
**implemented as Stage 6A** (`scripts/ad_gpu_benchmark_score.py` +
`scripts/run_ad_gpu_benchmark.py`, docs `AD_GPU_BENCHMARK_*`): it benchmarks
the three AD mainline solvers (global AD-MPS, one-site AD, two-site AD) on CPU
and the machine's single GPU (`cuda:0` when `LATTICETN_RUN_GPU=1` and CUDA is
available; clean-skip otherwise), with DMRG / exact diagonalization as
reference baselines only. It is never coupled to the default fast scores.

### Stage 6 — Model extensions (XXZ / TFI)

Add XXZ and transverse-field Ising (TFI) Hamiltonians as
`latticetn.operators` dense references + MPO builders + MPO-to-dense tests, plus
AD and DMRG-baseline comparisons. Validate the AD mainline generalizes beyond
the isotropic Heisenberg point.

### Stage 7 — Real-time evolution (optional, future)

Time-evolving methods (TEBD and/or TDVP) as an **optional** direction. If
pursued, the differentiable time-evolution should remain on the AD mainline
where feasible (e.g. differentiable Trotter steps trained/projected by autograd),
with classical TEBD/TDVP as reference baselines. Out of scope until a dedicated
goal.

## Non-goals

- Replacing the AD mainline with classical DMRG/Lanczos.
- Using SVD/eigh as the optimizer.
- Broad refactors that risk the passing validation suite or widen dependencies
  beyond torch/numpy/scipy/pytest/tqdm/matplotlib.
