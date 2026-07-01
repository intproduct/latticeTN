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

### Stage 5B — Two-site AD local optimization (next)

Extend Stage 5A's AD local-tensor optimization to a **two-site** active window:
train a fused two-site tensor by autograd on the differentiable Rayleigh
quotient, then move the orthogonality center and optionally **grow / compress**
the bond by projecting the merged tensor (Stage 3A `svd_compress` as a
post-step bond-growing projection). This is the AD analogue of two-site DMRG,
with **gradient descent replacing the local eigensolver**. SVD here is a
post-step projection/compression tool, **not** the solver.

### Stage 5C — GPU AD benchmark

An **opt-in** GPU benchmark of the global / local AD-MPS solvers at larger N and
chi, respecting the project GPU policy (name-matched device selection, never
`cuda:0` by default). Default tests stay CPU-only. Adds a `gpu_ad_score.py` and
report; never coupled to the default fast scores.

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
