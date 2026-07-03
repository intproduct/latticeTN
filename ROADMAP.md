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

### Stage 6B — Documentation, tutorials, GitHub Pages skeleton (DONE)

No new algorithms. Expand the English `USER_GUIDE.md` into a computation
guide; add a Chinese `USER_GUIDE.zh-CN.md` (design-motivation-oriented, not a
mechanical translation); add 6 bilingual tutorials
(`docs/tutorials/` + `docs/tutorials.zh-CN/`); add a MkDocs site skeleton
(`mkdocs.yml` + `requirements-docs.txt`, local preview only, no GitHub Pages
auto-deploy). Updates `INDEX`/`API_OVERVIEW`/`README`/`REPO_STATUS`. Physics
conventions and test thresholds unchanged.

### Stage 7A — Spinless fermion t-V chain (Jordan-Wigner) + unified GPU timing (DONE)

Add the open-boundary 1D spinless fermion t-V chain
`H = -t sum (c^d_i c_{i+1}+h.c.) + V sum (n_i-1/2)(n_{i+1}-1/2) - mu sum (n_i-1/2)`
on top of the **unchanged** AD mainline. New: `latticetn/fermion_operators.py`
(local `I,c,c^d,n,F=(-1)^n,n-1/2`), `operators.spinless_fermion_dense` (explicit
global JW parity string), `MPO.generate_spinless_fermion` (bond-dim-6 MPO with
a JW parity-carrying virtual state, `to_dense` matches the dense ref), and
fermion observables (`<n_i>`, `<n_i n_j>`, `<c^d_i c_{i+1}+h.c.>`). The AD loss
path is operator-agnostic and is NOT modified — only the Hamiltonian/MPO/
operator layer is new. A **unified GPU selector** (`scripts/gpu_selector.py`)
selects a V100/TITAN V (clean-skip otherwise; no fallback) and is the canonical
selector from Stage 7A onward. Scored by `scripts/fermion_score.py --fast`;
documented in `docs/FERMION_SPEC.md`, `docs/FERMION_PROTOCOL.md`,
`docs/FERMION_REPORT.md`. This is 1D Jordan-Wigner fermions, NOT a full graded
fermionic tensor network. No TDVP / finite-temperature / Hubbard / graded
fermionic tensors.

### Stage 7B — General 1D model builder + unified benchmark registry (DONE)

Abstract the existing Heisenberg and spinless-fermion t-V Hamiltonians behind a
**unified 1D model-construction layer** (`latticetn/model_builder.py`:
`ModelSpec` + term types + presets + `build_dense`/`build_mpo`) and add a
**unified CPU/GPU benchmark registry** (`latticetn/benchmarking.py`) that records
the Stage-7A+ timing contract (model, N, chi, solver, device, device_name,
dtype, runtime, speedup, final_energy, exact_error, below_ground,
gpu_skip_reason) across presets on CPU and (opt-in) a V100/TITAN V via
`scripts/gpu_selector.py`. This is a **model/MPO construction layer, NOT a new
solver** — the AD mainline is unchanged; SVD/QR/canonicalization remain
auxiliary stabilization; exact/DMRG/Lanczos remain reference baselines. The
builders dispatch to the existing validated generators so the physics is
byte-identical to Stage 1/7A. Scored by `scripts/model_builder_score.py --fast`;
documented in `docs/MODEL_BUILDER_SPEC.md`, `docs/MODEL_BUILDER_PROTOCOL.md`,
`docs/MODEL_BUILDER_REPORT.md`. Prepares for future Hubbard / TDVP /
finite-temperature work without touching the loss path. No TDVP /
finite-temperature / Hubbard / graded fermionic tensors.

### Stage 7C — Spinful Hubbard chain (Jordan-Wigner) (DONE)

Add the open-boundary 1D **spinful Hubbard chain** on top of the unchanged AD
mainline, reusing the Stage 7B `model_builder` + benchmark registry +
unified V100/TITAN V GPU selector. The model is
`H = -t Σ_{i,s}(c†_{i,s}c_{i+1,s}+h.c.) + U Σ_i(n_{i,↑}-1/2)(n_{i,↓}-1/2) - μ Σ_i(n_{i,↑}+n_{i,↓}-1) - h Σ_i(n_{i,↑}-n_{i,↓})`,
d=4, local basis `|0>,|↑>,|↓>,|↑↓>`, site-major global mode ordering
`(0↑,0↓,1↑,1↓,...)`. Adds `hubbard_local_operators` (4x4 local ops with the
on-site JW order up-first), `hubbard_dense` (explicit 2N-mode JW parity,
site-level standard-basis build cross-checked against the full JW string),
`MPO.generate_hubbard` (bond dim 6, d=4, no separate parity-carrying state —
the inter-site parity cancels in the product; the surviving site-`i` parity
is absorbed into the `@P`/`P@` left factors), the `hubbard_model` preset,
Hubbard observables (local densities, double occupancy, sz, spin-resolved NN
hopping), the Hubbard test suite, `scripts/run_hubbard_benchmark.py` +
`scripts/hubbard_score.py`. This is 1D Jordan-Wigner fermions, NOT graded
fermionic tensors; the AD loss path is unchanged. Scored by
`scripts/hubbard_score.py --fast`; documented in `docs/HUBBARD_SPEC.md`,
`docs/HUBBARD_PROTOCOL.md`, `docs/HUBBARD_REPORT.md`. No TDVP /
finite-temperature / graded fermionic tensors / long-range models.

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
