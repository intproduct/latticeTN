# latticeTN — Automatic-Differentiation Tensor Network Library

`latticeTN` is an **automatic-differentiation (AD) tensor-network** library built
with PyTorch. The primary solver path trains a matrix-product state (MPS) on a
matrix-product-operator (MPO) Hamiltonian by minimizing a **differentiable
Rayleigh quotient** with `loss.backward()` and a torch optimizer. Classical DMRG
and Lanczos are included **only as reference baselines / sanity oracles**, not as
the mainline.

> Scientific validation target: solve the finite open-boundary 1D spin-1/2
> Heisenberg chain with MPS + MPO + PyTorch autograd, verified against exact
> diagonalization.

## Mainline

```
MPS parameters (trainable nn.Parameter)
   -> differentiable Rayleigh quotient  E = <psi|H|psi> / <psi|psi>
   -> loss.backward()   (autograd)
   -> torch optimizer step (Adam / LBFGS)
```

- The differentiable energy is `latticetn.contractions.rayleigh_energy_native`
  — a plain einsum sweep with **no** `detach()`/`.data`/`torch.no_grad()`/
  unnecessary `.item()`, **no** `eigh`/`svd`/`qr`, and **no** call into
  `dmrg`/`lanczos` in the loss path (AST-enforced).
- SVD / QR / canonicalization / compression are **optional post-step
  stabilization / projection / compression only** — never the optimizer.
- `latticetn.dmrg` and `latticetn.lanczos` are **classical reference baselines**.

## Completed stages

| Stage | What | Status |
|---|---|---|
| Stage 1 | Physical validation (MPS overlap, dense ED) | done |
| Stage 2 | Observables + Heisenberg benchmark | done |
| Stage 2.5 | GPU readiness (opt-in smoke) | done |
| Stage 3A | Canonicalization + SVD compression | done |
| Stage 3B | Native (scalable) MPS/MPO contractions | done |
| Stage 4A/4B | Classical two-site DMRG + Lanczos **baseline** | done |
| Stage 4R | Global AD-MPS (all tensors trained at once) | done |
| Stage 5A | AD local-tensor optimization (center-tensor sweep) | done |
| Stage 5B | Two-site AD local optimization (optional bond growth) | done |
| Stage 6A | CPU/GPU AD solver benchmark (opt-in GPU) | done |
| Stage 6B | Bilingual docs, tutorials, MkDocs skeleton | done |

See `ROADMAP.md` for future directions (XXZ/TFI extensions, TEBD/TDVP). The
Stage 5C GPU AD benchmark slot from earlier drafts is **implemented as Stage
6A** — see `docs/AD_GPU_BENCHMARK_SPEC.md`.

## Quick install

```bash
# 1. create / activate an environment (conda / venv)
python -m venv .venv && source .venv/bin/activate   # or your conda env

# 2. install runtime + dev deps, then the package in editable mode
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

> Note: physics validation runs on **CPU** with `torch.complex128`. CUDA is
> opt-in and never used by default tests.

## Quick validation

Each stage has a `*_score.py --fast` runner:

```bash
python scripts/validation_score.py --fast      # Stage 1/2 physical validation
python scripts/benchmark_score.py --fast       # Stage 2 Heisenberg benchmark
python scripts/canonical_score.py --fast       # Stage 3A canonicalization
python scripts/contraction_score.py --fast     # Stage 3B native contractions
python scripts/ad_variational_score.py --fast  # Stage 4R global AD-MPS
python scripts/ad_local_opt_score.py --fast    # Stage 5A AD local optimization
python scripts/ad_two_site_score.py --fast     # Stage 5B two-site AD local optimization
python scripts/ad_gpu_benchmark_score.py --fast # Stage 6A CPU/GPU AD benchmark (CPU-only by default)

# all fast scores at once (no GPU smoke):
bash scripts/run_all_fast_scores.sh
```

Opt-in GPU smoke (only run deliberately; never in CI by default):

```bash
python scripts/gpu_score.py --fast              # Stage 2.5 GPU correctness smoke
# Stage 6A GPU portion (uses cuda:0; clean-skips if CUDA unavailable):
LATTICETN_RUN_GPU=1 python scripts/ad_gpu_benchmark_score.py --fast
```

## Minimal example

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.contractions import rayleigh_energy_native
from latticetn.operators import heisenberg_dense, exact_ground_energy

N, chi, dtype = 6, 8, tc.complex128
mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)

E = rayleigh_energy_native(mps, mpo)                      # differentiable
E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))
print("Rayleigh E =", float(E.real), " exact E0 =", E0)
```

Training it by autograd (see `examples/heisenberg_ad_mps.py`):

```python
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps
ad = ADVariationalMPS(mps, mpo)               # all site tensors trainable
res = train_ad_mps(ad, num_steps=300, lr=1e-2, optimizer="adam")
print("final E =", res["final_energy"])
```

## Where to go next

- **`docs/USER_GUIDE.md`** — the detailed, user-facing guide (EN) (install, concepts,
  examples, observables, canonicalization, adding new models, running scores,
  common pitfalls). 中文版：`docs/USER_GUIDE.zh-CN.md`.
- **`docs/tutorials/`** (EN) and **`docs/tutorials.zh-CN/`** (中文) — step-by-step,
  runnable walkthroughs (quickstart, each AD solver, CPU/GPU benchmark, add a
  new model) with expected output and common errors.
- **`docs/API_OVERVIEW.md`** — module-by-module reference.
- **`docs/INDEX.md`** — documentation navigation.
- **`mkdocs.yml`** + **`requirements-docs.txt`** — optional local doc site:
  `pip install -r requirements-docs.txt && mkdocs serve`.
- **`REPO_STATUS.md`** — current repository state and what's mainline vs. baseline.
- **`ROADMAP.md`** — future directions.

## Project conventions

- Spin operators `S = sigma / 2` (never silently mix with Pauli `sigma`).
- Open boundary; MPS index order `(left, phys, right)`, MPO `(left, right,
  phys_in, phys_out)`.
- Default dtype `torch.complex128`, default device CPU.
- The loss path stays autograd-clean; `no_grad`/`.data`/`.detach()` only appear
  in post-step stabilization / diagnostics (see `docs/AD_MAINLINE_POLICY.md`).

## License

MIT (see `pyproject.toml`). This is a research/educational codebase.
