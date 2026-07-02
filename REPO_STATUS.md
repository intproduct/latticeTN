# Repository Status

Last updated: 2026-07-02 (Stage 6A CPU/GPU AD benchmark)

## What this repository is

`latticeTN` is an **automatic-differentiation tensor-network** library. The
mainline solver trains an MPS on an MPO Hamiltonian by minimizing a
differentiable Rayleigh quotient with PyTorch autograd + a torch optimizer.
Classical DMRG / Lanczos are reference baselines, not the mainline.

## Repository layout

```
latticetn/            # the importable package (AD mainline + baselines)
  __init__.py
  mps.py              # MPS (trainable nn.Parameter site tensors)
  mpo.py              # MPO builder (Heisenberg / TFI)
  operators.py        # spin operators + dense Hamiltonians + ED reference
  contractions.py     # differentiable native contractions (the loss path)
  observables.py      # dense-reference observables
  canonical.py        # SVD/QR canonicalization + compression (Stage 3A)
  ad_variational.py   # Stage 4R global AD-MPS            [AD MAINLINE]
  ad_local.py         # Stage 5A AD local-tensor optim.   [AD MAINLINE]
  ad_two_site.py      # Stage 5B two-site AD local optim. [AD MAINLINE]
  dmrg.py             # Stage 4A/4B classical DMRG        [REFERENCE BASELINE]
  lanczos.py          # Stage 4B Krylov local eigensolver [REFERENCE BASELINE]
scripts/              # *_score.py runners + run_*.py smoke/generators
tests/                # formal pytest suite (default collection target)
docs/                 # specs, protocols, reports, user guide, policy
examples/             # short CPU usage examples
legacy/stage0_prototypes/  # archived pre-package prototypes (NOT used; traceable)
pyproject.toml        # packaging (pip install -e .)
pytest.ini            # scopes default collection to tests/
requirements*.txt     # runtime + dev deps
.gitignore            # caches / machine-private config / temp outputs
README.md, ROADMAP.md, REPO_STATUS.md
```

## AD mainline modules

These implement the primary solver and its differentiable loss path:

| Module | Role |
|---|---|
| `latticetn.mps` | traininable MPS (`nn.Parameter` site tensors), differentiable `overlap`/`energy_with_MPO`/`to_dense`. |
| `latticetn.mpo` | differentiable MPO construction. |
| `latticetn.operators` | spin operators + dense reference Hamiltonians + ED gold reference. |
| `latticetn.contractions` | differentiable native contractions; `rayleigh_energy_native` = the loss. |
| `latticetn.ad_variational` | **Stage 4R** global AD-MPS (`ADVariationalMPS`, `train_ad_mps`). |
| `latticetn.ad_local` | **Stage 5A** AD local-tensor optimization (`ADLocalOptimizer`, `train_ad_local`). |
| `latticetn.ad_two_site` | **Stage 5B** two-site AD local-tensor optimization (`ADTwoSiteOptimizer`, `train_ad_two_site`); optional bond growth via post-step SVD split. |

Loss-path cleanliness (no `detach`/`.data`/`no_grad`/unnecessary `.item`, no
`eigh`/`svd`/`qr`, no `dmrg`/`lanczos`) is AST-enforced by
`tests/test_ad_gauge_loss_integrity.py`, `tests/test_ad_local_opt_policy.py`,
and `tests/test_ad_two_site_policy.py`.

## Reference-baseline modules (NOT the mainline)

| Module | Role | Guard |
|---|---|---|
| `latticetn.dmrg` | classical two-site DMRG (Stage 4A/4B) | own score scripts; not imported by `ad_variational`/`ad_local`. |
| `latticetn.lanczos` | Krylov local eigensolver (Stage 4B) | same; not imported by the AD modules. |
| `latticetn.canonical` | SVD/QR canonicalization + compression | post-step stabilization / diagnostics only; never in the loss path. |
| `latticetn.observables` | dense-reference observables | small-system validation/diagnostics. |

## Passed score scripts (as of this update)

All run CPU-only, `torch.complex128`.

| Command | Result |
|---|:---:|
| `python scripts/validation_score.py --fast` | PASS |
| `python scripts/benchmark_score.py --fast` | PASS |
| `python scripts/canonical_score.py --fast` | PASS |
| `python scripts/contraction_score.py --fast` | PASS |
| `python scripts/ad_variational_score.py --fast` | PASS |
| `python scripts/ad_local_opt_score.py --fast` | PASS |
| `python scripts/ad_two_site_score.py --fast` | PASS |
| `python scripts/ad_gpu_benchmark_score.py --fast` | PASS (CPU-only by default; GPU opt-in via `LATTICETN_RUN_GPU=1`) |

Default `pytest -q` collects **233 tests, all under `tests/`** (Stage 5B was
215; Stage 6A adds three benchmark test files); nothing under `legacy/` or
`examples/` is collected.

## Legacy / archived code

`legacy/stage0_prototypes/` holds 30 pre-`latticetn`-package prototype files
(`AD_MPS.py`, `AD_DMRG.py`, `AD_MERA.py`, `AD_PEPS.py`, root `operators.py`,
early `test_*.py`/`*.ipynb`, `update_log.md`). They are **kept for
traceability**, are **not imported** by the active codebase, and are **not**
collected by the default test suite. They may violate current conventions
(`S=sigma/2`, complex128, autograd-clean loss paths) — treat their physics with
caution. See `legacy/README.md`.

## Not yet done

- No remote git repository; the repo is prepared for a local `git init` and a
  later (manual) remote push. See the suggested commands in the Repo-Prep
  handoff.
- Future stages (XXZ/TFI, TEBD/TDVP) are described in `ROADMAP.md` but not
  started. Stage 5C (GPU AD benchmark, shipped as **Stage 6A**) and Stage 5B
  (two-site AD local optimization with optional bond growth) are **done**.
