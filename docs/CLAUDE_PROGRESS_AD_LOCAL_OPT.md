# CLAUDE Progress — Stage 5A AD Local-Tensor Optimization

Date: 2026-07-01
Goal: implement the AD mainline's local-tensor optimization prototype.
Pre-condition: `docs/AD_MAINLINE_AUDIT.md` PASS (no drift from the AD mainline).

## Environment note

`python` on PATH is a broken MS Store stub. Used
`C:/Apps/Miniforge3/envs/comfyui/python.exe` (project memory `python-env`).
All runs CPU-only, `torch.complex128`.

## Design

AD local-tensor optimization = the autograd analogue of DMRG's local update,
but the local update is **gradient descent on a differentiable Rayleigh
quotient**, not a local eigensolver.

- Bring the MPS to mixed-canonical form with a chosen orthogonality `center`.
  Freeze every tensor except the center tensor, which is the only trainable
  `nn.Parameter`.
- Loss = `contractions.rayleigh_energy_native(mps, mpo)` (differentiable einsum
  sweep). `loss.backward()` populates only the center tensor's grad.
- Optimize the center tensor with a torch optimizer (LBFGS default; Adam
  supported).
- Optional post-step `stabilization` (OUTSIDE the loss graph, `no_grad`
  mutating `.data`): `none` | `tensor_norm` | `qr` (mixed-canonical projection,
  keeps center) | `canonical` (full left-canonical Stage-3A sweep).
- Sweep: move the orthogonality center to the next site by a QR sweep (right)
  or LQ sweep (left) — center movement, NOT the optimizer — and optimize the
  new center. Alternate right/left sweeps.

Policy: loss path autograd-clean (no `detach`/`.data`/`no_grad`/unnecessary
`.item`, no `eigh`/`svd`/`qr`, no `dmrg`/`lanczos`). Module imports neither
`dmrg` nor `lanczos`. Enforced by AST inspection in
`tests/test_ad_local_opt_policy.py`.

## Files added/updated

- `latticetn/ad_local.py` — `ADLocalOptimizer` + `train_ad_local`.
- `tests/test_ad_local_opt_loss.py` — loss is differentiable scalar,
  scale-invariant, center-only trainable; center movement preserves energy.
- `tests/test_ad_local_opt_gradients.py` — backward populates center grad
  (finite), frozen tensors get None, grad matches finite differences.
- `tests/test_ad_local_opt_step.py` — a step lowers energy; full sweeps converge
  N=4 (<1e-6) / N=6 (<1e-5); no below-ground across all stabilizations.
- `tests/test_ad_local_opt_vs_global_ad.py` — AD-local matches global AD-MPS at
  N=4 (<1e-3) and N=6 (<1e-2).
- `tests/test_ad_local_opt_policy.py` — AST guard: loss path clean, module does
  not import dmrg/lanczos, eigh nowhere, QR/SVD only in marked helpers.
- `scripts/run_ad_local_opt.py` — generates `docs/AD_LOCAL_OPT_REPORT.md`.
- `scripts/ad_local_opt_score.py` — `--fast`.
- `docs/AD_LOCAL_OPT_SPEC.md`, `docs/AD_LOCAL_OPT_PROTOCOL.md`,
  `docs/AD_LOCAL_OPT_REPORT.md`, this progress file.

## Commands and results

Existing regression (re-confirmed exit 0):

| Command | exit |
|---|:---:|
| `python scripts/validation_score.py --fast` | 0 |
| `python scripts/benchmark_score.py --fast` | 0 |
| `python scripts/canonical_score.py --fast` | 0 |
| `python scripts/contraction_score.py --fast` | 0 |
| `python scripts/ad_variational_score.py --fast` | 0 |

New Stage 5A AD-local:

| Command | result |
|---|---|
| `python -m pytest -q tests/test_ad_local_opt_*.py` | 25 passed |
| `python scripts/run_ad_local_opt.py --markdown-output docs/AD_LOCAL_OPT_REPORT.md` | pass=True (smoke) |
| `python scripts/ad_local_opt_score.py --fast` | **PASS** (exit 0) |

Numerical (LBFGS, seed 0, stabilization=`qr`, 4 sweeps × 20 local steps):

| N | exact E0 | AD-local final E | abs err | tol | below ground | runtime |
|---:|---:|---:|---:|---:|:---:|---:|
| 4 | -1.6160254038 | -1.6160254037 | 4.19e-11 | 1e-08 | False | 1.53s |
| 6 | -2.4935771339 | -2.4935771330 | 8.91e-10 | 1e-05 | False | 0.29s |

Comparisons (N=6): |AD-local − global AD-MPS| = 1.47e-05;
|AD-local − DMRG| = 8.91e-10. All checks pass.

## Notes / issues encountered

- Per-site Adam was slow to converge (one center tensor with a fixed
  environment needs many first-order steps). Switched the default optimizer to
  LBFGS (the local Rayleigh problem is near-quadratic in the center tensor):
  4 sweeps × 20 local steps reach ~1e-10 for N=4/6 in <2s. Adam remains
  supported and is exercised by `test_single_local_step_lowers_energy`.
- After a QR/LQ center move or a `canonical`/`qr` stabilization the center
  tensor's `.data` is non-contiguous, which broke LBFGS's `p.grad.view(-1)`.
  `ADLocalOptimizer.set_center` now re-creates the center as a fresh
  **contiguous** leaf Parameter before training each site. (Parameter-config
  mutation under `no_grad`, never in the loss path.)
- `requires_grad=True` -> scalar `UserWarning` appears only in report-path
  `float(...)` reads, never in the differentiable loss path (consistent with
  the Stage 4R / 5A-gauge reports).

## Conclusion

Stage 5A AD local-tensor optimization is implemented and passing. The mainline
is AD local optimization; SVD/QR/canonicalization are optional post-step
stabilization only, never the solver. DMRG/Lanczos remain classical reference
baselines. The loss path is autograd-clean (AST-enforced).

Stop conditions met:
1. existing regression scores all exit 0;
2. `python scripts/ad_local_opt_score.py --fast` exits 0;
3. `docs/AD_LOCAL_OPT_REPORT.md` contains exact / global-AD / DMRG comparisons,
   energy history, gradient check, stabilization settings, pass/fail, known
   limitations;
4. docs declare the mainline / stabilization policy explicitly;
5. this progress file records changes, commands, results, next step.

## Next-step suggestion

A future `/goal` could extend AD local-tensor optimization to **two-site AD
updates with bond-growing projection** (AD analogue of two-site DMRG), using
Stage 3A `svd_compress` as a post-step bond-growing/compression projection —
still purely AD on the differentiable Rayleigh quotient, classical solvers
isolated, loss path autograd-clean. Or scale the AD-local sweep to larger N /
higher chi with wall-clock budgets. No new feature work without a new `/goal`.
