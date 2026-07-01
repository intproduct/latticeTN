# AD Mainline Audit

Date: 2026-07-01 (re-run)
Auditor: Claude Code (self-audit, no new features developed)
Scope: re-verify the AD mainline against current source after running the full
existing regression suite. This re-audit refreshes the earlier 2026-07-01 audit
against the live code and the freshly-run scores.

## 0. Regression results (re-run this audit)

Python: `C:/Apps/Miniforge3/envs/comfyui/python.exe` (the `python` on PATH is a
broken MS Store stub; see project memory).

| Script | Result | Detail |
|---|:---:|---|
| `scripts/validation_score.py --fast` | **PASS** | 36 passed |
| `scripts/benchmark_score.py --fast` | **PASS** | 9 passed + benchmark smoke PASS |
| `scripts/canonical_score.py --fast` | **PASS** | 12 passed + canonical smoke PASS |
| `scripts/contraction_score.py --fast` | **PASS** | 20 passed + contraction smoke PASS |
| `scripts/ad_variational_score.py --fast` | **PASS** | 15 passed + AD-MPS Heisenberg smoke PASS |

All five existing regression scores pass. (Cosmetic-only `UserWarning:
Converting a tensor with requires_grad=True to a scalar…` appears in report
paths, not in any differentiable loss path.)

## 1. Is the project off the AD mainline?

**No.** The project's primary solve path is automatic differentiation of an MPS
variational ansatz, exactly as `CLAUDE.md` prescribes ("an
automatic-differentiation tensor network project built with PyTorch" whose
solver is "MPS + MPO + PyTorch autograd"). Stage 4R realigned the project to
the AD mainline after the Stage 4A/4B classical-DMRG detour, and Stage 5A
extended the mainline with post-step gauge projection while keeping the loss
path autograd-clean. No drift was found in this re-audit.

## 2. Modules in the AD mainline

| Module | Role |
|---|---|
| `latticetn/mps.py` | MPS with trainable `nn.Parameter` site tensors; differentiable `overlap`/`energy_with_MPO`/`to_dense`. |
| `latticetn/mpo.py` | Differentiable MPO construction (Heisenberg / TFI). |
| `latticetn/contractions.py` | **Differentiable** native contractions: `native_norm_sq`, `native_mpo_numerator`, `rayleigh_energy_native` (= the loss). Plain einsum sweeps; no `eigh`/`svd`/`qr`/`detach`/`.data`/`no_grad`/`.item()`. |
| `latticetn/ad_variational.py` | `ADVariationalMPS` (trainable params) + `train_ad_mps` (Adam/LBFGS) on `rayleigh_energy_native`. Stage 5A post-step gauge projection. |
| `latticetn/operators.py` | Spin operators + dense reference Hamiltonians + exact-diagonalization gold reference. |

Verified against current source: the loss path is
`ADVariationalMPS.energy()` → `contractions.rayleigh_energy_native` →
`native_mpo_expectation` → `native_mpo_numerator` (einsum) / `native_norm_sq`
(einsum). `ad_variational.py` imports only `.mps`, `.mpo`, `.contractions`,
`.canonical` — it does **not** import `.dmrg` or `.lanczos` (grep confirms zero
references; AST test enforces).

## 3. Modules that are classical baseline / oracle only

| Module | Role | Guarded by |
|---|---|---|
| `latticetn/dmrg.py` | Classical two-site DMRG (Stage 4A/4B) — reference only. | own score scripts; not in `validation_score`/`benchmark_score` default lists; not imported by `ad_variational.py`. |
| `latticetn/lanczos.py` | Krylov local eigensolver (Stage 4B) — reference only. | same; not imported by `ad_variational.py`. |
| `latticetn/canonical.py` | SVD/QR canonicalization + compression — used as **gauge projection / diagnostics / reference**, not as the optimizer. | called only in `ad_variational._project` (post-step, `no_grad`) and `_canonical_error` (diagnostic); never in the loss path. |
| exact diagonalization (`operators.exact_ground_energy`) | Golden reference. | report/diagnostic only. |

## 4. Are SVD / QR / canonicalization / compression mis-used as the main optimizer?

**No.** Their only roles are:
- **gauge fixing / projection** (post-step, outside the loss graph):
  `ad_variational._project(..., 'canonical')` uses `canonical.left_canonical`
  (built on QR) written onto `.data` under `no_grad`;
- **stabilization**: per-tensor L2 renormalization (`_renormalize`), same
  post-step, outside-the-graph contract;
- **compression**: `canonical.svd_compress` is a reporting/preprocessing/
  post-step tool, never the optimizer;
- **diagnostics**: `_canonical_error` (`canonical.left_orthonormal_all`).

`grep` of `latticetn/contractions.py` finds **no** `eigh`/`svd`/`qr`/`linalg`
occurrences — none of these reach `rayleigh_energy_native` /
`native_mpo_numerator` / `native_norm_sq` (the loss path).

## 5. Are DMRG / Lanczos mis-used in the AD main optimization path?

**No.** `latticetn/ad_variational.py` imports neither `dmrg` nor `lanczos`
(grep confirms zero references). Structural guards enforce this:
- `tests/test_ad_vs_dmrg_reference.py::test_ad_module_does_not_import_dmrg_into_loss_path`
  asserts the AD module source references none of `dmrg`, `lanczos`,
  `run_dmrg`, `two_site_sweep`, `lanczos_lowest_eigenpair`.
- `tests/test_ad_gauge_loss_integrity.py::test_loss_path_has_no_dmrg_lanczos_projection_or_forbidden`
  AST-inspects `energy()`/`loss()` and `rayleigh_energy_native` bodies,
  rejecting `dmrg`, `lanczos`, `no_grad`, `.detach()`, `.data`, `.item()`,
  `_project`, `left_canonical`.

DMRG/Lanczos are run only inside their own score scripts
(`dmrg_score.py`, `dmrg_benchmark_score.py`) and in the AD-vs-DMRG *reference*
test, never in the AD optimization path.

## 6. Autograd-rule compliance of the main loss path

Source inspection of `ad_variational.py` and `contractions.py`:
- `ADVariationalMPS.energy()` / `loss()`: only `return
  K.rayleigh_energy_native(...)` — no `no_grad`/`detach`/`.data`/`.item()`.
- `rayleigh_energy_native` → `native_mpo_expectation` → `native_mpo_numerator`
  / `native_norm_sq`: pure einsum, no forbidden calls.
- All `no_grad`/`.data`/`.detach()` occurrences in `ad_variational.py` are in
  explicitly-marked places **outside** the loss path: `__init__` (parameter
  setup), `_renormalize` (post-step), `_project` (post-step), `_grad_norm`
  (reads already-computed `.grad`), `_canonical_error` (diagnostic),
  `_record`/report reads.

This is enforced by AST tests `tests/test_ad_variational_loss.py` and
`tests/test_ad_gauge_loss_integrity.py`.

## 7. Can we proceed to Stage 5A?

**Stage 5A is already complete** (gauge-stabilized AD-MPS optimizer; see
`docs/AD_GAUGE_REPORT.md`). So the applicable question is whether the *next*
AD-local stage may proceed: **yes.** The mainline is intact, the loss path is
autograd-clean, classical solvers are isolated as references, and all five
existing regression scores pass. No correction is required before further AD
local-tensor work.

## 8. Suggested next `/goal` summary

> "Stage 5B: AD local-tensor optimization — extend the AD mainline further
> (e.g. mixed-canonical projection with a moving orthogonality center, or
> AD + Stage 3A `svd_compress` as a post-step bond-growing projection, or
> AD local two-site tensor updates trained by autograd). Do NOT extend
> classical DMRG/Lanczos; keep the loss path autograd-clean per
> docs/AD_MAINLINE_POLICY.md and add its own `*_score.py` + report."

## Verdict

```
AD mainline audit: PASS. Ready for Stage 5A AD local tensor optimization.
```

(Stage 5A itself is already implemented and passing — see
`docs/AD_GAUGE_REPORT.md` and `scripts/ad_variational_score.py` /
`scripts/ad_gauge_score.py`. This line confirms the re-audit found no drift and
the AD mainline remains the project's primary path.)
