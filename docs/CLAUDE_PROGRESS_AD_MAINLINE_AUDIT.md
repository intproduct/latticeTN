# CLAUDE Progress — AD Mainline Audit (re-run)

Date: 2026-07-01
Goal: AD mainline self-audit — verify the project has not drifted from the
"automatic differentiation tensor network" mainline, and decide whether Stage 5A
may continue. No new features developed.

## Environment note

`python` on PATH is a broken MS Store stub. Used
`C:/Apps/Miniforge3/envs/comfyui/python.exe` for all runs (project memory
`python-env`).

## Docs read

- `CLAUDE.md`
- `docs/AD_VARIATIONAL_REPORT.md`
- `docs/CONTRACTION_REPORT.md`
- `docs/CANONICALIZATION_REPORT.md`
- `docs/DMRG_REPORT.md`
- `docs/DMRG_BENCHMARK_REPORT.md`
- `docs/PROJECT_DIRECTION.md` (exists)
- `docs/AD_MAINLINE_POLICY.md` (exists)
- prior `docs/AD_MAINLINE_AUDIT.md` (refreshed this run)
- `docs/AD_GAUGE_REPORT.md` (Stage 5A reference)

## Commands run and results

| Command | Result |
|---|---|
| `C:/Apps/Miniforge3/envs/comfyui/python.exe scripts/validation_score.py --fast` | PASS (36 passed) |
| `C:/Apps/Miniforge3/envs/comfyui/python.exe scripts/benchmark_score.py --fast` | PASS (9 passed + smoke PASS) |
| `C:/Apps/Miniforge3/envs/comfyui/python.exe scripts/canonical_score.py --fast` | PASS (12 passed + smoke PASS) |
| `C:/Apps/Miniforge3/envs/comfyui/python.exe scripts/contraction_score.py --fast` | PASS (20 passed + smoke PASS) |
| `C:/Apps/Miniforge3/envs/comfyui/python.exe scripts/ad_variational_score.py --fast` | PASS (15 passed + AD-MPS Heisenberg smoke PASS) |

All five existing regression scores pass. No GPU runs; all CPU, complex128.

## Source re-inspection

- `latticetn/ad_variational.py`: imports only `.mps`, `.mpo`, `.contractions`,
  `.canonical`. No `dmrg`/`lanczos` import (grep: no matches).
- Loss path `energy()`/`loss()` → `contractions.rayleigh_energy_native` →
  `native_mpo_numerator`/`native_norm_sq`: pure einsum. `grep` of
  `contractions.py` for `eigh|svd|qr|linalg`: no matches.
- `no_grad`/`.data`/`.detach()` in `ad_variational.py` confined to `__init__`,
  `_renormalize`, `_project` (post-step gauge projection), `_grad_norm`
  (`.grad.detach()` read), `_canonical_error` (diagnostic), `_record` report
  reads — all outside the differentiable loss path.
- SVD/QR/canonicalization appear only as post-step gauge projection /
  stabilization / diagnostics, never as the optimizer and never inside the
  loss path.
- Structural guards pass: `tests/test_ad_vs_dmrg_reference.py::
  test_ad_module_does_not_import_dmrg_into_loss_path` and
  `tests/test_ad_gauge_loss_integrity.py::
  test_loss_path_has_no_dmrg_lanczos_projection_or_forbidden` (AST inspection).

## Conclusion

No drift from the AD mainline. The primary solve path is
`MPS parameters -> differentiable Rayleigh quotient -> loss.backward() ->
torch optimizer step`, autograd-clean. DMRG/Lanczos are isolated as classical
reference baselines; SVD/QR/canonicalization are gauge/diagnostics only.
Stage 5A (gauge-stabilized AD-MPS) is already implemented and passing.

```
AD mainline audit: PASS. Ready for Stage 5A AD local tensor optimization.
```

Stage 5A itself is NOT implemented further in this run (no new features), per
the goal's stop conditions. Suggested next step: a new `/goal` for Stage 5B AD
local-tensor optimization (see `docs/AD_MAINLINE_AUDIT.md` §8).

## Next action

Await a new `/goal` from the user before any Stage 5B / new feature work.
