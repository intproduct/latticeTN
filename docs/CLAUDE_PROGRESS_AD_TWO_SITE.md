# CLAUGE Progress — Stage 5B Two-Site AD Local Optimization

## Goal

Implement two-site AD local-tensor optimization with optional bond growth
(Stage 5B): construct a two-site center tensor `Θ`, train it as the only
trainable parameter on the differentiable local Rayleigh quotient via
`loss.backward()` + Adam/LBFGS, split `Θ` back into two MPS tensors by SVD
(allowed only as post-step split/compression/stabilization, NOT as the solver),
and sweep left-to-right / right-to-left. No Lanczos / `eigh` / classical DMRG
in the AD path.

## Files added / modified this stage

Added:
- `latticetn/ad_two_site.py` — two-site AD optimizer + sweep driver.
- `scripts/run_ad_two_site.py` — runner / report generator.
- `scripts/ad_two_site_score.py` — Stage 5B score script.
- `tests/test_ad_two_site_loss.py`
- `tests/test_ad_two_site_gradients.py`
- `tests/test_ad_two_site_split.py`
- `tests/test_ad_two_site_sweep_smoke.py`
- `tests/test_ad_two_site_vs_one_site.py`
- `tests/test_ad_two_site_policy.py`
- `docs/AD_TWO_SITE_SPEC.md`
- `docs/AD_TWO_SITE_PROTOCOL.md`
- `docs/AD_TWO_SITE_REPORT.md` (generated)
- `docs/CLAUDE_PROGRESS_AD_TWO_SITE.md`

Modified (none required for correctness this stage; the two-site module reuses
`MPS`, `MPO`, `contractions.rayleigh_energy_native`, `canonical` diagnostics
as-is). No changes to `latticetn/ad_local.py`, `canonical.py`, `mps.py`, or
`contractions.py` were needed — the two-site path is additive.

## Commands run and results

Existing regressions (all exit 0, on `main` before branching):
- `python scripts/validation_score.py --fast` → PASS
- `python scripts/benchmark_score.py --fast` → PASS
- `python scripts/canonical_score.py --fast` → PASS
- `python scripts/contraction_score.py --fast` → PASS
- `python scripts/ad_variational_score.py --fast` → PASS
- `python scripts/ad_local_opt_score.py --fast` → PASS

Stage 5B:
- `python -m pytest -q tests/test_ad_two_site_*.py` → 36 passed.
- `python scripts/ad_two_site_score.py --fast` → `AD two-site score: PASS`
  (exit 0), regenerates `docs/AD_TWO_SITE_REPORT.md`.

## Key results (LBFGS, seed 0, chi=8, max_bond_dim=8)

| N | exact E0 | two-site AD final E | abs err | below ground | energy decreased |
|---:|---:|---:|---:|:---:|:---:|
| 4 | -1.6160254038 | -1.6160254036 | 1.82e-10 | False | True |
| 6 | -2.4935771339 | -2.4935771330 | 8.39e-10 | False | True |

Cross-checks (N=6):
- |two-site − one-site AD| ≈ 5.2e-11
- |two-site − global AD-MPS| ≈ 1.5e-5
- |two-site AD − DMRG| ≈ 8.4e-10

All variational (no below-ground violation); Θ grad non-None and finite.

## Design notes / assumptions

- The local loss `E(Θ) = <Θ|H_eff|Θ>/<Θ|Θ>` equals the GLOBAL Rayleigh quotient
  while the chain is orthonormal around the two-site block (standard two-site
  variational principle). This is the differentiable analogue of the DMRG
  effective-Hamiltonian objective, but solved by gradient descent on `Θ`, not
  by `eigh`.
- `H_eff` is built from frozen, detached left/right MPO environments + the two
  MPO tensors; `Θ` is the only trainable leaf. The loss path is pure einsum.
- SVD split (`_split_theta`) and inter-bond QR re-canonicalization
  (`_two_site_mixed_canonical`) live under `no_grad` on detached data — post-step
  compression / gauge fixing, NOT the solver. Enforced by AST in
  `tests/test_ad_two_site_policy.py`.
- Optional `max_bond_dim` / `cutoff` provide bond growth / truncation; without a
  cap the bond can grow up to `min(l*d, d*r)` (full two-site growth), driven
  entirely by gradient descent on `Θ`.
- `_left_mpo_env` / `_right_mpo_env` einsums mirror the Stage 4A reference
  (`dmrg.py`) exactly; they are local pure-einsum helpers (the module does NOT
  import `dmrg`).

## Next-step suggestions

- Larger N (N=8,10) smoke once GPU/opt-in approved; two-site AD should track
  exact up to the chosen `max_bond_dim`.
- Subspace expansion / noise insertion (a la two-site DMRG) is OUT of scope for
  the AD mainline; the bond-growth lever already provides the entanglement
  freedom.
- Consider a mixed Adam+LBFGS schedule per bond if conditioning degrades at
  larger N.
