# Stage 5B — Two-Site AD Local Tensor Optimization Protocol

> Operational protocol and validation steps for `latticetn/ad_two_site.py`.
> Two-site AD is the AD mainline. SVD/QR are split/compression/stabilization
> ONLY, **not the solver**.

## 1. Scope

Validate the two-site AD local-tensor optimizer end-to-end on the open-boundary
1D spin-1/2 Heisenberg chain, against exact diagonalization, one-site AD,
global AD-MPS, and classical DMRG (reference only). All paths CPU-only,
`torch.complex128`, small systems (N=4,6) for `--fast`.

## 2. Required files

- Module: `latticetn/ad_two_site.py`
- Runner: `scripts/run_ad_two_site.py`
- Score: `scripts/ad_two_site_score.py`
- Docs: `docs/AD_TWO_SITE_SPEC.md`, `docs/AD_TWO_SITE_PROTOCOL.md`,
  `docs/AD_TWO_SITE_REPORT.md`, `docs/CLAUDE_PROGRESS_AD_TWO_SITE.md`
- Tests:
  - `tests/test_ad_two_site_loss.py`
  - `tests/test_ad_two_site_gradients.py`
  - `tests/test_ad_two_site_split.py`
  - `tests/test_ad_two_site_sweep_smoke.py`
  - `tests/test_ad_two_site_vs_one_site.py`
  - `tests/test_ad_two_site_policy.py`

## 3. Command

```bash
python scripts/ad_two_site_score.py --fast
```

The score script:
1. runs the six required test files with `pytest -q`;
2. runs `scripts/run_ad_two_site.py --markdown-output docs/AD_TWO_SITE_REPORT.md`
   to regenerate the report;
3. checks the report contains all required sections/terms;
4. prints `AD two-site score: PASS` and exits 0 on success.

## 4. Test coverage (what is checked)

- **loss**: `E(Θ)` is a real, finite scalar with `requires_grad=True`; only `Θ`
  is trainable (the rest frozen); `Θ` is a 4-axis two-site block;
  scale-invariant in `Θ`; equals the global Rayleigh quotient in mixed-canonical
  form; `reset_bond` preserves the global energy (gauge invariance).
- **gradients**: `backward()` populates a non-None, finite, non-zero `Θ.grad`;
  one LBFGS step does not increase the energy; one Adam step does not strongly
  increase it; autograd gradient matches a real-part finite-difference.
- **split**: full-rank split preserves the dense two-site state (fidelity ~ 1);
  `max_bond_dim` is respected; truncation error is non-negative, finite, in
  `[0,1]`; `direction='right'` gives a left-canonical `A_i`,
  `direction='left'` gives a right-canonical `A_{i+1}`; cutoff drops tiny
  singular values; `split` writes back into the MPS and the global energy does
  not increase.
- **sweep smoke**: N=4 and N=6 sweeps lower the energy; final energy within
  tolerance of exact and NOT below ground beyond tolerance; sweep direction
  alternates right/left; truncation errors non-negative/finite; bond dim
  respects the cap.
- **vs one-site**: agrees with one-site AD, global AD-MPS, and DMRG reference
  within tolerance; consistent with exact (variational, no below-ground); bond
  growth reaches the exact energy.
- **policy (AST)**: module imports neither `dmrg` nor `lanczos`; the loss path
  is clean (no `no_grad`/`detach`/`.data`/`.item`/`eigh`/`svd`/`qr`, no call
  into split/canonicalization helpers); `eigh` appears nowhere; `qr`/`svd` live
  only in the marked helpers; the driver drives optimization via `backward()`
  + `optimizer.step()` and never calls `dmrg`/`lanczos`/`eigh`.

## 5. Report requirements (`docs/AD_TWO_SITE_REPORT.md`)

Must contain: mainline statement; stabilization policy (SVD/QR are not the
solver); exact comparison; one-site AD comparison; global AD-MPS comparison;
DMRG reference comparison; energy history (per sweep, with direction);
gradient check; bond growth / compression; truncation errors; overall
pass/fail; known limitations. Must explicitly state that **two-site AD is the
mainline** and that **SVD/QR are split/compression/stabilization, not the
solver**.

## 6. Tolerances

N=4: `abs_err_vs_exact < 1e-8`; N=6: `< 1e-5`. No `below_ground` beyond `1e-6`.
These match the one-site AD tolerances; they are NOT weakened for this stage.

## 7. Physics conventions

Unchanged: `H = J * sum_i S_i.S_{i+1}`, `S = sigma/2`, `J = 1.0`, open
boundary, `complex128`. No silent switch between `S` and `sigma`.
