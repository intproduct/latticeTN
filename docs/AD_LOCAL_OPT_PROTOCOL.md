# Stage 5A — AD Local Tensor Optimization Protocol

## Pre-conditions

- `docs/AD_MAINLINE_AUDIT.md` PASS (no drift from the AD mainline).
- Existing regression scores all exit 0:
  `validation_score --fast`, `benchmark_score --fast`, `canonical_score --fast`,
  `contraction_score --fast`, `ad_variational_score --fast`.

## Forbidden / allowed (mirror `docs/AD_MAINLINE_POLICY.md`)

FORBIDDEN in the main optimization path (loss + backward + optimizer.step):
- `lanczos`, any `eigh` / classical local eigensolver
- `dmrg.two_site_sweep` / classical DMRG two-site update
- using SVD **as the solver**
- using canonicalization **as the optimizer**

ALLOWED (must be outside the loss graph, `no_grad` mutating `.data`):
- SVD / QR / canonicalization as post-step stabilization / projection
- QR / SVD as orthogonality-center movement / compression
- classical DMRG as reference baseline (its own score scripts only)
- dense exact diagonalization as a small-N reference

## Files to produce

- `latticetn/ad_local.py` — `ADLocalOptimizer` + `train_ad_local`.
- `tests/test_ad_local_opt_loss.py` — loss is a differentiable scalar,
  scale-invariant, autograd-clean.
- `tests/test_ad_local_opt_gradients.py` — backward populates the center
  tensor's grad (finite), grads are zero/absent on frozen tensors.
- `tests/test_ad_local_opt_step.py` — a local step lowers the energy; a full
  sweep lowers it further; energy never goes below exact by > tolerance.
- `tests/test_ad_local_opt_vs_global_ad.py` — AD-local and global AD-MPS reach
  the same variational minimum (within tol); both at/above exact.
- `tests/test_ad_local_opt_policy.py` — AST guard: loss path free of
  dmrg/lanczos/eigh/svd/qr/no_grad/detach/.data/item; module does not import
  dmrg/lanczos.
- `scripts/run_ad_local_opt.py` — generates `docs/AD_LOCAL_OPT_REPORT.md`.
- `scripts/ad_local_opt_score.py` — `--fast` runs the tests + the runner and
  checks the report contains required sections.

## Score command

```bash
python scripts/ad_local_opt_score.py --fast
```

## Report must contain

`docs/AD_LOCAL_OPT_REPORT.md`:
1. exact comparison (small N, with tolerances);
2. global AD-MPS comparison;
3. DMRG reference comparison;
4. energy history (sampled);
5. gradient check (center-tensor grad finite);
6. stabilization setting (which option was used);
7. pass/fail;
8. known limitations;
9. explicit statement: this stage's mainline is AD local tensor optimization;
   SVD/QR/canonicalization are optional stabilization, NOT the solver.

## Stop conditions

1. existing regression scores all exit 0;
2. `python scripts/ad_local_opt_score.py --fast` exits 0;
3. `docs/AD_LOCAL_OPT_REPORT.md` contains all required sections;
4. docs declare the mainline / stabilization policy explicitly;
5. `docs/CLAUDE_PROGRESS_AD_LOCAL_OPT.md` records changes, commands, results,
   next-step suggestion.

Pause-and-ask if the variational energy falls below exact ground energy by more
than tolerance, or a dependency beyond torch/numpy/scipy/pytest/tqdm/matplotlib
seems necessary.
