# Claude Progress Log: Stage 4R AD-MPS Variational Solver

## Checkpoint 2026-07-01 — Stage 4R complete

```text
Date/time: 2026-07-01
Stage: Stage 4R AD-MPS autograd variational solver (the AD mainline; final)
Files changed (NEW):
  - latticetn/ad_variational.py : ADVariationalMPS (trainable nn.Parameter MPS)
        + differentiable loss = contractions.rayleigh_energy_native;
        train_ad_mps with Adam/LBFGS; per-tensor L2 renormalization under no_grad
        OUTSIDE the loss graph (scale-invariant, like Stage 1 _full_normalize);
        history: energy / grad norm / norm / max bond.
  - tests/test_ad_variational_loss.py     : scalar/finite/requires_grad, real,
        scale-invariant, AST source-clean (no detach/.data/item/no_grad in loss)
  - tests/test_ad_variational_gradients.py: backward -> all grads not-None+finite;
        finite-difference consistency
  - tests/test_ad_mps_optimizer_smoke.py  : Adam lowers energy; N=4/6 vs exact
        (tol N4<1e-6, N6<1e-3 at 200 steps); LBFGS runs; history metadata
  - tests/test_ad_vs_dmrg_reference.py    : AD close to DMRG ref (<1e-3); AD
        module does NOT import dmrg/lanczos into the loss path
  - tests/test_ad_variational_score.py    : ad_variational_score.py --list smoke
  - scripts/ad_variational_score.py       : --fast / --list
  - scripts/run_ad_mps_heisenberg.py      : writes docs/AD_VARIATIONAL_REPORT.md
  - docs/AD_VARIATIONAL_SPEC.md
  - docs/AD_VARIATIONAL_PROTOCOL.md
  - docs/AD_VARIATIONAL_REPORT.md         (generated)
  - docs/CLAUDE_PROGRESS_AD_VARIATIONAL.md (this file)
Files changed (MODIFIED): none required — reuses contractions.rayleigh_energy_native
  (unchanged) and the Stage 1 MPS nn.Parameter tensors. mps.py/contractions.py untouched.
Commands run (and results):
  - validation / benchmark / canonical / contraction / dmrg _score --fast
        -> all exit 0
  - pytest -q (5 Stage 4R test files) -> 15 passed
  - python scripts/ad_variational_score.py --fast -> exit 0 (AD variational score: PASS)
  - GPU smoke (matched GPU) re-run for regression -> exit 0 (intact)
Numerical highlights (from docs/AD_VARIATIONAL_REPORT.md, Adam, seed 0, 200 steps):
  - N=4 chi=8: init -0.078 -> final -1.6160253893 vs exact -1.6160254038
        (abs err 1.45e-8 < tol 1e-6), below_ground False, grad all not-None+finite.
  - N=6 chi=8: init -0.386 -> final -2.4933160938 vs exact -2.4935771339
        (abs err 2.61e-4 < tol 1e-3), below_ground False, grad all not-None+finite.
        (600-step Adam reaches 2.9e-14 = machine precision, confirmed in dev.)
  - DMRG reference N=6: -2.4935771339 (== exact). |AD - DMRG| = 2.61e-4 < 1e-3.
  - LBFGS runs and lowers energy (N=4, 15 iters -> 2e-10 vs exact in dev).
Autograd mainline note:
  - loss = contractions.rayleigh_energy_native(mps, mpo): fully differentiable,
    NO detach()/.data/unnecessary .item()/no_grad() in the loss path
    (AST source inspection enforces this).
  - per-tensor L2 renormalization after each step is under no_grad mutating .data
    (a scale-invariant stability projection; Rayleigh quotient unchanged) ->
    it does NOT touch the differentiable energy path.
  - AD mainline does NOT import or call dmrg/lanczos (structural guard test);
    Lanczos/DMRG are classical reference baselines only.
Reusability note:
  - the AD solver is built ENTIRELY on the Stage 3B native differentiable
    contraction + Stage 1 trainable MPS params; nothing new was needed at the
    tensor-contraction layer. The differentiability contract from Stage 3B is
    what makes the AD mainline possible.
Current failing item: none
Next action / suggestions:
  - Stage 5 (optional): make per-tensor renormalization a canonical-form
    projection (Stage 3A mixed_canonical) instead of plain L2, so the trained
    MPS stays near canonical form and observables/entropy are cheap to evaluate.
  - Optional: bond-growing AD (initialize small chi, increase when energy plateaus)
    using Stage 3A svd_compress as the grow step (non-differentiable projection
    outside the loss graph) to combine AD flexibility with low-chi efficiency.
  - Optional: LR scheduling / convergence-accelerated Adam for the N=6
    machine-precision regime without 600 steps.
  - Optional: expose the AD solver as the Stage 2 benchmark's variational path
    (currently Stage 2 uses run_heisenberg_small's Adam; consolidating would
    unify the variational reference).
Notes:
  - conventions unchanged: S=sigma/2, J=1.0, open boundary, complex128.
  - No TEBD/TDVP/finite-T/GPU benchmark; no long training.
  - GPU-readiness files untouched; Stage 4R tests are CPU-only and not in any
    default GPU / Stage1-2-3A-3B-4A-4B path.
```
