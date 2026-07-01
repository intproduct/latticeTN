# Claude Progress Log: Stage 5A Gauge-Stabilized AD-MPS

## Checkpoint 2026-07-01 — Stage 5A complete

```text
Date/time: 2026-07-01
Stage: Stage 5A gauge-stabilized AD-MPS optimizer (canonical projection) (final)
Files changed (NEW):
  - tests/test_ad_gauge_projection.py     : Rayleigh invariance, fidelity~1,
        canon-error drop, params stay trainable, none==identity, invalid raises
  - tests/test_ad_gauge_loss_integrity.py : scalar/finite/requires_grad, all grads
        not-None+finite, AST loss path clean of dmrg/lanczos/projection/detach/
        .data/no_grad/item
  - tests/test_ad_gauge_optimizer_smoke.py: each projection lowers energy,
        N=4/6 not-below-ground within tol, canon error decreases, history records
  - tests/test_ad_gauge_vs_baseline.py    : canonical not worse than tensor_norm;
        none not better than canonical
  - tests/test_ad_gauge_score.py          : ad_gauge_score.py --list smoke
  - scripts/ad_gauge_score.py             : --fast / --list
  - scripts/run_ad_gauge_heisenberg.py    : writes docs/AD_GAUGE_REPORT.md (3-projection compare)
  - docs/AD_GAUGE_SPEC.md
  - docs/AD_GAUGE_PROTOCOL.md
  - docs/AD_GAUGE_REPORT.md               (generated)
  - docs/CLAUDE_PROGRESS_AD_GAUGE.md      (this file)
Files changed (MODIFIED, non-breaking):
  - latticetn/ad_variational.py : added `projection='none'|'tensor_norm'|'canonical'`
        to train_ad_mps (default 'tensor_norm' = Stage 4R behavior); _project,
        _canonical_error, _state_norm helpers; new history keys
        canonical_error_history / state_norm_history / projection; norm_history
        kept as back-compat alias for Stage 4R. Loss path UNCHANGED (still
        rayleigh_energy_native; no detach/.data/item/no_grad in loss).
Commands run (and results):
  - validation / benchmark / canonical / contraction / ad_variational _score --fast
        -> all exit 0
  - pytest -q (Stage 4R tests)            -> 15 passed (back-compat intact)
  - pytest -q (5 Stage 5A test files)     -> 18 passed
  - python scripts/ad_gauge_score.py --fast -> exit 0 (AD gauge score: PASS)
  - GPU smoke (matched GPU) re-run for regression -> exit 0 (intact)
Numerical highlights (from docs/AD_GAUGE_REPORT.md, Adam, seed 0, 200 steps):
  - Projection invariance (canonical): Rayleigh energy dE=1.6e-16; dense-state
    fidelity = 1.0; canonical error 1.01e+01 -> 6.66e-16; params remain trainable leaves.
  - N=4: none final=-1.615698 (err 3.3e-4); tensor_norm final=-1.6160253893 (err 1.45e-8);
         canonical final=-1.6160252074 (err 1.96e-7, canon err 4.4e-16).
  - N=6: none final=-2.483977 (err 9.6e-3); tensor_norm final=-2.493316 (err 2.61e-4);
         canonical final=-2.4935769821 (err 1.52e-7, canon err 6.7e-16).
  - canonical is the BEST gauge: N=6 err 1.5e-7 vs tensor_norm 2.6e-4 vs none 9.6e-3.
  - DMRG reference N=6 = -2.4935771339 (== exact); |AD canonical - DMRG| = 1.52e-7 < 1e-3.
  - Canonical error stays ~1e-16 throughout training under the canonical projection.
Differentiability / mainline note:
  - Loss = rayleigh_energy_native(mps, mpo), fully differentiable; backward on it only.
  - Projection runs AFTER optimizer.step() under torch.no_grad mutating .data
    (Stage 3A left_canonical written back onto live params) — a NON-differentiable
    gauge stabilization OUTSIDE the loss graph. Rayleigh quotient is gauge- and
    scale-invariant, so projection changes the gauge / conditioning, NOT the physics.
  - AST test confirms the loss path contains no projection/dmrg/lanczos/detach/.data/
    no_grad/item.
  - DMRG/Lanczos remain classical reference baselines only; never in the AD path.
Reusability note:
  - The gauge projection reuses the Stage 3A left_canonical QR sweep (verified
    state-preserving in Stage 3A) and the Stage 4R AD training loop; the only new
    idea is writing the canonical tensors back onto the live nn.Parameter .data so
    the optimizer's leaf reference is preserved.
Current failing item: none
Next action / suggestions:
  - Stage 5B (optional): mixed-canonical projection with a moving center (sweep the
    orthogonality center across bonds during training) to keep every site
    near-canonical, not just the left-canonical gauge.
  - Optional: bond-growing AD — initialize small chi, run AD+canonical gauge, then
    use Stage 3A svd_compress (or its reverse) to grow bonds when energy plateaus
    (non-differentiable projection, outside the loss graph).
  - Optional: study why canonical beats tensor_norm noticeably (the gauge appears
    to improve optimizer conditioning / reduce effective step noise); could justify
    canonical as the new default for the AD mainline.
  - Optional: expose projection in the Stage 2/4R benchmark variational path so
    the gauge comparison is visible there too.
Notes:
  - conventions unchanged: S=sigma/2, J=1.0, open boundary, complex128.
  - No TEBD/TDVP/finite-T/GPU benchmark; no long training.
  - GPU-readiness files untouched; Stage 5A tests are CPU-only and not in any
    default GPU / Stage1-2-3A-3B-4A-4B-4R path.
```
