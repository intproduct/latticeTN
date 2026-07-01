# Stage 5A Gauge-Stabilized AD-MPS Protocol

## Stop condition

Stage 5A is complete only when ALL hold:

1. `python scripts/validation_score.py --fast` exits 0.
2. `python scripts/benchmark_score.py --fast` exits 0.
3. `python scripts/canonical_score.py --fast` exits 0.
4. `python scripts/contraction_score.py --fast` exits 0.
5. `python scripts/ad_variational_score.py --fast` exits 0.
6. `python scripts/ad_gauge_score.py --fast` exits 0.
7. `docs/AD_GAUGE_REPORT.md` contains:
   - projection comparison (none / tensor_norm / canonical),
   - energy history, grad norm, state norm, canonical error,
   - exact / DMRG reference,
   - pass/fail, known limitations.
8. The report explicitly states projection is a post-step, non-differentiable
   gauge stabilization, NOT part of the loss path.
9. `docs/CLAUDE_PROGRESS_AD_GAUGE.md` records changes, commands, results,
   next-step suggestions.

GPU smoke is NOT a blocking item; existing GPU-readiness files must stay intact
and default test paths must not depend on GPU.

## Files

New:

- `tests/test_ad_gauge_projection.py`
- `tests/test_ad_gauge_loss_integrity.py`
- `tests/test_ad_gauge_optimizer_smoke.py`
- `tests/test_ad_gauge_vs_baseline.py`
- `tests/test_ad_gauge_score.py`
- `scripts/ad_gauge_score.py`
- `scripts/run_ad_gauge_heisenberg.py`
- `docs/AD_GAUGE_SPEC.md`
- `docs/AD_GAUGE_PROTOCOL.md`
- `docs/AD_GAUGE_REPORT.md` (generated)
- `docs/CLAUDE_PROGRESS_AD_GAUGE.md`

Modified (non-breaking): `latticetn/ad_variational.py` — add `projection`
option to `train_ad_mps` (default `tensor_norm`, preserving Stage 4R behavior),
`_project`, `_canonical_error`, `_state_norm` helpers, and
`canonical_error_history` / `state_norm_history` / `projection` fields in the
result dict (with `norm_history` kept as a back-compat alias for Stage 4R).
`latticetn/canonical.py` / `latticetn/mps.py` may be touched but this stage was
additive; no interface break.

## Required implementation stages

### 5A.1 projection option

`_project(mps, projection)` with none / tensor_norm / canonical (Stage 3A
left-canonical QR written onto `.data` under `no_grad`); `_canonical_error`,
`_state_norm` diagnostics; `train_ad_mps(..., projection=...)` records them.

### 5A.2 tests

- projection: Rayleigh invariance, fidelity ~1, canon error drop, params stay
  trainable, none==identity, invalid raises.
- loss integrity: scalar/finite/requires_grad, all grads, AST loss-path clean
  of dmrg/lanczos/projection/detach/.data/no_grad/item.
- optimizer smoke: each projection lowers energy; N=4/6 not-below-ground within
  tol; canon error decreases; history records diagnostics.
- vs baseline: canonical not worse than tensor_norm; none not better than canonical.

### 5A.3 score + report

`scripts/ad_gauge_score.py --fast` runs tests then
`scripts/run_ad_gauge_heisenberg.py` writes `docs/AD_GAUGE_REPORT.md`.

## Pause conditions

Pause and report instead of forcing tests to pass if:

- canonical projection breaks the dense state beyond a global phase,
- projection loses differentiability of the loss path,
- training (any projection) undershoots exact ground beyond tolerance,
- canonical is materially worse than the tensor_norm baseline (would indicate a
  gauge bug),
- the autograd energy path is touched with detach/.data/unnecessary item,
- a new large dependency is needed.

## Hard constraints

- Projection runs ONLY after `optimizer.step()`, OUTSIDE the loss graph.
- No local eigensolver / Lanczos / DMRG sweep / projection / no_grad in the loss path.
- No `.detach()`/`.data`/unnecessary `.item()` in the differentiable energy path.
- No changes to Stage 1/2/3A/3B/4A/4B/4R physics conventions or thresholds.
- `energy_with_MPO` / `rayleigh_energy_native` not broken.
- No large new dependencies; no long training.
- GPU tests stay out of the default test path.
