# Stage 4R AD-MPS Variational Solver Protocol

## Stop condition

Stage 4R is complete only when ALL hold:

1. `python scripts/validation_score.py --fast` exits 0.
2. `python scripts/benchmark_score.py --fast` exits 0.
3. `python scripts/canonical_score.py --fast` exits 0.
4. `python scripts/contraction_score.py --fast` exits 0.
5. `python scripts/dmrg_score.py --fast` exits 0.
6. `python scripts/ad_variational_score.py --fast` exits 0.
7. `docs/AD_VARIATIONAL_REPORT.md` contains:
   - exact comparison (N<=6),
   - DMRG baseline comparison,
   - energy history,
   - gradient check,
   - optimizer settings,
   - pass/fail, known limitations.
8. The report explicitly states that Lanczos/DMRG are classical baselines,
   not the AD mainline.
9. `docs/CLAUDE_PROGRESS_AD_VARIATIONAL.md` records changes, commands,
   results, next-step suggestions.

GPU smoke is NOT a blocking item; existing GPU-readiness files must stay intact
and default test paths must not depend on GPU.

## Files

New:

- `latticetn/ad_variational.py` — `ADVariationalMPS`, `train_ad_mps`.
- `tests/test_ad_variational_loss.py`
- `tests/test_ad_variational_gradients.py`
- `tests/test_ad_mps_optimizer_smoke.py`
- `tests/test_ad_vs_dmrg_reference.py`
- `tests/test_ad_variational_score.py`
- `scripts/ad_variational_score.py`
- `scripts/run_ad_mps_heisenberg.py`
- `docs/AD_VARIATIONAL_SPEC.md`
- `docs/AD_VARIATIONAL_PROTOCOL.md`
- `docs/AD_VARIATIONAL_REPORT.md` (generated)
- `docs/CLAUDE_PROGRESS_AD_VARIATIONAL.md`

Modified (optional, non-breaking): `latticetn/mps.py`, `latticetn/contractions.py`.
This stage was additive; no modification was strictly required (it reuses the
existing `rayleigh_energy_native` and the Stage 1 `nn.Parameter` MPS tensors).

## Required implementation stages

### 4R.1 AD mainline module

`latticetn/ad_variational.py` with `ADVariationalMPS` (trainable params,
differentiable Rayleigh loss) and `train_ad_mps` (Adam/LBFGS, history,
renormalization outside the loss graph).

### 4R.2 tests

- loss: scalar/finite/requires_grad, real, scale-invariant, AST-clean source.
- gradients: backward populates all param grads (non-None, finite); finite-diff
  consistency.
- optimizer smoke: Adam lowers energy; N=4/6 vs exact within tolerances; LBFGS
  runs; history/metadata recorded.
- AD vs DMRG reference: AD close to DMRG; AD module does not import dmrg/lanczos
  into the loss path.

### 4R.3 score + report

`scripts/ad_variational_score.py --fast` runs tests then
`scripts/run_ad_mps_heisenberg.py` writes `docs/AD_VARIATIONAL_REPORT.md`.

## Pause conditions

Pause and report instead of forcing tests to pass if:

- the loss path cannot be made differentiable without `detach`/`.data`/`no_grad`,
- `loss.backward()` fails to populate all param grads after two focused attempts,
- AD energy undershoots exact ground beyond tolerance,
- AD and DMRG references disagree beyond tolerance after a documented step bump,
- a new large dependency is needed,
- runtime becomes too high for CPU fast validation.

## Hard constraints

- No local eigensolver / Lanczos / DMRG sweep / `no_grad` in the main loss path.
- No `.detach()`/`.data`/unnecessary `.item()` in the differentiable energy path.
- No changes to Stage 1/2/3A/3B/4A/4B physics conventions or thresholds.
- `energy_with_MPO` / `rayleigh_energy_native` not broken.
- No large new dependencies; no long training.
- GPU tests stay out of the default test path.
