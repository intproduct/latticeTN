# Stage 3B Native Contraction Protocol

## Stop condition

Stage 3B is complete only when ALL hold:

1. `python scripts/validation_score.py --fast` exits 0.
2. `python scripts/benchmark_score.py --fast` exits 0.
3. `python scripts/canonical_score.py --fast` exits 0.
4. `python scripts/contraction_score.py --fast` exits 0.
5. `docs/CONTRACTION_REPORT.md` contains:
   - native/dense comparison table (norm, local, two-site, bond energy),
   - MPO energy comparison (native vs Stage1 vs dense),
   - gradient check (native energy backward -> grads not None + finite),
   - scalability smoke (N=20, chi<=8, no to_dense, no ED),
   - pass/fail, known limitations.
6. `docs/CLAUDE_PROGRESS_CONTRACTION.md` records changes, commands, results,
   and next-step suggestions.

GPU smoke is NOT a blocking item; existing GPU-readiness files must stay intact
and default test paths must not depend on GPU.

## Files

New:

- `latticetn/contractions.py` — native contractions core.
- `tests/test_native_norm_contraction.py`
- `tests/test_native_observable_contractions.py`
- `tests/test_native_mpo_energy_contraction.py`
- `tests/test_contraction_scalability_smoke.py`
- `tests/test_contraction_score.py`
- `scripts/contraction_score.py`
- `scripts/run_contraction_smoke.py`
- `docs/CONTRACTION_SPEC.md`
- `docs/CONTRACTION_PROTOCOL.md`
- `docs/CONTRACTION_REPORT.md` (generated)
- `docs/CLAUDE_PROGRESS_CONTRACTION.md`

Modified (optional, non-breaking): `latticetn/observables.py`, `latticetn/mps.py`
(this stage can be additive only; no modification was strictly required).

## Required implementation stages

### 3B.1 core module

Implement `latticetn/contractions.py` with native norm / local / two-site /
bond-energy / correlation / MPO numerator / Rayleigh energy contractions.

### 3B.2 tests

- norm vs dense + vs Stage 1 overlap + gradient.
- observable contractions vs dense (incl. non-commuting order, i>j).
- MPO energy vs Stage 1 and dense; gradient check; requires_grad guard.
- scalability smoke N=20 chi<=8 (finite, shape, device/dtype, runtime, no
  to_dense).
- `contraction_score.py --list` smoke.

### 3B.3 score + report

`scripts/contraction_score.py --fast` runs the tests, then
`scripts/run_contraction_smoke.py` writes `docs/CONTRACTION_REPORT.md`.

## Pause conditions

Pause and report instead of forcing tests to pass if:

- native observables disagree with dense references after two focused attempts,
- the native energy path loses differentiability or breaks Stage 1 energy,
- a change would require `.detach()/.data`/unnecessary `.item()` in the
  differentiable energy path,
- a new large dependency is needed,
- scalability smoke becomes unstable or absurdly slow.

## Hard constraints

- No DMRG / TEBD / GPU performance benchmark.
- No enlarged system sizes beyond the N=20 scalability smoke; no long training.
- No changes to Stage 1/2/3A physics conventions or thresholds.
- No `.detach()/.data`/unnecessary `.item()` in the differentiable energy path.
- No large new dependencies.
- No broad refactor of legacy files.
- GPU tests stay out of the default test path.
