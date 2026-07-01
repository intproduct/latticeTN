# Claude Progress Log: Stage 3B Native Contractions

## Checkpoint 2026-07-01 — Stage 3B complete

```text
Date/time: 2026-07-01
Stage: Stage 3B native MPS/MPO contractions + scalable observables (final)
Files changed (NEW):
  - latticetn/contractions.py : native contraction core
        native_norm_sq / native_norm,
        native_local_expect, native_two_site_expect (caller-order, non-commuting, i>j),
        native_bond_energy_heisenberg, native_correlation,
        native_mpo_numerator, native_mpo_expectation, rayleigh_energy_native,
        _left_norm_env / _right_norm_env environment builders.
  - tests/test_native_norm_contraction.py        : norm vs dense/overlap + grad
  - tests/test_native_observable_contractions.py : local/two-site/bond/corr vs dense
  - tests/test_native_mpo_energy_contraction.py  : MPO energy vs stage1+dense + grad
  - tests/test_contraction_scalability_smoke.py  : N=20 chi<=8 finite/shape/dev/dtype, no to_dense/ED
  - tests/test_contraction_score.py              : contraction_score.py --list smoke
  - scripts/contraction_score.py                 : --fast / --list
  - scripts/run_contraction_smoke.py             : writes docs/CONTRACTION_REPORT.md
  - docs/CONTRACTION_SPEC.md
  - docs/CONTRACTION_PROTOCOL.md
  - docs/CONTRACTION_REPORT.md                   (generated)
  - docs/CLAUDE_PROGRESS_CONTRACTION.md          (this file)
Files changed (MODIFIED): none required (stage is additive; mps.py/observables.py untouched).
Commands run (and results):
  - python scripts/validation_score.py --fast    -> exit 0 (PASS)
  - python scripts/benchmark_score.py --fast     -> exit 0 (PASS)
  - python scripts/canonical_score.py --fast     -> exit 0 (PASS)
  - pytest -q (5 Stage 3B test files)            -> 20 passed
  - python scripts/contraction_score.py --fast   -> exit 0 (Contraction score: PASS)
  - GPU smoke (matched GPU) re-run for regression -> exit 0 (GPU-readiness intact)
Numerical highlights (from docs/CONTRACTION_REPORT.md, N=6 chi=4 seed=0):
  - Native vs dense reference abs diffs ~1e-12..1e-14 for norm^2, <Sz_i>,
    <Sz_i Sz_j>, and bond energy.
  - MPO energy: stage1 energy_with_MPO == native rayleigh (0 diff); native vs
    dense-state energy 1.4e-16.
  - Gradient check: native energy backward -> all MPS params grad not None,
    all finite, requires_grad True.
  - Scalability smoke N=20 chi<=8: energy finite, bond dims capped at 8,
    cpu complex128, runtime ~0.009s, NO to_dense / NO exact diagonalization.
Differentiability note:
  - The energy path (native_mpo_numerator / rayleigh_energy_native /
    native_norm*) uses NO .detach()/.data/unnecessary .item()/no_grad.
    Observable/report paths may use torch.no_grad() and are kept separate.
  - One subtle bug during development: an initial right-sweep einsum bound the
    bra leg to the ket environment (and vice versa), which gave correct norm
    but wrong local observables except the last site; fixed by binding
    A.conj() -> bra env, A -> ket env consistently. Caught by dense comparison.
Current failing item: none
Next action / suggestions:
  - Stage 3C (optional): a two-site optimization sweep combining the Stage 3A
    mixed-canonical + SVD compression with the Stage 3B native energy/MPO
    contraction (i.e., a minimal DMRG-like optimizer). The primitives now
    exist; this stage explicitly stayed out of scope.
  - Optional: rewire Stage 2 observables.py (mps_expect_* / mps_entanglement_*)
    to call the native Stage 3B contractions instead of to_dense for the
    observable path, then deprecate the dense-reference branch with a fallback.
  - Optional: expose a canonical-MPS-bond entanglement entropy that reuses
    both Stage 3A's mixed_canonical and Stage 3B's native environment.
Notes:
  - conventions unchanged: S=sigma/2, J=1.0, open boundary, complex128.
  - No DMRG / TEBD / GPU performance benchmark in this stage.
  - GPU-readiness files untouched; contraction tests are CPU-only and not in
    any default GPU / Stage1-2-3A path.
```
