# Claude Progress Log: Stage 3A Canonicalization

## Checkpoint 2026-07-01 — Stage 3A complete

```text
Date/time: 2026-07-01
Stage: Stage 3A MPS canonicalization + compression (final)
Files changed (NEW):
  - latticetn/canonical.py            : canonical-form core
        left_canonical, right_canonical, mixed_canonical,
        canonical_norm, center_frob_norm,
        svd_compress (truncation-error reporting),
        entanglement_entropy (canonical SVD),
        from_dense (successive SVDs),
        left_orthonormal_error / right_orthonormal_error / *_all diagnostics
  - tests/test_mps_canonicalization.py   : left/right/mixed orthonormality+fidelity+norm
  - tests/test_mps_compression.py        : bond cap, exact recovery, energy error + variational bound
  - tests/test_canonical_entanglement.py : canonical vs dense entropy, product=0, Bell=ln2
  - tests/test_canonical_score.py        : canonical_score.py --list smoke
  - scripts/canonical_score.py           : --fast / --list
  - scripts/run_canonical_smoke.py       : writes docs/CANONICALIZATION_REPORT.md
  - docs/CANONICALIZATION_SPEC.md
  - docs/CANONICALIZATION_PROTOCOL.md
  - docs/CANONICALIZATION_REPORT.md      (generated)
  - docs/CLAUDE_PROGRESS_CANONICAL.md    (this file)
Files changed (MODIFIED, non-breaking):
  - latticetn/mps.py : added MPS.from_tensors classmethod (wraps canonical-form
        tensors, requires_grad=False default; does NOT touch energy_with_MPO).
Commands run (and results):
  - python scripts/validation_score.py --fast   -> exit 0 (PASS, 36 Stage1 tests)
  - python scripts/benchmark_score.py --fast     -> exit 0 (PASS)
  - pytest -q tests/test_mps_canonicalization.py tests/test_mps_compression.py
            tests/test_canonical_entanglement.py tests/test_canonical_score.py
                                                 -> 12 passed
  - python scripts/canonical_score.py --fast     -> exit 0 (Canonical score: PASS)
Numerical highlights (from docs/CANONICALIZATION_REPORT.md):
  - left/right/mixed orthonormality errors ~6.7e-16 / 8.9e-16; fidelity 1.0;
    canonical norm == dense norm (21.4820594439).
  - canonical entanglement entropy matches dense SVD reference to ~1e-16 at
    every cut (N=5).
  - Compression of exact N=6 Heisenberg ground state:
      * chi=8 (= full bond) exact: energy err 4.4e-16, bond dims [2,4,8,4,2].
      * chi=2 truncation: energy -1.96020084 (err 0.533 vs E0=-2.4936),
        bond dims [2,2,2,2,2], still satisfies variational bound (>= E0 - tol).
Differentiability note:
  - canonical.py operates under torch.no_grad on detached tensors and wraps via
    MPS.from_tensors(requires_grad=False). No .detach()/.data/unnecessary .item()
    added to energy_with_MPO. The differentiable energy path is unchanged.
Current failing item: none
Next action / suggestions:
  - Stage 3B (optional): replace the dense-reference MPS observables in
    observables.py (mps_expect_*, mps_entanglement_entropy) with canonical-MPS
    contractions so observables scale beyond small N. The canonical-form
    entropy already proves this path; the local/two-site observables can reuse
    the mixed-canonical center.
  - Optional: two-site DMRG using svd_compress + mixed_canonical (explicitly
    out of scope for 3A, but the primitives now exist).
  - Optional: integrate canonical entanglement entropy into the Stage 2
    benchmark report alongside the dense reference.
Notes:
  - conventions unchanged: S=sigma/2, J=1.0, open boundary, complex128.
  - No DMRG / TEBD / GPU performance benchmark in this stage.
  - GPU-readiness files untouched; canonical tests are CPU-only and not in any
    default GPU or validation path.
```
