# Stage 7C Progress Log — Spinful Hubbard Chain (Jordan-Wigner)

## Goal

Add 1D open-boundary spinful Hubbard chain support to `latticeTN` on top of
the unchanged AD mainline, reusing the Stage 7B `model_builder` /
benchmark-registry / unified V100/TITAN V GPU selector. No TDVP /
finite-temperature / graded fermionic tensors / long-range models.

## Model and conventions (fixed)

```
H = -t  sum_{i,sigma} (c^d_{i sigma} c_{i+1,sigma} + h.c.)
    + U  sum_i (n_{i up} - 1/2)(n_{i down} - 1/2)
    - mu sum_i (n_{i up} + n_{i down} - 1)
    - h  sum_i (n_{i up} - n_{i down})
```

- Local basis `|0>`, `|up>`, `|down>`, `|up,down>` (indices 0,1,2,3), d=4.
- Global mode ordering **site-major**: `(0_up,0_down,1_up,1_down,...)`,
  2N modes; up of site i = global mode `2*i`, down = `2*i+1`.
- On-site JW order: `up` first, `down` second; the local 4x4 `cdown` /
  `cdagdown` carry `F_up` internally (so the on-site CARs hold). Per-site
  parity `P = F_up x F_down = diag(1,-1,-1,1)`.
- Hopping: the two global factors' JW strings cancel on sites 0..i-1; the
  only surviving parity is one `P` on site `i` from the right factor. The
  dense reference and the MPO both emit the left factor as `(c^d_sigma @ P)`
  (for `c^d_i c_{i+1}`) and `(P @ c_sigma)` (for the h.c. `c^d_{i+1} c_i`);
  these differ by a sign (P anticommutes with c/c^d) — required for
  Hermiticity.
- `torch.complex128`, open boundary, CPU-default.

## Changes (files)

New files:
- `tests/test_hubbard_operators.py` — local CAR algebra, basis actions,
  global 2N-mode JW anticommutation, site-level vs full-JW factorization.
- `tests/test_hubbard_dense.py` — Hermitian, free-fermion E0, atomic limit,
  high-field limit, particle-hole symmetry, matches full 2N-mode JW build,
  not a hard-core-boson build.
- `tests/test_hubbard_mpo_dense.py` — MPO.to_dense == hubbard_dense for
  N=2..4 across many (t,U,mu,h); ground energy; shapes D=6/d=4; Hermitian.
- `tests/test_hubbard_native_energy.py` — native Rayleigh == dense;
  differentiable; scale-invariant.
- `tests/test_hubbard_observables.py` — densities / double-occ / sz / NN
  spin-resolved hopping on known states + ED cross-check; dense vs MPS agree.
- `tests/test_model_builder_hubbard.py` — preset alignment, MPO==dense,
  native Rayleigh, not hard-core-boson.
- `tests/test_hubbard_ad_solvers.py` — global/one-site/two-site AD lower
  energy + not below ground; weak-interaction; two-site approaches ED at N=2.
- `tests/test_hubbard_gpu_timing.py` — CPU/GPU parity + timing
  (clean-skip when no V100/TITAN V).
- `scripts/run_hubbard_benchmark.py` — CPU/GPU benchmark of the three AD
  mainline solvers on the Hubbard chain, with ED reference.
- `scripts/hubbard_score.py` — Stage 7C score: runs the 8 test files,
  regenerates `docs/HUBBARD_REPORT.md`, checks required terms.
- `docs/HUBBARD_SPEC.md`, `docs/HUBBARD_PROTOCOL.md`, `docs/HUBBARD_REPORT.md`,
  this file.

Modified files:
- `latticetn/fermion_operators.py` — added `hubbard_local_operators(dtype,
  device)` → `{I, cup, cdagup, cdown, cdagdown, nup, ndown, ntot, sz,
  double_occ, parity}` in the standard Hubbard basis (on-site JW order
  up-first, down carries F_up; permuted to standard basis).
- `latticetn/operators.py` — added `hubbard_dense(N,t,U,mu,h,dtype,device)`
  with the explicit global JW parity (site-level standard-basis build,
  cross-checked against the full 2N-mode `_jw_global_mode` build); helpers
  `_jw_global_mode`, `_global_hubbard`, `_global_local4`.
- `latticetn/mpo.py` — added `MPO.generate_hubbard(t,U,mu,h)`, bond dim 6,
  local d=4, no separate parity-carrying state (the inter-site parity cancels
  in product; the surviving site-i parity is in the `@P`/`P@` left factors);
  `to_dense` matches `hubbard_dense`.
- `latticetn/model_builder.py` — added `hubbard_model(N,t,U,mu,h)` preset;
  `build_dense` / `build_mpo` dispatch to `hubbard_dense` /
  `MPO.generate_hubbard`.
- `latticetn/observables.py` — Hubbard observables: `dense_hubbard_local_density`
  (up/down/tot), `dense_hubbard_double_occ`, `dense_hubbard_local_sz`,
  `dense_hubbard_nn_hopping` (spin-resolved), and MPS variants; helper
  `_hubbard_global_two_site`.
- `latticetn/__init__.py` — export `hubbard_dense`, `hubbard_local_operators`.
- `docs/USER_GUIDE.md`, `docs/USER_GUIDE.zh-CN.md`, `docs/API_OVERVIEW.md`,
  `docs/INDEX.md`, `ROADMAP.md`, `REPO_STATUS.md` — Stage 7C sections added.

## Validation loop

- Smallest relevant pytest target per layer:
  `tests/test_hubbard_operators.py` → `tests/test_hubbard_dense.py` →
  `tests/test_hubbard_mpo_dense.py` → `tests/test_hubbard_native_energy.py` →
  `tests/test_model_builder_hubbard.py` → `tests/test_hubbard_ad_solvers.py`.
- `python scripts/hubbard_score.py --fast` at checkpoint (CPU default).
- `LATTICETN_RUN_GPU=1 python scripts/hubbard_score.py --fast` for the GPU
  parity + timing (clean-skip if no V100/TITAN V).
- Core scores regression: `python scripts/model_builder_score.py --fast` and
  `python scripts/fermion_score.py --fast` (CPU + GPU) still pass.

## Key design notes

- The Hubbard MPO is D=6 (NOT D=7 with a parity-carrying state, unlike the
  spinless-fermion MPO). The reason: in the spinful chain the two global
  factors of a hop BOTH carry the per-site parity on sites 0..i-1 (because
  the site-major global mode index of any spin on site i+1 exceeds all modes
  on sites 0..i), so the strings square to identity and cancel in the
  product. Only a single `P` on site `i` survives (from the right factor),
  and it is absorbed into the left-factor emit `(c^d @ P)` / `(P @ c)`.
- The h.c. term uses `(P @ c)` while the forward term uses `(c^d @ P)`; they
  differ by a sign and this asymmetry is REQUIRED for Hermiticity (verified:
  `t1+t2` is Hermitian and matches `hubbard_dense`).
- The dense reference builds the global operators with the full 2N-mode JW
  string (via `_jw_global_mode`) AND with the site-level standard-basis
  factorization (via `_global_hubbard`); the two are algebraically identical
  and cross-checked in `test_hubbard_dense.py::test_dense_matches_full_2n_mode_jw_build`
  and `test_hubbard_operators.py::test_site_level_standard_basis_factorization_matches_full_jw`.

## Items not run

- No long benchmarks (only the `--fast` preset: N=4, chi=4/8).
- No N>6 (the d=4 dense reference is expensive at N=5,6; dense/MPO alignment
  is verified at N=2..4 per the Stage 7C spec; ED ground-energy and
  free-fermion E0 checks cover N up to 4 for the dense build).
- No TDVP, finite-temperature, graded fermionic tensors, or long-range
  models (out of scope by the hard constraints).
