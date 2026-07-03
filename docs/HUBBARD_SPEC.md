# Stage 7C — Spinful Hubbard Chain (Jordan-Wigner) Spec

## Scope

Stage 7C adds the open-boundary 1D **spinful Hubbard chain** to `latticeTN`,
on top of the unchanged AD mainline. The model is:

```
H = -t  sum_{i,sigma} (c^d_{i sigma} c_{i+1,sigma} + h.c.)
    + U  sum_i (n_{i up} - 1/2)(n_{i down} - 1/2)
    - mu sum_i (n_{i up} + n_{i down} - 1)
    - h  sum_i (n_{i up} - n_{i down})
```

with local basis `|0>`, `|up>`, `|down>`, `|up,down>` (d=4), default
`torch.complex128`, open boundary.

**This is 1D Jordan-Wigner (JW) fermions, NOT a full graded fermionic tensor
network.** Fermionic operators are represented as tensor products of bosonic
4x4 (per-site) matrices with an explicit Jordan-Wigner parity structure. The
JW parity is the key object; without it the Hamiltonian would be a spin /
hard-core-boson Hamiltonian (wrong signs for hopping).

## Conventions (fixed)

- **Local basis** (per site, standard Hubbard ordering):

  | index | state       | (n_up, n_down) |
  |------:|-------------|----------------|
  |   0   | `|0>`       | (0, 0)         |
  |   1   | `|up>`      | (1, 0)         |
  |   2   | `|down>`    | (0, 1)         |
  |   3   | `|up,down>` | (1, 1)         |

  So `d = 4` and the dense matrix is `(4**N, 4**N)`.

- **Global mode ordering** is fixed to **site-major**:
  `(0_up, 0_down, 1_up, 1_down, ..., (N-1)_up, (N-1)_down)`. There are `2N`
  two-level (single-spin) modes; the up mode of site `i` is global mode `2*i`
  and the down mode of site `i` is global mode `2*i + 1`.

- **On-site Jordan-Wigner order** is `up` first, `down` second. The local
  4x4 down operators `cdown` / `cdagdown` carry the up-mode parity `F_up`
  internally (`cdown = F_up x c_down` in the JW-natural basis, then indexed
  to the standard Hubbard basis). This makes the on-site canonical
  anticommutation relations hold:
  `{c_up, c^d_up} = I`, `{c_down, c^d_down} = I`,
  `{c_up, c_down} = {c_up, c^d_down} = {c^d_up, c_down} = 0`.

- **Per-site JW parity** `P = F_up x F_down = (-1)^{n_up+n_down} =
  diag(1, -1, -1, 1)`. This is the parity carried across sites by the global
  operators.

- **Hopping**: the spin-resolved nearest-neighbor hop
  `c^d_{i sigma} c_{i+1, sigma}` is built as a product of two global
  operators, each carrying the per-site parity `P` on every site left of its
  factor. In the product, the parity strings on sites `0..i-1` square to
  identity and cancel; the **only surviving parity** is a single `P` on site
  `i` contributed by the right factor's string. The dense reference
  (`operators.hubbard_dense`) and the MPO (`MPO.generate_hubbard`) both
  capture this: the left factor is emitted as `(c^d_sigma @ P)` (for
  `c^d_i c_{i+1}`) or `(P @ c_sigma)` (for the h.c. `c^d_{i+1} c_i`); the two
  differ by a sign because `P` anticommutes with `c`/`c^d`, and this
  asymmetry is required for Hermiticity.

- The diagonal terms (Hubbard `U` interaction, chemical potential `mu`,
  field `h`) commute with the parity string and need no JW string.

- Exact diagonalization (`numpy.linalg.eigh`) is the reference baseline;
  DMRG/Lanczos remain available as classical baselines. None are in the AD
  path.

## What is added (and what is NOT)

Added (Stage 7C):
- `latticetn/fermion_operators.py::hubbard_local_operators` — local 4x4
  operators `I, cup, cdagup, cdown, cdagdown, nup, ndown, ntot, sz,
  double_occ, parity`.
- `latticetn/operators.py::hubbard_dense` — dense reference Hamiltonian with
  the explicit global JW parity (site-major, 2N modes). NOT a spin / hard-core
  boson H.
- `latticetn/operators.py::_jw_global_mode` — the full 2N-mode JW string
  builder (used as a cross-check of the site-level build).
- `latticetn/mpo.py::MPO.generate_hubbard` — bond-dim-6 fermionic Hubbard
  MPO (local d=4); `to_dense` matches `hubbard_dense`.
- `latticetn/model_builder.py::hubbard_model` — preset; `build_dense` /
  `build_mpo` dispatch to the validated generators.
- `latticetn/observables.py` — Hubbard observables: local
  `<n_up_i>/<n_down_i>/<n_tot_i>`, double occupancy `<n_up_i n_down_i>`,
  local `<S^z_i>`, NN spin-resolved hopping
  `<c^d_{i,s} c_{i+1,s} + h.c.>`; dense + MPS variants.
- `scripts/run_hubbard_benchmark.py` + `scripts/hubbard_score.py`.
- Tests + docs (this file, HUBBARD_PROTOCOL, HUBBARD_REPORT,
  CLAUDE_PROGRESS_HUBBARD, GPU_TESTING_PROTOCOL update, USER_GUIDE,
  USER_GUIDE.zh-CN, API_OVERVIEW, INDEX, ROADMAP, REPO_STATUS updates).

NOT added (hard constraints):
- No TDVP, no finite-temperature, no graded fermionic tensors, no long-range
  models.
- No change to the AD mainline / loss path (it is operator-agnostic; only the
  Hamiltonian/MPO/operator layer is new).
- No change to Heisenberg or spinless-fermion conventions or existing
  thresholds.
- No new large dependencies (torch/numpy/scipy/pytest only).

## AD mainline (unchanged)

The three AD mainline solvers run on the Hubbard MPO unchanged:
- global AD-MPS (`ad_variational.train_ad_mps`, Adam);
- one-site AD local (`ad_local.train_ad_local`, LBFGS);
- two-site AD local (`ad_two_site.train_ad_two_site`, LBFGS).

The differentiable loss is still `contractions.rayleigh_energy_native` —
operator-agnostic. Stage 7C only swaps the Hamiltonian/MPO.

## GPU rules (Stage 7C, reuses the Stage 7B unified selector)

- The **unified GPU selector** (`scripts/gpu_selector.py`) selects a GPU whose
  name contains `V100` or `TITAN V`/`Titan V`. No other GPU is used; there is
  no fallback to `cuda:0`.
- GPU is opt-in via `LATTICETN_RUN_GPU=1`.
- If no matching GPU is present (or GPU not opted in), the GPU portion
  **clean-skips** (exit 0; CPU results still recorded).
- If a matching GPU is present, Stage 7C runs CPU and GPU small-system tests
  and records: device name, dtype, N, chi, solver, final energy, exact error,
  runtime, speedup, below_ground.
- dtype default `torch.complex128`. The GPU is NOT required to be faster;
  speedup is recorded only to observe AD-TN GPU acceleration trends.

## Algebra summary (verified by `tests/test_hubbard_operators.py`)

Local (on-site) in the standard Hubbard basis:
- `n_up = diag(0,1,0,1)`, `n_down = diag(0,0,1,1)`, `n_tot = diag(0,1,1,2)`.
- `sz = diag(0, +1/2, -1/2, 0)`.
- `double_occ = diag(0,0,0,1)`.
- `parity = diag(1,-1,-1,1) = (-1)^{n_tot}`, `parity^2 = I`.
- `c_up |up> = |0>`, `c_up |updown> = |down>`;
  `c_down |down> = |0>`, `c_down |updown> = -|up>` (the last minus sign is
  the on-site JW sign, since the down mode is JW-ordered after the up mode).

Global (across sites, site-major ordering):
- Global `c_{i, up}` = `P_0 x ... x P_{i-1} x cup_i x I x ...`.
- Global `c_{i, down}` = `P_0 x ... x P_{i-1} x cdown_i x I x ...` (the
  intra-site `F_up` is already inside `cdown_i`).
- Global operators on different (site, spin) modes anticommute:
  `{c_g, c_h} = 0`, `{c_g, c^d_h} = 0` for `g != h`; `{c_g, c^d_g} = I`.
