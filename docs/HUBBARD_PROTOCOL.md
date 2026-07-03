# Stage 7C — Spinful Hubbard Chain Protocol

> Operational protocol and validation steps for `scripts/hubbard_score.py`
> and `scripts/run_hubbard_benchmark.py`. The benchmark measures the
> **AD mainline solvers** on the spinful Hubbard chain on CPU and (opt-in) a
> V100/TITAN V GPU. Exact diagonalization is a **reference baseline only**.

## 1. Scope

Benchmark the three AD mainline solvers (global AD-MPS, one-site AD, two-site
AD) on the open-boundary 1D spinful Hubbard chain, comparing CPU and (opt-in)
a V100/TITAN V GPU for numerical parity and runtime/speedup. All default runs
are CPU-only, `torch.complex128`, small systems (N=4, chi=4/8) for `--fast`.
GPU is opt-in via `LATTICETN_RUN_GPU=1` and uses the unified GPU selector
(`scripts/gpu_selector.py`) which picks a V100/TITAN V when present.

## 2. Required files

- Local ops: `latticetn/fermion_operators.py` (`hubbard_local_operators`)
- Operators: `latticetn/operators.py` (`hubbard_dense`, `_jw_global_mode`)
- MPO: `latticetn/mpo.py` (`MPO.generate_hubbard`)
- Model builder: `latticetn/model_builder.py` (`hubbard_model`)
- Observables: `latticetn/observables.py` (Hubbard observables)
- Runner: `scripts/run_hubbard_benchmark.py`
- Score: `scripts/hubbard_score.py`
- GPU selector: `scripts/gpu_selector.py`
- Docs: `docs/HUBBARD_SPEC.md`, `docs/HUBBARD_PROTOCOL.md`,
  `docs/HUBBARD_REPORT.md`, `docs/CLAUDE_PROGRESS_HUBBARD.md`,
  `docs/GPU_TESTING_PROTOCOL.md` (updated)
- Tests:
  - `tests/test_hubbard_operators.py`
  - `tests/test_hubbard_dense.py`
  - `tests/test_hubbard_mpo_dense.py`
  - `tests/test_hubbard_native_energy.py`
  - `tests/test_hubbard_observables.py`
  - `tests/test_model_builder_hubbard.py`
  - `tests/test_hubbard_ad_solvers.py`
  - `tests/test_hubbard_gpu_timing.py`

## 3. Commands

```bash
# CPU-only (default; always works, no GPU required):
python scripts/hubbard_score.py --fast

# GPU benchmark (opt-in; runs only if LATTICETN_RUN_GPU=1 AND a V100/TITAN V
# is selected by the unified gpu_selector):
LATTICETN_RUN_GPU=1 python scripts/hubbard_score.py --fast

# List required files:
python scripts/hubbard_score.py --list
```

The score script:
1. runs `scripts/run_hubbard_benchmark.py --markdown-output
   docs/HUBBARD_REPORT.md --json-output <path>` to regenerate the report and
   JSON results;
2. runs the eight required test files with `pytest -q` (CPU-only env by
   default; GPU visible when opted in);
3. checks the report contains all required sections/terms;
4. prints `Hubbard score: PASS` and exits 0 on success.

A clean GPU skip (no `LATTICETN_RUN_GPU=1`, or no matching V100/TITAN V) is a
**successful exit 0**: the CPU benchmark still runs and the report records the
skip reason.

## 4. Test coverage (what is checked)

- **operators** (`test_hubbard_operators.py`): same-spin and cross-spin CARs
  (`{c_s,c^d_s}=I`, `{c_up,c_down}=0`, etc.); number/sz/double-occ/parity
  diagonals; parity anticommutes with c/c^d; c/c^d actions on the basis
  (including the JW sign on `c_down |updown>`); global 2N-mode JW
  anticommutation on different (site, spin) modes; the site-level
  standard-basis factorization matches the full 2N-mode JW build.
- **dense** (`test_hubbard_dense.py`): Hermitian; free-fermion (U=0) E0
  matches `2 * sum of negative single-particle levels`; atomic limit
  (t=0, U=4, N=2) E0 = -U/2; high-field (t=0, h=5) E0 = -h*N; particle-hole
  symmetry at half filling (trace 0 at mu=0); matches the explicit full-2N-mode
  JW build; differs from a no-parity hard-core-boson build for all N>=2.
- **MPO-dense** (`test_hubbard_mpo_dense.py`): `to_dense` of the Hubbard MPO
  matches `hubbard_dense` for N=2..4 across many `(t,U,mu,h)`; ground energy
  matches ED; shapes/bond dim (D=6, d=4); Hermitian; t and U scaling.
- **native energy** (`test_hubbard_native_energy.py`): native Rayleigh energy
  == dense energy on a random MPS; matches the Stage-1 path; differentiable;
  scale-invariant.
- **observables** (`test_hubbard_observables.py`): local densities / double
  occ / sz on known basis states; NN spin-resolved hopping on a one-electron
  delocalized state (gives 1.0) and cross-checked against the ED ground state
  (`sum_s hop_{i,s} = -E/t`); dense and MPS variants agree on a random MPS.
- **model_builder** (`test_model_builder_hubbard.py`): `hubbard_model` preset
  dense == `hubbard_dense`; statistics is `fermion`; terms are fermionic;
  ground energy matches ED; MPO.to_dense == build_dense; native Rayleigh ==
  dense; not a hard-core-boson build.
- **AD solvers** (`test_hubbard_ad_solvers.py`): global/one-site/two-site AD
  each LOWER the energy and do NOT undershoot the exact ground beyond
  tolerance (`below_ground=False`); weak-interaction regime; two-site AD
  approaches ED for small N (N=2 free Hubbard).
- **GPU timing** (`test_hubbard_gpu_timing.py`): the selector respects the
  V100/TITAN V filter (no fallback); CPU baseline always runs; when a matching
  GPU is present, CPU/GPU final energies agree within tolerance and the GPU
  energy does not undershoot the exact ground. Runtime/speedup recorded.

## 5. Report requirements (`docs/HUBBARD_REPORT.md`)

Must contain: mainline statement (AD mainline benchmarked; ED is reference
baseline only, not the solver; AD loss path unchanged — only the
Hamiltonian/MPO/operator layer is new); device info (GPU name, CUDA version,
PyTorch version, device, dtype, allowed filters, all/matched GPUs); exact
reference; CPU/GPU comparison table per solver (final energy, energy error,
energy per bond, runtime, speedup, below-ground flag); overall pass/fail;
known limitations. Must explicitly state this is **1D Jordan-Wigner fermions,
NOT graded fermionic tensors**, that the **JW parity string is the key**, and
the **site-major** global ordering.

## 6. Tolerances

- CPU/GPU final-energy agreement per AD solver: `|E_cpu - E_gpu| < 1e-6` for
  N=4 (matching the existing AD tolerances; not widened).
- GPU (and CPU) final energy must not undershoot the exact ground energy by
  more than `1e-6` (`below_ground=False`).
- Runtime and speedup must be **recorded**; the GPU is **not** required to be
  faster, because small systems are overhead-dominated.

## 7. Physics conventions

Open-boundary spinful Hubbard chain, d=4, `torch.complex128`, local basis
`|0>,|up>,|down>,|up,down>`, site-major global ordering
`(0_up,0_down,1_up,1_down,...)`. ED is CPU-only; the GPU runs only the AD
solver optimization. This is 1D Jordan-Wigner fermions, not a full graded
fermionic tensor network. No silent switch between fermion and spin
conventions (the Hubbard module never mixes with `spin_operators`; the
spinless-fermion and Heisenberg conventions are unchanged).

## 8. Stop conditions

Stage 7C is complete when:
1. The core scores still pass (`model_builder_score.py --fast`,
   `fermion_score.py --fast`, both CPU and GPU).
2. `python scripts/hubbard_score.py --fast` exits 0.
3. Hubbard dense, MPO-to-dense, model_builder, and native energy all align.
4. The AD solvers lower the energy on the Hubbard chain and do not undershoot
   the exact ground.
5. A V100/TITAN V GPU, if present, gives CPU/GPU parity + timing; otherwise
   the GPU portion clean-skips.
6. The report lists modified files, API, commands, CPU/GPU timing, items not
   run, and a suggested commit command.

## 9. Hard constraints

- No TDVP, no finite-temperature, no graded fermionic tensors, no long-range
  models.
- No new large dependencies; no widening of existing thresholds; no change to
  Heisenberg or spinless-fermion conventions; no long benchmarks; no git
  mutation commands.
