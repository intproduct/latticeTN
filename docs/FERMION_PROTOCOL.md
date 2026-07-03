# Stage 7A — Spinless Fermion t-V Chain Protocol

> Operational protocol and validation steps for `scripts/fermion_score.py`
> and `scripts/run_spinless_fermion_benchmark.py`. The benchmark measures the
> **AD mainline solvers** on the spinless fermion t-V chain on CPU and (opt-in)
> a V100/TITAN V GPU. Exact diagonalization is a **reference baseline only**.

## 1. Scope

Benchmark the three AD mainline solvers (global AD-MPS, one-site AD, two-site
AD) on the open-boundary 1D spinless fermion t-V chain, comparing CPU and
(opt-in) a V100/TITAN V GPU for numerical parity and runtime/speedup. All
default runs are CPU-only, `torch.complex128`, small systems (N=4,6) for
`--fast`. GPU is opt-in via `LATTICETN_RUN_GPU=1` and uses the unified GPU
selector (`scripts/gpu_selector.py`) which picks a V100/TITAN V when present.

## 2. Required files

- Module: `latticetn/fermion_operators.py`
- Operators: `latticetn/operators.py` (`spinless_fermion_dense`)
- MPO: `latticetn/mpo.py` (`MPO.generate_spinless_fermion`)
- Observables: `latticetn/observables.py` (fermion observables)
- Runner: `scripts/run_spinless_fermion_benchmark.py`
- Score: `scripts/fermion_score.py`
- GPU selector: `scripts/gpu_selector.py`
- Docs: `docs/FERMION_SPEC.md`, `docs/FERMION_PROTOCOL.md`,
  `docs/FERMION_REPORT.md`, `docs/CLAUDE_PROGRESS_FERMION.md`,
  `docs/GPU_TESTING_PROTOCOL.md` (updated)
- Tests:
  - `tests/test_fermion_operators.py`
  - `tests/test_spinless_fermion_dense.py`
  - `tests/test_spinless_fermion_mpo_dense.py`
  - `tests/test_spinless_fermion_native_energy.py`
  - `tests/test_spinless_fermion_ad_solvers.py`
  - `tests/test_fermion_gpu_timing.py`

## 3. Commands

```bash
# CPU-only (default; always works, no GPU required):
python scripts/fermion_score.py --fast

# GPU benchmark (opt-in; runs only if LATTICETN_RUN_GPU=1 AND a V100/TITAN V
# is selected by the unified gpu_selector):
LATTICETN_RUN_GPU=1 python scripts/fermion_score.py --fast

# List required files:
python scripts/fermion_score.py --list
```

The score script:
1. runs the six required test files with `pytest -q` (CPU-only env by default;
   GPU visible when opted in);
2. runs `scripts/run_spinless_fermion_benchmark.py --markdown-output
   docs/FERMION_REPORT.md --json-output <path>` to regenerate the report and
   JSON results;
3. checks the report contains all required sections/terms;
4. prints `Fermion score: PASS` and exits 0 on success.

A clean GPU skip (no `LATTICETN_RUN_GPU=1`, or no matching V100/TITAN V) is a
**successful exit 0**: the CPU benchmark still runs and the report records the
skip reason.

## 4. Test coverage (what is checked)

- **operators** (`test_fermion_operators.py`): `{c,c^d}=I`, `c^2=(c^d)^2=0`,
  `n=c^d c`, `F^2=I`, `F c = -c F`; global JW operators anticommute on
  different sites (`{c_i,c_j}=0`, `{c_i,c^d_j}=0` for `i!=j`).
- **dense** (`test_spinless_fermion_dense.py`): Hermitian; free-fermion E0
  matches the single-particle formula; matches the explicit JW global-operator
  build; particle-hole symmetry at half filling; differs from the
  hard-core-boson H for N>=3.
- **MPO-dense** (`test_spinless_fermion_mpo_dense.py`): `to_dense` of the
  fermion MPO matches `spinless_fermion_dense` for N=2..6 across many
  `(t,V,mu)`; ground energy matches ED; shapes/bond dim; Hermitian.
- **native energy** (`test_spinless_fermion_native_energy.py`): native
  Rayleigh energy == dense energy on a random MPS; matches the Stage-1 path;
  differentiable; scale-invariant.
- **AD solvers** (`test_spinless_fermion_ad_solvers.py`): global/one-site/
  two-site AD each LOWER the energy and do NOT undershoot the exact ground
  beyond tolerance (`below_ground=False`); attractive regime; two-site
  approaches ED for small N.
- **GPU timing** (`test_fermion_gpu_timing.py`): the selector respects the
  V100/TITAN V filter (no fallback); CPU baseline always runs; when a matching
  GPU is present, CPU/GPU final energies agree within tolerance and the GPU
  energy does not undershoot the exact ground. Runtime/speedup recorded.

## 5. Report requirements (`docs/FERMION_REPORT.md`)

Must contain: mainline statement (AD mainline benchmarked; ED is reference
baseline only, not the solver; AD loss path unchanged — only the
Hamiltonian/MPO/operator layer is new); device info (GPU name, CUDA version,
PyTorch version, device, dtype, allowed filters, all/matched GPUs); exact
reference; CPU/GPU comparison table per solver (final energy, energy error,
energy per bond, runtime, speedup, below-ground flag); overall pass/fail;
known limitations. Must explicitly state this is **1D Jordan-Wigner fermions,
NOT graded fermionic tensors**, and that the **JW parity string is the key**.

## 6. Tolerances

- CPU/GPU final-energy agreement per AD solver: `|E_cpu - E_gpu| < 1e-6` for
  N=4 and `< 1e-5` for N=6 (matching the existing AD tolerances; not widened).
- GPU (and CPU) final energy must not undershoot the exact ground energy by
  more than `1e-6` (`below_ground=False`).
- Runtime and speedup must be **recorded**; the GPU is **not** required to be
  faster, because small systems are overhead-dominated.

## 7. Physics conventions

Open-boundary spinless fermion t-V chain, `d=2`, `torch.complex128`. ED is
CPU-only; the GPU runs only the AD solver optimization. This is 1D
Jordan-Wigner fermions, not a full graded fermionic tensor network. No silent
switch between fermion and spin conventions (the fermion module never mixes
with `spin_operators`).

## 8. Stop conditions

Stage 7A is complete when:
1. The core scores still pass (or their non-run is recorded).
2. `python scripts/fermion_score.py --fast` exits 0.
3. JW dense reference, MPO-to-dense, and native energy all align.
4. The AD solvers lower the energy on the fermion chain and do not undershoot
   the exact ground.
5. A V100/TITAN V GPU, if present, gives CPU/GPU parity + timing; otherwise
   the GPU portion clean-skips.
6. The report lists modified files, API, commands, CPU/GPU timing, items not
   run, and a suggested commit command.

## 9. Hard constraints

- No TDVP, no finite-temperature, no Hubbard, no graded fermionic tensors.
- No new large dependencies; no widening of existing thresholds; no change to
  Heisenberg conventions; no long benchmarks; no git mutation commands.
