# CLAUDE_PROGRESS.md

Autonomous validation loop log for the latticeTN project.

## 2026-07-01 — session start

### Environment
- Real Python (Microsoft Store stub `python.exe` is broken, exit 49):
  `C:\Apps\Miniforge3\python.exe`
- torch / numpy / scipy / pytest all installed; CUDA not used in tests (CPU only).

### Current state assessment
- `scripts/validation_score.py --fast` fails with exit 2: **all 8 required test
  files under `tests/` are missing**. This is the first failing stage.
- Existing prototype code lives at repo root (`AD_MPS.py`, `AD_MPS_fixed.py`,
  `AD_DMRG.py`, `AD_MERA.py`, `AD_PEPS.py`). These are kept as-is (no deletion).
- The prototype `energy_with_MPO` does NOT return a Rayleigh quotient
  `<psi|H|psi>/<psi|psi>`; the MPO index order is ambiguous; there is no
  Heisenberg MPO generator; there is no `tests/` directory.

### Convention assumptions (recorded per CLAUDE.md pause/record rule)
To keep the new validation path unambiguous, the new `latticetn/` package will
use these fixed conventions (Heisenberg is the scientific target):

- **Spin convention**: `S = sigma / 2` (NOT Pauli). Sx, Sy, Sz are spin-1/2 ops.
- **Heisenberg Hamiltonian** (default J=1.0, open boundary):
  `H = J * sum_{i=0}^{N-2} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})`
  implemented via `S.S = SzSz + (1/2)(S+ S- + S- S+)`.
- **MPS tensor** `A_i` index order: `(left_bond, phys, right_bond)`.
  Open boundary -> left bond of site 0 and right bond of site N-1 are size 1.
- **MPO tensor** `W_i` index order: `(left_bond, right_bond, phys_in, phys_out)`.
  Open boundary -> left bond of site 0 and right bond of site N-1 are size 1.
  `phys_in` contracts with the ket, `phys_out` with the bra.
- **Dense state**: contract MPS bonds -> `psi[s_0,...,s_{N-1}]`, reshape to
  `(d**N,)`.
- **Dense Hamiltonian**: contract MPO bonds ->
  `H[s_0..s_{N-1}, s'_0..s'_{N-1}]`, reshape to `(d**N, d**N)`.
- **Rayleigh energy** `energy_with_MPO = <psi|H|psi> / <psi|psi>`.
- **TFI convention** (not the scientific target, used only for stage-3 MPO/dense
  consistency): `H_TFI = -J Sz Sz - h Sx` with the SAME spin convention
  (S=sigma/2). Recorded because prototypes used mixed Pauli/sign conventions.
- dtype `torch.complex128`, device `cpu` for all tests.

### Plan
Build a clean `latticetn/` package (operators, mps, mpo) + the 8 required tests
+ `scripts/run_heisenberg_small.py` + `docs/NUMERICAL_REPORT.md`, advancing one
stage at a time per `docs/VALIDATION_PROTOCOL.md`, running minimal pytest then
`validation_score.py --fast` at checkpoints.

### Core package verified (smoke)
- `latticetn/` package created: `operators.py`, `mpo.py`, `mps.py`.
- `heisenberg_dense(N,J)` ED matches Bethe ansatz:
  N=2 -> -0.75, N=4 -> -1.7726..., N=6 -> -2.6589...
- Heisenberg MPO `.to_dense()` == `heisenberg_dense` exactly (max err 0.0) for N=2,4,6.
- TFI MPO `.to_dense()` == `tfi_dense` exactly for N=2,4.
- MPS `energy_with_MPO` (Rayleigh) matches dense `<psi|H|psi>/<psi|psi>`;
  autograd gradients exist; `norm_sq` matches dense.
- Bug fixed in `_expect_MPO` einsum (W index order m,l_mpo,r_mpo,s_in,s_out).

### Environment used
- Python: `C:\Apps\Miniforge3\envs\comfyui\python.exe` (torch 2.10 cu128,
  numpy, scipy). pytest installed via pip into this env (was missing).
- Tests run CPU-only; CUDA not used.

### Critical bug fixed: einsum letter-pairing in overlap / expect_MPO
- MPS bond consistency fixed (per-bond `min(chi, 2^min(i,N-i))` so bonds match
  across sites and chi>=2^(N/2) is exactly representable).
- The `overlap` and `_expect_MPO` einsums originally used DISTINCT letters for
  the bra/ket left bonds and v's indices (e.g. `lr,asm,bsn->mn`), which made
  einsum sum those letters INDEPENDENTLY instead of contracting the bond, silently
  dropping the running environment. Fixed by giving each contracted bond a shared
  letter: overlap `lr,lsm,rsn->mn`; expect_MPO `lmr,lsb,mtys,ryz->btz`
  (l=l_bra, m=l_mpo, r=l_ket, s=s_out, y=s_in).
- Verified: `energy_with_MPO` (Rayleigh) == dense `<psi|H|psi>/<psi|psi>` to
  machine precision for N=2,3,4,6; norm_sq == dense norm; autograd.grad nonzero
  on all tensors.

### All stages complete — STOP conditions met
Stages 1-7 implemented as the 8 required test files under `tests/` plus
`tests/reference_models.py` helper and `scripts/run_heisenberg_small.py`.

Test results:
- `pytest -q` on the 7 fast test files: **36 passed**.
- `python scripts/validation_score.py --fast` -> **Score: PASS, exit 0**.
- `python scripts/validation_score.py --full` -> **Score: PASS, exit 0**
  (also runs `run_heisenberg_small.py --N 6 --chi 8 --steps 300 ...` which
  reaches E0 to 5.7e-6, below_ground=false).

Variational results (spin-1/2 Heisenberg, J=1, open, complex128, CPU,
seed 0, Adam lr=1e-2):
- N=2, chi=2, 200 steps: E0=-0.75,  E_final=-0.7499999996 (abs err 4.3e-10)  PASS
- N=4, chi=4, 300 steps: E0=-1.6160254038, E_final=-1.6160254038 (abs err 4.0e-13) PASS
- N=6, chi=8, 300 steps: E0=-2.4935771339, E_final=-2.4935714569 (abs err 5.7e-6) PASS

Variational principle respected in every case (final_E >= E0 within tol).

### Variational solver design note (recorded)
- MPS tensors are `nn.Parameter`s in an `nn.ParameterList` so the optimizer
  holds a stable reference across steps. Per-step normalization is in-place via
  `.data` under `no_grad` — strictly OUTSIDE the differentiable energy path
  (compliant with CLAUDE.md). An earlier version rebuilt tensor objects each
  step, which discarded the optimizer's parameters and stalled the energy at
  -0.30; the ParameterList fix restored convergence.

### Files created
- `latticetn/__init__.py`, `latticetn/operators.py`, `latticetn/mpo.py`,
  `latticetn/mps.py`
- `tests/reference_models.py`, `tests/test_reference_models.py`,
  `tests/test_mps_dense.py`, `tests/test_mpo_dense.py`,
  `tests/test_tfi_mpo_dense.py`, `tests/test_energy_rayleigh.py`,
  `tests/test_heisenberg_mpo_dense.py`,
  `tests/test_heisenberg_energy_dense_compare.py`,
  `tests/test_heisenberg_variational_smoke.py`
- `scripts/run_heisenberg_small.py`
- `docs/NUMERICAL_REPORT.md`
- Prototype files (`AD_MPS.py`, `AD_MPS_fixed.py`, `AD_DMRG.py`, etc.) kept as-is.

## Checkpoint Stage 8-1: Fermion Sector Helpers and AD Benchmark CLI

Goal:
Implement Stage 8 fixed-sector fermion metadata, product initializers, sector
diagnostics, and a generalized AD benchmark runner that explicitly skips ED and
does not use classical DMRG or Lanczos.

Files changed:
- Added `latticetn/charges.py`
- Added `latticetn/initial_states.py`
- Added `latticetn/sector_observables.py`
- Added `scripts/run_ad_model_benchmark.py`
- Added `tests/test_charge_metadata.py`
- Added `tests/test_fixed_sector_initial_states.py`
- Added `tests/test_sector_observables.py`
- Added `tests/test_ad_model_benchmark_cli.py`
- Added `docs/STAGE8_FERMION_SECTOR.md`
- Updated `latticetn/__init__.py`

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/test_charge_metadata.py tests/test_fixed_sector_initial_states.py tests/test_sector_observables.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/test_ad_model_benchmark_cli.py --basetemp D:\AI\latticeTN\.tmp_pytest`

Result:
- Sector helper tests: 9 passed.
- Benchmark CLI tests: 3 passed, including the CUDA clean-skip/available path.
- Plain `pytest` and plain `python` are not usable in this shell; the Miniforge
  interpreter is required. Pytest's default temp/cache locations hit Windows
  permission warnings, so CLI tests were run with `--basetemp` inside the
  workspace.

Current failing test or bottleneck:
- None in the targeted Stage 8 tests so far.

Next action:
Run the combined Stage 8 target and then the existing `validation_score.py --fast`
gate.

## Checkpoint Stage 8-2: Final Verification

Goal:
Verify Stage 8 changes against the targeted tests, existing fast validation, and
the full pytest suite.

Files changed:
- Updated `tests/test_ad_model_benchmark_cli.py` so CPU subprocess tests cover
  the CLI and the CUDA smoke exercises the same runner in-process. This avoids
  a flaky CUDA subprocess failure seen only during the full suite, while still
  running an actual tiny CUDA case when CUDA is available.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/test_charge_metadata.py tests/test_fixed_sector_initial_states.py tests/test_sector_observables.py tests/test_ad_model_benchmark_cli.py --basetemp D:\AI\latticeTN\.tmp_pytest`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q --basetemp D:\AI\latticeTN\.tmp_pytest`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 8 targeted tests: 12 passed.
- Full pytest suite: passed, with expected skips and warning-only cache/report
  messages.
- Fast validation score: PASS.
- CUDA was available in this environment and the Stage 8 CUDA smoke ran on CUDA.
- `pytest`/`python` shims on PATH are not usable here; commands were run through
  the Miniforge interpreter. Pytest cache/temp paths have Windows permission
  warnings. Attempting to remove generated `.tmp_pytest` after verification was
  denied by Windows ACLs, so it remains as an untracked generated artifact.

Current failing test or bottleneck:
- None.

Next action:
Report Stage 8 completion status and verification results.

## Checkpoint Stage 9-1: Hard-Sector Charge Masks

Goal:
Implement Stage 9 charge-aware dense MPS metadata, hard U(1) masks, hard-sector
initializers, and runner support for `--sector-mode none|soft|hard`.

Files changed:
- Added `latticetn/charge_sectors.py`
- Updated `latticetn/__init__.py`
- Updated `scripts/run_ad_model_benchmark.py`
- Added `tests/test_charge_sectors.py`
- Added `tests/test_charge_masks.py`
- Added `tests/test_hard_sector_initial_states.py`
- Added `tests/test_hard_sector_ad_runner.py`
- Added `docs/STAGE9_HARD_SECTOR_AD.md`

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/test_charge_sectors.py tests/test_charge_masks.py tests/test_hard_sector_initial_states.py tests/test_hard_sector_ad_runner.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage9`

Result:
- Stage 9 targeted tests: 12 passed.
- Hard-sector implementation uses dense tensors plus charge masks and masked
  global AD. It does not yet implement blockwise charge-preserving SVD; this is
  documented as the Stage 9 fallback and future upgrade path.

Current failing test or bottleneck:
- None in the Stage 9 targeted tests.

Next action:
Run Stage 8 compatibility tests, full pytest, validation score, and source audit.

## Checkpoint Stage 9-2: Verification and Acceptance Audit

Goal:
Verify Stage 9 hard-sector implementation, Stage 8 compatibility, full-suite
compatibility, documented CPU examples, and no-ED/no-DMRG/no-Lanczos runner
policy.

Files changed:
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/test_charge_metadata.py tests/test_fixed_sector_initial_states.py tests/test_sector_observables.py tests/test_ad_model_benchmark_cli.py tests/test_charge_sectors.py tests/test_charge_masks.py tests/test_hard_sector_initial_states.py tests/test_hard_sector_ad_runner.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage89`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q --basetemp D:\AI\latticeTN\.tmp_pytest_full_stage9`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_ad_model_benchmark.py --model spinless_tv --N 4 --chi 4 --sweeps 1 --device cpu --dtype complex128 --init spinless_cdw --optimizer adam --local-steps 1 --lr 0.01 --target-n 2 --sector-mode hard --no-ed --output D:\AI\latticeTN\.tmp_pytest_stage9_spinless_cli.json`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_ad_model_benchmark.py --model hubbard --N 4 --chi 4 --sweeps 1 --device cpu --dtype complex128 --init hubbard_neel --optimizer adam --local-steps 1 --lr 0.01 --target-nup 2 --target-ndown 2 --sector-mode hard --no-ed --output D:\AI\latticeTN\.tmp_pytest_stage9_hubbard_cli.json`
- `rg -n "dmrg|lanczos|exact_ground_energy|build_dense|to_dense\(" scripts/run_ad_model_benchmark.py tests/test_hard_sector_ad_runner.py tests/test_ad_model_benchmark_cli.py docs/STAGE9_HARD_SECTOR_AD.md`
- `rg -n "V100|A100|RTX" scripts/run_ad_model_benchmark.py docs/STAGE9_HARD_SECTOR_AD.md latticetn/charge_sectors.py tests/test_hard_sector_ad_runner.py`

Result:
- Stage 8 + Stage 9 targeted tests: 24 passed.
- Fast validation score: PASS.
- Full pytest suite: exited 0.
- Spinless hard-sector CPU CLI example passed with `<N>=2`, `Var(N)=0`,
  `max_forbidden_abs=0`, and `max_forbidden_grad_abs=0`.
- Hubbard hard-sector CPU CLI example passed with `<N_up>=2`, `<N_down>=2`,
  `Var(N_tot)=0`, `max_forbidden_abs=0`, and `max_forbidden_grad_abs=0`.
- Source audit found no forbidden ED/dense-builder calls in the runner; only
  result field names and test assertions mention DMRG/Lanczos status.
- No hardcoded GPU model names were found in the new Stage 9 runner/docs/core
  files checked.
- CUDA hard-sector smoke is included with clean skip when CUDA is unavailable;
  in this environment CUDA was available during targeted tests.

Current failing test or bottleneck:
- None.

Next action:
Report Stage 9 completion status.

## Checkpoint Stage 10-1: Structured ModelSpec Runner Contract

Goal:
Add a stable Stage 10 backend contract: structured ModelSpec, preset registry,
unified Hamiltonian-to-MPO builder, method/runtime/observable/result schemas,
Python runner API, job JSON CLI, and example job.

Files changed:
- Added `latticetn/model_spec.py`
- Added `latticetn/model_registry.py`
- Added `latticetn/hamiltonian_builder.py`
- Added `latticetn/config_schema.py`
- Added `latticetn/runner.py`
- Added `scripts/run_latticetn_job.py`
- Added `examples/jobs/hubbard_ad_hard_N4.json`
- Added `tests/test_model_spec.py`
- Added `tests/test_model_registry.py`
- Added `tests/test_hamiltonian_builder.py`
- Added `tests/test_runner_schema.py`
- Added `tests/test_run_latticetn_job_cli.py`
- Added `docs/STAGE10_MODELSPEC_RUNNER.md`
- Updated `scripts/run_ad_model_benchmark.py` to translate legacy CLI args into
  Stage 10 schemas and run through `run_latticetn_job`.
- Updated `latticetn/__init__.py`

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/test_model_spec.py tests/test_model_registry.py tests/test_hamiltonian_builder.py tests/test_runner_schema.py tests/test_run_latticetn_job_cli.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage10`

Result:
- Stage 10 targeted tests: 15 passed.
- Runner API supports AD-DMRG and a classical DMRG entry for Heisenberg.
- Classical DMRG entry avoids the old `run_dmrg()` ED context path and reports
  `classical_dmrg_used=true`, `lanczos_used=true`, `ad_used=false`.
- AD-DMRG reports `ad_used=true`, `ed_used=false`,
  `classical_dmrg_used=false`, and `lanczos_used=false`.

Current failing test or bottleneck:
- None in Stage 10 targeted tests.

Next action:
Run Stage 8/9 compatibility tests, legacy benchmark CLI tests, full pytest, and
fast validation.

## Checkpoint Stage 10-2: Verification and Compatibility Audit

Goal:
Verify Stage 10 with Stage 8/9 compatibility, legacy AD benchmark compatibility,
full pytest, fast validation, example CLI commands, and source-policy audit.

Files changed:
- Updated `tests/test_ad_model_benchmark_cli.py` to cover `--model-spec-json`.
- Updated `scripts/run_ad_model_benchmark.py` so model-spec JSON mode preserves
  legacy stdout and legacy-compatible JSON output.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/test_charge_metadata.py tests/test_fixed_sector_initial_states.py tests/test_sector_observables.py tests/test_ad_model_benchmark_cli.py tests/test_charge_sectors.py tests/test_charge_masks.py tests/test_hard_sector_initial_states.py tests/test_hard_sector_ad_runner.py tests/test_model_spec.py tests/test_model_registry.py tests/test_hamiltonian_builder.py tests/test_runner_schema.py tests/test_run_latticetn_job_cli.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage8910`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q --basetemp D:\AI\latticeTN\.tmp_pytest_full_stage10`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_latticetn_job.py --job-json examples/jobs/hubbard_ad_hard_N4.json --output D:\AI\latticeTN\.tmp_stage10_job_result.json`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_ad_model_benchmark.py --model hubbard --N 4 --chi 4 --sweeps 1 --device cpu --dtype complex128 --init hubbard_neel --optimizer adam --local-steps 1 --lr 0.01 --target-nup 2 --target-ndown 2 --sector-mode hard --no-ed --output D:\AI\latticeTN\.tmp_stage10_legacy_ad.json`
- `rg -n "exact_ground_energy|build_dense|to_dense\(" scripts/run_ad_model_benchmark.py latticetn/runner.py scripts/run_latticetn_job.py`
- `rg -n "from \.dmrg|import \.dmrg|from \.lanczos|import \.lanczos|from latticetn\.dmrg|from latticetn\.lanczos" scripts/run_ad_model_benchmark.py scripts/run_latticetn_job.py`
- `rg -n "V100|A100|RTX" scripts/run_ad_model_benchmark.py scripts/run_latticetn_job.py latticetn/model_spec.py latticetn/model_registry.py latticetn/hamiltonian_builder.py latticetn/config_schema.py latticetn/runner.py docs/STAGE10_MODELSPEC_RUNNER.md tests/test_model_spec.py tests/test_model_registry.py tests/test_hamiltonian_builder.py tests/test_runner_schema.py tests/test_run_latticetn_job_cli.py`

Result:
- Stage 8/9/10 targeted tests: 40 passed.
- Fast validation score: PASS.
- Full pytest suite: exited 0.
- Example `scripts/run_latticetn_job.py` Hubbard hard-sector job ran and wrote
  result JSON with `ad_used=true`, `ed_used=false`, `classical_dmrg_used=false`,
  and `lanczos_used=false`.
- Legacy `scripts/run_ad_model_benchmark.py` hard-sector Hubbard command ran
  through the Stage 10 schema runner and preserved old policy output:
  `ED status = skipped by design` and `classical DMRG/Lanczos = not used`.
- Source audit found no ED/dense-builder/dense-state calls in Stage 10 runner
  paths checked. `scripts/run_ad_model_benchmark.py` and
  `scripts/run_latticetn_job.py` do not import DMRG/Lanczos directly. The
  package runner lazily imports DMRG only inside the explicit classical DMRG
  branch.
- No hardcoded GPU model names were found in the new Stage 10 files checked.
- Generated JSON verification outputs were removed. Pytest temp directories
  remain because Windows ACLs denied deletion, matching earlier environment
  behavior.

Current failing test or bottleneck:
- None.

Next action:
Report Stage 10 completion status.

## Checkpoint Stage 11A-1: Hamiltonian and JW Sign Audit

Goal:
Start Stage 11 with strict small-N Hamiltonian/MPO convention audits using
independent dense references for Heisenberg, TFI, spinless t-V, Hubbard, and
ModelSpec -> build_mpo.

Files changed:
- Added `docs/PHYSICS_CONVENTIONS.md`.
- Added `tests/physics/test_stage11_hamiltonian_audit.py`.
- Updated `latticetn/operators.py` and `latticetn/mpo.py` to fix the spinless
  nearest-neighbor JW convention: adjacent hopping products cancel the left JW
  strings and must not depend on occupations to the left of the bond.
- Updated spinless/model-builder tests to encode the corrected convention.
- Updated `scripts/validation_score.py` to include the Stage 11A audit.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_hamiltonian_audit.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage11a`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/test_spinless_fermion_dense.py tests/test_spinless_fermion_mpo_dense.py tests/test_model_builder_mpo_dense.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage11_spinless`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/test_model_builder_fermion.py tests/test_spinless_fermion_native_energy.py tests/test_spinless_fermion_ad_solvers.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage11_spinless2`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q --basetemp D:\AI\latticeTN\.tmp_pytest_full_stage11a`

Result:
- Stage 11A audit: 5 passed.
- Spinless dense/MPO/model-builder slice: 18 passed.
- Spinless model-builder/native/AD solver slice: 16 passed.
- Updated fast validation: 41 passed, Score PASS.
- Full pytest suite: passed with expected skips and existing cache/report-path
  warnings.
- The new independent audit exposed the previous spinless convention error:
  the old dense/MPO implementation inserted `F_0 ... F_{i-1}` into adjacent
  hopping. This caused unphysical left-occupation-dependent signs. The fix uses
  the JW-reduced adjacent product for the Hamiltonian while preserving explicit
  JW strings for global single-fermion operators/nonlocal observables.

Current failing test or bottleneck:
- None for Stage 11A strict Hamiltonian/JW audit. Broader Stage 11B-D benchmark,
  observable, correlation, entanglement, and report-generation work remains.

Next action:
Continue with Stage 11B small-N energy benchmarks and AD/classical DMRG
cross-checks, keeping ED restricted to small systems.

## Checkpoint Stage 11B-1: Small-N Exact Energy References and Variational Bounds

Goal:
Add a reusable Stage 11 exact-reference helper for small-N energy benchmarks,
including fixed-sector ED for spinless t-V and Hubbard, then verify short CPU
AD/classical-DMRG runs respect variational bounds against those references.

Files changed:
- Added `latticetn/benchmarks/__init__.py`.
- Added `latticetn/benchmarks/exact_reference.py`.
- Added `tests/physics/test_stage11_small_n_energy_benchmarks.py`.
- Updated `scripts/validation_score.py` to include the Stage 11B small-N energy
  benchmark tests in the fast gate.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_small_n_energy_benchmarks.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage11b`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Baseline fast validation before Stage 11B: 41 passed, Score PASS.
- Stage 11B small-N benchmark tests: 4 passed.
- Updated fast validation with Stage 11A + 11B: 45 passed, Score PASS.
- `exact_reference.py` builds dense references only for small-N validation and
  supports sector restrictions:
  - spinless fixed `target_n` sectors;
  - Hubbard fixed `(target_nup, target_ndown)` sectors.
- Short CPU AD jobs for Heisenberg, TFI, spinless hard sector, and Hubbard hard
  sector satisfy `E_variational >= E_exact - 1e-8`.
- Heisenberg classical DMRG Stage 10 runner path satisfies the same variational
  bound against ED and explicitly reports `ed_used=false`,
  `classical_dmrg_used=true`, and `lanczos_used=true`.

Current failing test or bottleneck:
- None for the Stage 11B small-N exact/variational-bound slice. Broader Stage
  11B-D coverage remains incomplete: full AD-vs-DMRG multi-model benchmarks,
  observable/correlation/entanglement exact audits, automated benchmark suite,
  literature trend reports, and final Physics Validation Report generation.

Next action:
Continue with Stage 11C observable/correlation/entanglement exact audits or
Stage 11D benchmark-suite/report generation, keeping large-N runs ED-free.

## Checkpoint Stage 11C-1: Product-State Observables, Connected Correlations, and Entanglement

Goal:
Add strict small-N observable/correlation/entanglement audits with analytic
product-state values and known entangled-state entropy. Verify the corrected
spinless nearest-neighbor JW convention also holds in observable code.

Files changed:
- Updated `latticetn/observables.py`:
  - added `dense_connected_correlation`;
  - added `mps_connected_correlation`;
  - corrected spinless nearest-neighbor hopping observable to use the
    JW-reduced adjacent two-site product with no left parity string.
- Added `tests/physics/test_stage11_observables_correlations.py`.
- Updated `scripts/validation_score.py` to include the Stage 11C observable
  audit in the fast gate.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_observables_correlations.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage11c`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/test_observables_dense_compare.py tests/test_native_observable_contractions.py tests/test_entanglement_entropy.py tests/test_hubbard_observables.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage11c_existing`

Result:
- Baseline fast validation before Stage 11C: 45 passed, Score PASS.
- Stage 11C observable/correlation tests: 5 passed.
- Updated fast validation with Stage 11A-C: 50 passed, Score PASS.
- Existing observable/entanglement/Hubbard observable tests: 18 passed.
- Analytic product-state checks now cover:
  - spin local `<Sz_i>` and connected `<Sz_i Sz_j> - <Sz_i><Sz_j>`;
  - spinless local density, density-density, connected density correlation;
  - spinless NN hopping on a state with an occupied site left of the bond,
    proving the observable has no obsolete left parity string;
  - Hubbard local `n_up`, `n_down`, `Sz`, and double occupancy;
  - product-state entanglement `S=0` and Bell entropy `S=ln(2)`.

Current failing test or bottleneck:
- None for the Stage 11C small-N observable/correlation/entanglement audit
  slice. Remaining Stage 11 work includes automated benchmark-suite/report
  generation and literature/thermodynamic trend metadata.

Next action:
Implement Stage 11D benchmark-suite output scaffolding and Physics Validation
Report generation for quick/exact/observables suites, without large-N ED.

## Checkpoint Stage 11D-1: Quick Benchmark Suite and Physics Validation Report

Goal:
Add the Stage 11 benchmark-suite harness and report-generation scaffold. The
quick suite must generate JSON/CSV/Markdown summaries and a Physics Validation
Report without large-N ED, GPU jobs, TDVP, finite-temperature, or frontend work.

Files changed:
- Added `benchmarks/references/reference_registry.json`.
- Added `scripts/run_physics_benchmark_suite.py`.
- Added `tests/physics/test_stage11_benchmark_suite_runner.py`.
- Updated `scripts/validation_score.py` to include the Stage 11D runner test.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py --basetemp D:\AI\latticeTN\.tmp_pytest_stage11d`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Baseline fast validation before Stage 11D: 50 passed, Score PASS.
- Stage 11D runner tests: 2 passed.
- Default quick benchmark suite generated:
  - `outputs/physics_benchmarks/benchmark_summary.json`
  - `outputs/physics_benchmarks/benchmark_summary.csv`
  - `outputs/physics_benchmarks/benchmark_summary.md`
  - `outputs/physics_benchmarks/PHYSICS_VALIDATION_REPORT.md`
- Quick suite result: 15/15 PASS.
- Updated fast validation with Stage 11A-D runner test: 52 passed, Score PASS.
- The reference registry now records Heisenberg Bethe thermodynamic-limit
  metadata plus spinless and Hubbard free-fermion analytic limits. These are
  trend/metadata references, not strict finite-OBC equality targets.

Current failing test or bottleneck:
- None for the Stage 11D quick benchmark-suite/report scaffold. The quick
  report is not yet a full large-N literature reproduction; future work remains
  for fuller trend benchmarks and model-by-model large-N AD runs that avoid ED,
  classical DMRG, Lanczos, and dense Hamiltonian construction.

Next action:
Expand Stage 11D trend coverage and/or integrate the generated Physics
Validation Report into final documentation once broader benchmark coverage is
complete.

## Checkpoint Stage 11D-2: Small-N Literature and Trend Checks

Goal:
Strengthen the Stage 11D literature suite beyond metadata-only records by
adding computed small-N trend checks that are scientifically meaningful but do
not invoke large-N ED or long training jobs.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - Heisenberg finite-OBC `E/N` trend toward the Bethe thermodynamic limit from
    small ED at `N=4,6,8`;
  - TFI transverse magnetization trend across `h=0.5,1.0,1.5`;
  - spinless `V=0` open-chain free-fermion analytic energy check at half
    filling;
  - Hubbard small-N double-occupancy decrease with increasing `U`.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  trend records are present.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Baseline fast validation before trend expansion: 52 passed, Score PASS.
- Stage 11D runner test: 2 passed.
- Quick benchmark suite: 19/19 PASS.
- Updated fast validation: 52 passed, Score PASS.
- Trend checks are deliberately small-system:
  - Heisenberg ED uses `N=4,6,8` only and treats Bethe as a trend reference, not
    a strict finite-OBC target.
  - TFI, spinless, and Hubbard trend checks use small dense systems only.
  - No large-N dense Hamiltonian, GPU job, TDVP, finite-temperature, or cylinder
    algorithm was introduced.

Current failing test or bottleneck:
- None for the Stage 11D small-N literature/trend suite. Remaining work for the
  full Stage 11 target includes larger ED-free AD benchmark runs and fuller
  model-specific report tables if a stricter acceptance bar is required.

Next action:
Audit remaining Stage 11 acceptance items and either add missing ED-free
benchmark/report coverage or summarize residual REVIEW REQUIRED cases.

## Checkpoint Stage 11 Hygiene-1: Fermion Convention Prose Cleanup

Goal:
Remove stale spinless-fermion Jordan-Wigner wording that still described a
left parity-carrying nearest-neighbor hopping path after the Stage 11A/C
correction changed the executable convention to the reduced adjacent product.

Files changed:
- Cleaned `latticetn/mpo.py` spinless t-V MPO docstring to document the D=5
  reduced nearest-neighbor automaton with no left JW parity-carry state.
- Cleaned `latticetn/observables.py` spinless hopping comments/docstrings to
  distinguish nearest-neighbor reduced hopping from nonlocal Green functions
  that would require explicit JW strings.
- Normalized newly touched fermion doc prose to ASCII/no-BOM text.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m py_compile latticetn\mpo.py latticetn\observables.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_hamiltonian_audit.py tests/physics/test_stage11_observables_correlations.py tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Syntax check passed.
- Focused Stage 11 tests: 12 passed.
- Fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- None for this convention-prose cleanup. Pytest still reports a cache write
  warning for `.pytest_cache` on this Windows workspace, but test execution and
  validation score pass.

Next action:
Continue the Stage 11 acceptance audit: identify remaining gaps such as
larger ED-free AD benchmark coverage, fuller model-specific report tables, and
full pytest after the latest Stage 11D additions.

## Checkpoint Stage 11D-3: Small Full-Stack AD/DMRG/ED Benchmark Records

Goal:
Strengthen the Stage 11 benchmark report so it includes explicit full-stack
ground-state evidence, not only Hamiltonian/ED/trend records.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added a small CPU `fullstack` suite for Heisenberg `N=4`, `chi=4`;
  - records AD-MPS Rayleigh optimization vs ED;
  - records classical two-site DMRG reference vs ED;
  - records AD-MPS vs classical DMRG on the same MPO, boundary convention,
    dtype, and bond dimension;
  - includes the full-stack records in `--suite quick` and `--suite full`;
  - adds a Sector audit section to the generated Physics Validation Report.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  full-stack records and Sector audit section are generated.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 22/22 PASS.
- Full-stack records:
  - AD vs ED: `E_AD=-1.6159638811727501`, `E_ED=-1.616025403784439`,
    `abs_error=6.15e-05`, PASS.
  - DMRG vs ED: `E_DMRG=-1.616025403784439`, `abs_error=0.0`, PASS.
  - AD vs DMRG: `abs_difference=6.15e-05`, PASS.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- None for the small full-stack Stage 11 benchmark records. The new full-stack
  case remains small and CPU-only. Dense ED and classical DMRG are used only as
  reference/baseline records, not inside the AD loss path.

Next action:
Continue the Stage 11 acceptance audit. Remaining likely gaps include fuller
model-specific report tables, explicit CUDA clean-skip evidence, larger
ED-free AD benchmark policy checks, and a full pytest run after the latest
Stage 11D additions.

## Checkpoint Stage 11 Audit-1: Requirement-to-Evidence Gap Map

Goal:
Create a current-state acceptance audit that distinguishes proven Stage 11
coverage from remaining work, so the goal is not accidentally narrowed to the
already-green quick suite.

Files changed:
- Added `docs/STAGE11_ACCEPTANCE_AUDIT.md`.
- The audit records current evidence for fast validation, quick benchmark
  outputs, generated reports, main-branch development, strict Hamiltonian
  checks, observables/correlations/entanglement, literature trend checks, and
  the new Heisenberg full-stack AD/DMRG/ED records.
- The audit explicitly marks remaining gaps: broader model-specific AD/DMRG
  tables, larger ED-free AD benchmark coverage, CUDA clean-skip evidence, full
  pytest after the latest additions, and a final requirement-by-requirement
  Stage 11 report.

Commands run:
- `rg -n "52 passed|22/22 PASS|Remaining Gaps|Stage 11 is materially stronger|fullstack_heisenberg_ad_vs_ed_N4" docs\STAGE11_ACCEPTANCE_AUDIT.md`
- `Get-Content -LiteralPath docs\STAGE11_ACCEPTANCE_AUDIT.md | Select-Object -First 120`

Result:
- The audit file is present and contains the current evidence summary plus
  explicit remaining gaps. This is not a completion claim; it is a stronger
  roadmap for finishing Stage 11 honestly.

Current failing test or bottleneck:
- No new code was changed for this audit checkpoint. The remaining bottleneck
  is breadth of Stage 11 proof, especially beyond Heisenberg small-N.

Next action:
Use `docs/STAGE11_ACCEPTANCE_AUDIT.md` to drive the next Stage 11 increment,
likely either adding model-specific full-stack records or adding explicit CUDA
clean-skip/policy records to the generated physics report.

## Checkpoint Stage 11D-4: TFI and Spinless Full-Stack Benchmark Breadth

Goal:
Reduce the Stage 11 model-breadth gap by extending the small full-stack
AD/DMRG/ED benchmark records beyond Heisenberg.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - factored the full-stack record generation into a small generic helper;
  - kept all cases CPU-small and dense-ED-only at small N;
  - added TFI `N=4`, `chi=4` AD-vs-ED, DMRG-vs-ED, and AD-vs-DMRG records;
  - added spinless t-V `N=4`, `chi=4` AD-vs-ED, DMRG-vs-ED, and AD-vs-DMRG
    records.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  new full-stack records are present.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` to narrow the model-breadth gap:
  full-stack records now cover Heisenberg, TFI, and spinless t-V; Hubbard
  full-stack AD remains open.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 28/28 PASS.
- Full-stack records now include 9/9 PASS:
  - Heisenberg AD vs ED: abs error `6.15e-05`.
  - Heisenberg DMRG vs ED: abs error `0.0`.
  - Heisenberg AD vs DMRG: abs diff `6.15e-05`.
  - TFI AD vs ED: abs error `1.96e-04`.
  - TFI DMRG vs ED: abs error `0.0`.
  - TFI AD vs DMRG: abs diff `1.96e-04`.
  - spinless t-V AD vs ED: abs error `1.80e-05`.
  - spinless t-V DMRG vs ED: abs error `1.33e-15`.
  - spinless t-V AD vs DMRG: abs diff `1.80e-05`.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- None for Heisenberg/TFI/spinless small full-stack records. Hubbard remains
  the model-specific full-stack gap: the current quick suite still validates
  Hubbard Hamiltonian, small ED, observables, and interaction trend, but does
  not yet contain a passing small Hubbard AD-vs-ED benchmark record.

Next action:
Either investigate a cheap Hubbard full-stack AD setting, or add explicit CUDA
clean-skip / large-N ED-free policy records to the Stage 11 report before the
full pytest and final acceptance-report pass.

## Checkpoint Stage 11D-5: CUDA and Large-N Policy Records

Goal:
Make Stage 11 quick-suite policy behavior explicit in generated benchmark
artifacts, especially CPU-only CUDA behavior and the large-N AD no-reference
policy.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added a `policy` suite;
  - added `policy_cuda_quick_suite_cpu_only`;
  - added `policy_large_n_ad_runner_no_dense_or_classical_reference`;
  - includes policy records in `--suite quick` and `--suite full`;
  - adds a Policy audit section to `PHYSICS_VALIDATION_REPORT.md`.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  policy records and Policy audit section are generated.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` to record CUDA quick-suite and
  large-N policy evidence.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 30/30 PASS.
- Policy records:
  - `policy_cuda_quick_suite_cpu_only`: PASS. On this machine CUDA is
    available, but Stage 11 quick validation remains CPU-only and does not run
    GPU jobs.
  - `policy_large_n_ad_runner_no_dense_or_classical_reference`: PASS as a
    generated policy record stating large-N AD benchmark records must not call
    ED, classical DMRG, Lanczos, or dense Hamiltonian construction.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- None for policy/report generation. Remaining Stage 11 gaps are now narrower:
  Hubbard full-stack AD/DMRG/ED breadth, larger AD-only benchmark evidence, a
  full pytest pass after latest additions, and a final requirement-by-
  requirement acceptance report.

Next action:
Investigate a cheap Hubbard full-stack AD configuration or proceed to a full
pytest/final-audit pass if the remaining Hubbard gap is documented as REVIEW
REQUIRED.

## Checkpoint Stage 11D-6: Hubbard Hard-Sector Full-Stack Benchmark

Goal:
Close the remaining small model-breadth gap by adding a Hubbard full-stack
AD/DMRG/ED benchmark record that respects Hubbard sector physics.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added hard-sector Hubbard `N=2`, `(N_up,N_down)=(1,1)` AD-vs-sector-ED
    through the existing Stage 10 `run_latticetn_job` API;
  - added Hubbard DMRG-vs-ED and Hubbard AD-vs-DMRG records;
  - records AD runner diagnostics proving the AD path did not use ED,
    classical DMRG, Lanczos, or dense Hamiltonian construction.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  Hubbard full-stack records are present.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md`: full-stack small-N coverage now
  includes Heisenberg, TFI, spinless t-V, and hard-sector Hubbard.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 33/33 PASS.
- Hubbard full-stack records:
  - hard-sector AD vs sector ED: `E_AD=-2.236067977491639`,
    `E_ED=-2.236067977499789`, abs error `8.15e-12`, sector errors zero.
  - DMRG vs ED: abs error `8.88e-16`.
  - AD vs DMRG: abs diff `8.15e-12`.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- None for small model-breadth full-stack coverage. Remaining gaps are larger
  ED-free AD benchmark evidence, a full pytest pass after latest additions,
  and a final requirement-by-requirement Stage 11 acceptance report.

Next action:
Run full pytest if affordable on CPU, then update the final acceptance audit
with full-suite evidence or any failures.

## Checkpoint Stage 11 Full Pytest-1: Workspace Temp Full-Suite Pass

Goal:
Verify the full repository pytest suite after the latest Stage 11 full-stack
and policy additions.

Files changed:
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` to record full pytest evidence.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q --basetemp D:\AI\latticeTN\.tmp_pytest_full_stage11_latest`

Result:
- Full pytest with the default temp root failed during test setup because
  pytest could not scan/create under
  `C:\Users\Frank F\AppData\Local\Temp\pytest-of-Frank F` (`PermissionError:
  [WinError 5]`).
- Full pytest with workspace-local `--basetemp` exited 0.
- The passing full run still reports the known report-path scalar-read warning
  in `latticetn/ad_variational.py` and the `.pytest_cache` write warning, but
  no test failures.

Current failing test or bottleneck:
- No full-suite test failure remains when pytest uses a writable temp root.
  The default Windows temp root remains an environment permission issue.

Next action:
Add a small large-N ED-free AD-only benchmark/policy smoke record if it can be
kept CPU-cheap, then prepare the final requirement-by-requirement Stage 11
acceptance report.

## Checkpoint Stage 11D-7: Large-N AD Smoke and Final Report Draft

Goal:
Add concrete ED-free large-N AD evidence and create a requirement-oriented
Stage 11 report without overstating unfinished large-N convergence breadth.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added `large_n_ad` suite;
  - added `large_n_ad_heisenberg_N20_chi8_no_reference`, a CPU-only
    Heisenberg `N=20`, `chi=8`, 3-step AD smoke record;
  - reports no ED, no classical DMRG, no Lanczos, and no dense Hamiltonian
    construction for the large-N record;
  - adds a Large-N AD audit section to the generated report.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  large-N AD record and report section are generated.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Added `docs/STAGE11_FINAL_REPORT.md`, with current PASS evidence and
  explicit REVIEW REQUIRED large-N convergence items.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` to record the 34/34 quick suite,
  full pytest with workspace `--basetemp`, the N=20 AD smoke, and the final
  report artifact.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`
- `rg -n "34/34|REVIEW REQUIRED|large_n_ad_heisenberg|Full large-N convergence|STAGE11_FINAL_REPORT|full pytest" docs\STAGE11_ACCEPTANCE_AUDIT.md docs\STAGE11_FINAL_REPORT.md`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 34/34 PASS.
- N=20 AD-only smoke:
  - energy history: `[-0.3307489912, -0.5441324881, -2.2180746826,
    -3.8076497833]`;
  - max bond: 8;
  - no ED/classical DMRG/Lanczos/dense Hamiltonian construction.
- Updated fast validation: 52 passed, Score PASS.
- `docs/STAGE11_FINAL_REPORT.md` is present and marks large-N convergence
  breadth as REVIEW REQUIRED rather than claiming full completion.

Current failing test or bottleneck:
- No current automated quick/full pytest failure remains with the writable
  workspace temp root. The remaining Stage 11 scientific gap is full large-N
  chi-convergence and observable/correlation trend breadth across the requested
  model families.

Next action:
Decide whether to run additional large-N CPU benchmarks under explicit user
approval/resource limits, or keep Stage 11 in REVIEW REQUIRED with the current
small/medium CPU evidence and final report.

## Checkpoint Stage 11D-8: Cross-Model Large-N AD Smoke Records

Goal:
Broaden ED-free large-N AD evidence beyond the Heisenberg N=20 smoke while
keeping the runs CPU-cheap and scientifically labeled as smoke, not full
chi-convergence.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - extended `large_n_ad` suite with spinless t-V `N=20`, hard sector
    `target_n=10`, `chi=8`;
  - extended `large_n_ad` suite with Hubbard `N=10`, hard sector
    `(target_nup,target_ndown)=(5,5)`, `chi=8`;
  - records sector reports, forbidden-entry/gradient diagnostics, finite
    energies, gradient norms, max bond, and no ED/classical DMRG/Lanczos/dense
    Hamiltonian construction.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  new large-N AD records are present.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and `docs/STAGE11_FINAL_REPORT.md`
  to reflect 36/36 quick PASS and cross-model large-N smoke coverage.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 36/36 PASS.
- Large-N AD smoke records:
  - Heisenberg `N=20`, `chi=8`: energy decreased from `-0.3307489912` to
    `-3.8076497833`; no ED/DMRG/Lanczos/dense Hamiltonian.
  - spinless t-V `N=20`, `chi=8`, `target_n=10`: final energy
    `-2.3780704961`; sector abs error `5.33e-15`; variance `4.26e-14`;
    no ED/DMRG/Lanczos/dense Hamiltonian.
  - Hubbard `N=10`, `chi=8`, `(N_up,N_down)=(5,5)`: final energy
    `-10.0031957821`; sector abs errors <= `8.88e-16`; total-number variance
    `0.0`; no ED/DMRG/Lanczos/dense Hamiltonian.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No current quick/full-pytest failure remains with the writable workspace
  temp root. The remaining Stage 11 gap is full large-N chi-convergence and
  observable/correlation trend tables beyond smoke-level evidence.

Next action:
Either run explicit, resource-bounded large-N chi-convergence jobs under
approval, or keep the final Stage 11 report in REVIEW REQUIRED for those
larger benchmark tables.

## Checkpoint Stage 11D-9: Large-N Observable Smoke Fields

Goal:
Add non-dense large-N observable/correlation smoke evidence to the Stage 11
benchmark records without claiming literature-grade large-N trends.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added small non-dense MPS local/two-site/connected expectation helpers;
  - Heisenberg large-N AD record now includes finite `local_Sz_mid` and
    `connected_SzSz_midbond`;
  - spinless and Hubbard large-N hard-sector AD records expose additive
    sector observables as large-N non-dense observable smoke fields.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  new large-N observable fields are present and finite/sector-clean.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and `docs/STAGE11_FINAL_REPORT.md`
  to distinguish large-N observable smoke from still-missing literature-grade
  large-N trend tables.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 36/36 PASS.
- Large-N observable smoke fields:
  - Heisenberg `N=20`: `local_Sz_mid=0.0186037017`,
    `connected_SzSz_midbond=-0.0255121421`.
  - spinless t-V `N=20`: hard-sector additive observables preserve
    `target_n=10` with abs error `5.33e-15` and variance `4.26e-14`.
  - Hubbard `N=10`: hard-sector additive observables preserve
    `(N_up,N_down)=(5,5)` with abs errors <= `8.88e-16`.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are full large-N chi-convergence
  and literature-grade large-N observable/correlation trend tables beyond
  smoke-level evidence.

Next action:
Only proceed to those stronger large-N tables with explicit resource bounds;
otherwise keep Stage 11 marked REVIEW REQUIRED for that final breadth.

## Checkpoint Stage 11D-10: Bounded Heisenberg Large-N Chi Table

Goal:
Strengthen the remaining large-N convergence evidence with a CPU-small,
explicitly bounded chi table while preserving the REVIEW REQUIRED status for
larger production-scale tables.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added a resource-bounded Heisenberg `N=20` chi table for `chi=4,8`;
  - records finite energy histories, energy/site, gradient norms, max bond,
    non-dense local/connected `Sz` observables, and explicit no-ED/no-DMRG/
    no-Lanczos/no-dense-Hamiltonian flags;
  - keeps the existing `chi=8` large-N smoke record for backwards-compatible
    report evidence.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  new chi-table record, policy flags, finite observables, and max-bond bounds.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to report the bounded chi table without
  claiming full large-N completion.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 37/37 PASS.
- Bounded Heisenberg `N=20` chi table:
  - `chi=4`: final energy `-2.0594337898`, energy/site
    `-0.1029716895`, finite `local_Sz_mid=0.0891780579`,
    `connected_SzSz_midbond=-0.0323883644`.
  - `chi=8`: final energy `-3.8076497833`, energy/site
    `-0.1903824892`, finite `local_Sz_mid=0.0186037017`,
    `connected_SzSz_midbond=-0.0255121421`.
  - each chi energy decreased over the three-step CPU smoke budget;
    chi=8 finished below chi=4; no ED/classical DMRG/Lanczos/dense
    Hamiltonian construction was used.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are full Heisenberg N=40/80
  chi-convergence, cross-model large-N chi tables, and literature-grade
  large-N observable/correlation trends beyond bounded smoke evidence.

Next action:
Continue strengthening large-N evidence with resource-bounded cross-model chi
tables, or keep Stage 11 marked REVIEW REQUIRED for the larger requested
benchmark breadth if those runs exceed the allowed CPU-small envelope.

## Checkpoint Stage 11D-11: Cross-Model Bounded Large-N Chi Tables

Goal:
Extend bounded large-N chi-table evidence beyond Heisenberg to the hard-sector
spinless t-V and Hubbard records while avoiding unsupported monotonic
convergence claims.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added spinless t-V `N=20` hard-sector chi=4/8 table;
  - added Hubbard `N=10` hard-sector chi=4/8 table;
  - each table records finite energies, energy/site, sector/additive
    observables, gradient norm, max bond, diagnostics, and explicit no-ED/
    no-DMRG/no-Lanczos/no-dense-Hamiltonian evidence;
  - table pass criteria check finite energies, sector cleanliness, forbidden
    entry/gradient cleanliness, and max bond <= chi, but intentionally do not
    require monotonic energy improvement for the tiny CPU budget.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  cross-model chi-table records and policy/sector-clean fields.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to report 39/39 quick PASS and bounded
  chi=4/8 evidence for all three large-N model families.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 39/39 PASS.
- Cross-model bounded chi tables:
  - spinless t-V `N=20`, `target_n=10`:
    - `chi=4`: final energy `-2.3785951159`, sector abs error
      `3.55e-15`, variance `0.0`;
    - `chi=8`: final energy `-2.3780704961`, sector abs error
      `5.33e-15`, variance `4.26e-14`.
  - Hubbard `N=10`, `(N_up,N_down)=(5,5)`:
    - `chi=4`: final energy `-10.0017836658`, sector abs errors
      `8.88e-16`;
    - `chi=8`: final energy `-10.0031957821`, sector abs errors <=
      `8.88e-16`.
  - all rows are finite, max bond <= chi, forbidden entries/gradients are
    clean, and no ED/classical DMRG/Lanczos/dense Hamiltonian construction was
    used.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are the larger requested
  production-scale large-N chi-convergence tables and literature-grade
  large-N observable/correlation trend tables.

Next action:
If staying inside CPU-small limits, continue by improving large-N observable
trend artifacts; otherwise the next true completion step needs explicit
resource bounds for N=40/80 and Hubbard N=20/40 convergence runs.

## Checkpoint Stage 11D-12: Spinless Large-N Analytic Observable Trend

Goal:
Add a genuine large-N observable/correlation trend artifact using an
independent analytic reference that remains CPU-small and avoids dense
many-body ED.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added open-chain spinless free-fermion sine-mode observable helper;
  - added `trend_spinless_free_fermion_large_n_observables` literature record
    for `N=40,80` at half filling;
  - records analytic energy/site, mid-chain density, neighbor density, and
    connected midbond density correlation;
  - checks that `N=80` energy/site is closer to the thermodynamic `-2/pi`
    value than `N=40`, that mid density stays near half filling, and that the
    connected density correlation is finite/nonzero.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  new large-N analytic trend record and its trend fields.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to report 40/40 quick PASS and distinguish
  this V=0 analytic large-N evidence from the still-missing interacting
  large-N trend tables.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 40/40 PASS.
- Spinless open-chain free-fermion analytic large-N trend:
  - `N=40`: energy/site `-0.6276949278`, distance to `-2/pi`
    `0.0089248446`, mid density `0.5000000000`, connected midbond density
    correlation `-0.1093087776`.
  - `N=80`: energy/site `-0.6321179224`, distance to `-2/pi`
    `0.0045018499`, mid density `0.5000000000`, connected midbond density
    correlation `-0.1053050030`.
  - no many-body ED or dense Hamiltonian construction was used for this
    analytic reference.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are full interacting large-N
  chi-convergence tables and literature-grade interacting large-N
  observable/correlation trend tables.

Next action:
Continue with another independent CPU-small large-N trend if available, or
move to explicit resource-bounded interacting N=40/80 and Hubbard N=20/40
benchmark planning before claiming Stage 11 completion.

## Checkpoint Stage 11D-13: Hubbard U=0 Large-N Analytic Observable Trend

Goal:
Add an independent large-N Hubbard observable/correlation trend at the
noninteracting `U=0` limit without many-body ED or dense Hamiltonian
construction.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added a spin-resolved open-chain Hubbard `U=0` sine-mode observable helper;
  - added `trend_hubbard_free_fermion_large_n_observables` for `N=40,80`,
    half filling with `(N_up,N_down)=(N/2,N/2)`;
  - records analytic energy/site, mid density, double occupancy, neighbor
    density/double occupancy, and connected midbond density correlation;
  - checks that `N=80` energy/site is closer to the thermodynamic `-4/pi`
    value than `N=40`, mid density remains near `1.0`, double occupancy
    remains near `0.25`, and connected density correlation is finite/nonzero.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  new Hubbard large-N analytic record and its trend fields.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to report 41/41 quick PASS and distinguish
  Hubbard `U=0` analytic large-N evidence from still-missing interacting
  large-N tables.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 41/41 PASS.
- Hubbard `U=0` open-chain analytic large-N trend:
  - `N=40`: energy/site `-1.2553898556`, distance to `-4/pi`
    `0.0178496892`, mid density `1.0000000000`, double occupancy
    `0.2500000000`, connected midbond density correlation `-0.2186175551`.
  - `N=80`: energy/site `-1.2642358449`, distance to `-4/pi`
    `0.0090036999`, mid density `1.0000000000`, double occupancy
    `0.2500000000`, connected midbond density correlation `-0.2106100060`.
  - no many-body ED or dense Hamiltonian construction was used for this
    analytic reference.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are full interacting large-N
  chi-convergence tables and literature-grade interacting large-N
  observable/correlation trend tables.

Next action:
Continue only with CPU-small independent trend evidence or request explicit
resource bounds before attempting the larger interacting N=40/80 and Hubbard
N=20/40 convergence jobs required for final Stage 11 completion.

## Checkpoint Stage 11D-14: Focused Large-N Evidence Audit Artifacts

Goal:
Make the current large-N evidence and remaining REVIEW REQUIRED gaps directly
machine-readable and human-readable from the benchmark output directory.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added `outputs/physics_benchmarks/large_n_evidence.json`;
  - added `outputs/physics_benchmarks/large_n_evidence.md`;
  - the JSON/Markdown artifacts extract large-N AD smokes, bounded chi tables,
    spinless `V=0` and Hubbard `U=0` analytic `N=40/80` trend records, and
    explicit remaining interacting large-N review-required items.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to require
  the new output files and assert their evidence/review-required contents.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to reference the focused large-N evidence
  artifacts.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 41/41 PASS.
- New focused large-N evidence files:
  - `outputs/physics_benchmarks/large_n_evidence.json`;
  - `outputs/physics_benchmarks/large_n_evidence.md`.
- `large_n_evidence.json` reports status `REVIEW REQUIRED`, 8 current
  evidence records, and 4 explicit remaining review-required items:
  Heisenberg N=40/80 interacting AD chi-convergence, spinless t-V N=40/80
  interacting AD chi-convergence, Hubbard N=20/40 interacting AD
  chi-convergence, and interacting large-N observable/correlation literature
  trends.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are full interacting large-N
  chi-convergence tables and literature-grade interacting large-N
  observable/correlation trend tables.

Next action:
Proceed only with further CPU-small independent evidence, or obtain explicit
resource bounds before running the larger interacting benchmark jobs needed
for final Stage 11 completion.

## Checkpoint Stage 11D-15: Larger Interacting Large-N Smoke Records

Goal:
Move closer to the requested interacting large-N benchmark envelope with
CPU-small larger-size smoke records while preserving the distinction from full
chi-convergence.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added Heisenberg `N=40`, `chi=8`, three-step AD-only smoke;
  - added spinless t-V `N=40`, hard-sector `target_n=20`, `chi=8` smoke;
  - added Hubbard `N=20`, hard-sector `(N_up,N_down)=(10,10)`, `chi=8`
    smoke;
  - all new records keep explicit no-ED/no-classical-DMRG/no-Lanczos/
    no-dense-Hamiltonian diagnostics.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert
  the larger interacting smoke records and their sector/policy fields.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`;
  `large_n_evidence.json` now contains 11 evidence records and 4 explicit
  REVIEW REQUIRED items.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to report 44/44 quick PASS and the
  larger-size smoke evidence.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 44/44 PASS.
- Larger interacting smoke records:
  - Heisenberg `N=40`, `chi=8`: energy decreased from `-0.0503191825` to
    `-7.2436011523`; energy/site `-0.1810900288`; finite
    `local_Sz_mid=0.0637607698`, `connected_SzSz_midbond=-0.0390051530`.
  - spinless t-V `N=40`, `chi=8`, `target_n=20`: final energy
    `-4.8837190124`; energy/site `-0.1220929753`; sector abs error and
    variance are both `0.0`.
  - Hubbard `N=20`, `chi=8`, `(N_up,N_down)=(10,10)`: final energy
    `-20.0065749785`; energy/site `-1.0003287489`; sector abs errors and
    variances are all `0.0`.
  - all new records are finite, max bond <= chi, forbidden entries/gradients
    are clean, and no ED/classical DMRG/Lanczos/dense Hamiltonian construction
    was used.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are still full interacting
  chi-convergence tables (multiple chi values at the requested larger sizes)
  and literature-grade interacting large-N observable/correlation trends.

Next action:
Further progress toward completion now requires either explicit resource
bounds for true interacting chi-convergence jobs, or another independent
CPU-small literature/trend artifact that does not masquerade as those jobs.

## Checkpoint Stage 11D-16: Larger-Size Bounded Chi Tables

Goal:
Convert the larger interacting smoke records into bounded chi=4/8 evidence at
the largest CPU-small sizes already proven cheap, while continuing to label the
remaining production-scale jobs as REVIEW REQUIRED.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - generalized the Heisenberg chi-table helper and added Heisenberg `N=40`,
    `chi=4/8` table;
  - extended hard-sector chi tables to spinless t-V `N=40`, `target_n=20`;
  - extended hard-sector chi tables to Hubbard `N=20`,
    `(N_up,N_down)=(10,10)`;
  - updated `large_n_evidence.{json,md}` review reasons to reflect bounded
    larger-size chi=4/8 evidence.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  new larger-size chi-table records and their evidence-artifact inclusion.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`;
  `large_n_evidence.json` now contains 14 evidence records and 4 explicit
  REVIEW REQUIRED items.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to report 47/47 quick PASS and the
  larger-size bounded chi tables.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 47/47 PASS.
- Larger-size bounded chi tables:
  - Heisenberg `N=40`: `chi=4` final energy `-4.1196170410`,
    energy/site `-0.1029904260`; `chi=8` final energy `-7.2436011523`,
    energy/site `-0.1810900288`; both histories decrease and chi=8 finishes
    lower than chi=4.
  - spinless t-V `N=40`, `target_n=20`: `chi=4` final energy
    `-4.8828830862`, sector abs error `1.78e-14`; `chi=8` final energy
    `-4.8837190124`, sector abs error `0.0`.
  - Hubbard `N=20`, `(N_up,N_down)=(10,10)`: `chi=4` final energy
    `-20.0049777606`, sector abs errors <= `1.78e-15`; `chi=8` final energy
    `-20.0065749785`, sector abs errors `0.0`.
  - all rows are finite, max bond <= chi, forbidden entries/gradients are
    clean, and no ED/classical DMRG/Lanczos/dense Hamiltonian construction was
    used.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are now production-scale
  interacting chi-convergence beyond this bounded CPU-small envelope
  (notably N=80 for Heisenberg/spinless and N=40 for Hubbard) and
  literature-grade interacting observable/correlation trend tables.

Next action:
Do not claim completion from these bounded tables. Either add another
independent CPU-small trend artifact, or request explicit resource limits for
the remaining production-scale interacting jobs.

## Checkpoint Stage 11D-17: Requested-Size Bounded Chi Tables

Goal:
Add bounded chi=4/8 evidence at the requested large sizes that still fit the
CPU-small envelope: Heisenberg `N=80`, spinless t-V `N=80`, and Hubbard
`N=40`.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - added Heisenberg `N=80`, `chi=8` smoke and `chi=4/8` table;
  - added spinless t-V `N=80`, hard-sector `target_n=40`, `chi=8` smoke and
    `chi=4/8` table;
  - added Hubbard `N=40`, hard-sector `(N_up,N_down)=(20,20)`, `chi=8` smoke
    and `chi=4/8` table;
  - updated `large_n_evidence.{json,md}` review reasons to distinguish bounded
    requested-size evidence from production-depth convergence.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert the
  requested-size smoke/table records and their evidence-artifact inclusion.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`;
  `large_n_evidence.json` now contains 20 evidence records and 4 explicit
  REVIEW REQUIRED items.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to report 53/53 quick PASS and requested-size
  bounded chi tables.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 53/53 PASS.
- Requested-size bounded chi tables:
  - Heisenberg `N=80`: `chi=4` final energy `-8.9464081375`,
    energy/site `-0.1118301017`; `chi=8` final energy `-16.0064711774`,
    energy/site `-0.2000808897`; both histories decrease and chi=8 finishes
    lower than chi=4.
  - spinless t-V `N=80`, `target_n=40`: `chi=4` final energy
    `-9.8891675005`, sector abs error `0.0`; `chi=8` final energy
    `-9.8907643678`, sector abs error `2.13e-14`.
  - Hubbard `N=40`, `(N_up,N_down)=(20,20)`: `chi=4` final energy
    `-40.0119347250`, sector abs errors `1.42e-14`; `chi=8` final energy
    `-40.0160469617`, sector abs errors `3.55e-15`.
  - all rows are finite, max bond <= chi, forbidden entries/gradients are
    clean, and no ED/classical DMRG/Lanczos/dense Hamiltonian construction was
    used.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are production-depth convergence
  over larger chi/step budgets and literature-grade interacting large-N
  observable/correlation trend tables.

Next action:
Do not mark Stage 11 complete from smoke-budget requested-size tables. Either
add independent CPU-small interacting observable evidence or request explicit
resource bounds for production-depth convergence jobs.

## Checkpoint Stage 11D-18: Heisenberg Large-N Entanglement Smoke

Goal:
Strengthen the large-N observable side with non-dense Heisenberg mid-bond
entanglement entropy fields for the existing bounded AD records.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - reused `latticetn.canonical.entanglement_entropy` as report-only
    postprocessing for Heisenberg large-N MPS records;
  - added finite nonnegative `entanglement_entropy_midbond` fields to
    Heisenberg `N=20/40/80` smoke and chi-table rows;
  - included entropy in the pass criteria for Heisenberg large-N records
    without claiming entanglement scaling or literature-grade convergence.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert
  finite nonnegative entropy fields on Heisenberg large-N records and chi
  table rows.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to include Heisenberg large-N entropy smoke
  evidence.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 53/53 PASS.
- Heisenberg bounded chi-table mid-bond entropy samples:
  - `N=20`: `chi=4` entropy `0.6534781430`, `chi=8` entropy
    `0.5158153895`.
  - `N=40`: `chi=4` entropy `0.8007495829`, `chi=8` entropy `0.0`.
  - `N=80`: `chi=4` entropy `0.0`, `chi=8` entropy `0.0`.
- Entropy entries are finite and nonnegative; they are reported as smoke
  observables, not as a literature-grade entanglement-scaling trend.
- Updated fast validation: 52 passed, Score PASS.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are production-depth convergence
  over larger chi/step budgets and literature-grade interacting large-N
  observable/correlation trend tables.

Next action:
Continue only with bounded CPU-small evidence, or request explicit resource
bounds for production-depth convergence/literature trend jobs before claiming
Stage 11 completion.

## Checkpoint Stage 12A-01: Exact Dense Canonical Retraction Primitives

Goal:
Add the exact dense-MPS canonicalization API required for gauge-retracted AD.

Files changed:
- `latticetn/canonical.py`
- `tests/test_stage12a_dense_canonical.py`

Commands run:
- `C:\Apps\Miniforge3\envs\agent-env\python.exe -m pytest -q tests/test_stage12a_dense_canonical.py`

Result:
- 5 passed (including the later periodic-retraction stability control).
- Exact QR and non-truncating SVD left sweeps preserve the normalized dense
  state and TFI Rayleigh energy to `1e-12`.
- Mixed canonicalization has residual below `1e-12`; center normalization
  produces physical norm one.
- The truncated-SVD negative control changes both state and energy.
- An AD update followed by exact canonicalization preserves the AD-updated
  physical state.

Current failing test or bottleneck:
- Dense primitives pass; charge-block canonicalization is not implemented yet.

Next action:
Implement blockwise QR using virtual charge labels and verify spinless and
Hubbard masks, sectors, state, norm, and energy.

## Checkpoint Stage 12A-02: Charge-Block Retraction and Global AD Integration

Goal:
Implement hard-sector canonicalization and make every accepted Global AD state
a canonical unit-norm physical output.

Files changed:
- `latticetn/charge_sectors.py`
- `latticetn/config_schema.py`
- `latticetn/runner.py`
- `latticetn/ad_variational.py`
- `scripts/run_ad_model_benchmark.py`
- `tests/test_stage12a_sector_canonical.py`
- `tests/test_stage12a_runner.py`
- `standalone_stage12a_gauge_retraction_test.py`
- `docs/STAGE12A_GAUGE_RETRACTION_REPORT.md`
- `docs/NUMERICAL_REPORT.md`

Commands run:
- `C:\Apps\Miniforge3\envs\agent-env\python.exe -m pytest -q tests/test_stage12a_sector_canonical.py`
- `C:\Apps\Miniforge3\envs\agent-env\python.exe -m pytest -q tests/test_stage12a_runner.py`
- Combined Stage 12A/canonical/runner CPU regression selection (`-k "not cuda"`)
- CLI and hard-sector CPU regression selection (`-k "not cuda"`)
- Existing Global AD optimizer/gauge regression selection
- `C:\Apps\Miniforge3\envs\agent-env\python.exe standalone_stage12a_gauge_retraction_test.py --N 4 --chi 4 --steps 5 --interval 2`
- `C:\Apps\Miniforge3\envs\agent-env\python.exe scripts/validation_score.py --fast`

Result:
- Charge-block tests: 3 passed.
- Runner semantics/metadata tests: 4 passed (including hard-sector dense-QR rejection).
- Combined Stage 12A and relevant CPU regressions: 29 passed after the added controls.
- CLI/hard-sector CPU regressions: 6 passed.
- Existing Global AD optimizer/gauge regressions: 13 passed.
- Fast validation: Score PASS, exit code 0.
- Exact hard-sector QR preserves spinless `N` and Hubbard `(N_up,N_down)`,
  physical state, energy, and forbidden zeros.
- Final Global AD outputs are canonicalized, center-normalized, checked for
  energy invariance and norm one, and used by observables/result serialization.
- Result metadata distinguishes raw internal norm from final physical norm and
  records retraction/reset events.

Current failing test or bottleneck:
- Two CUDA-marked regression tests cannot execute in the local validation
  environment: Torch reports CUDA available but raises
  `cudaErrorNoKernelImageForDevice`. CPU Stage 12A and required fast validation
  pass.
- There is no reusable MPS checkpoint serializer in the repository; future
  checkpoint support must call the physical-output finalizer before writing.

Next action:
Run the documented manual `N=80` spinless and `N=40` Hubbard A/B commands only
with explicit resource approval. Stage 12B differentiable canonicalization
remains deferred.

## Checkpoint Stage 12A-03: Broad CPU Compatibility and Final Score

Goal:
Check cross-stage compatibility and confirm the required stopping condition.

Files changed:
- `latticetn/runner.py`
- `docs/STAGE12A_GAUGE_RETRACTION_REPORT.md`
- `docs/CLAUDE_PROGRESS.md`

Commands run:
- `C:\Apps\Miniforge3\envs\agent-env\python.exe -m pytest -q -k "not cuda and not gpu"`
- `C:\Apps\Miniforge3\envs\agent-env\python.exe -m pytest -q --lf -k "not cuda and not gpu"`
- `C:\Apps\Miniforge3\envs\agent-env\python.exe scripts/validation_score.py --fast`

Result:
- The broad CPU run exposed three compatibility assertions: two projection-hook
  expectations and one deprecated alias serialization expectation.
- Projection hooks now remain observable at the configured event cadence, and
  serialized deprecated jobs preserve requested `ad_dmrg` while reporting
  resolved `ad_global` explicitly.
- Previously failing selection: 8 passed.
- Final required fast validation: Score PASS, exit code 0.

Current failing test or bottleneck:
- No known CPU Stage 12A or fast-validation failure.
- CUDA remains unavailable for execution because of the local binary/GPU
  architecture mismatch; no GPU work is required by Stage 12A acceptance.

Next action:
Stage 12A implementation is complete. Keep Stage 12B and production-depth A/B
benchmarks deferred until explicitly requested.

## Checkpoint Stabilization-01: Method Identity and Conditioning Repair

Goal:
Pause Stage 11 expansion and repair numerical conditioning, method identity,
CLI/config propagation, optimizer lifecycle, hard-sector initialization,
packaging discovery, and generated-artifact hygiene.

Files changed:
- `latticetn/ad_two_site.py`: added explicit pre-optimization
  `normalize_theta_`, default `precondition="theta_norm"` in
  `train_ad_two_site`, explicit LBFGS tolerances, and optimizer/closure counts.
- `latticetn/config_schema.py`, `latticetn/runner.py`,
  `scripts/run_ad_model_benchmark.py`: added canonical `ad_global` and
  `ad_two_site` identities; kept `ad_dmrg` only as a deprecated alias to
  `ad_global`; propagated init/projection/precondition/stabilization/grad
  clipping/LBFGS settings; made global Adam persistent across global steps.
- `latticetn/charge_sectors.py`: preserved requested hard-sector product
  charge paths through chi trimming.
- `latticetn/model_registry.py`: advertises canonical method names.
- `pyproject.toml`: switched to `latticetn*` package discovery.
- `.gitignore`: ignores local pytest temp output and generated Stage 11 scratch.
- Added focused regression tests for theta preconditioning, algorithm identity,
  hard-sector initializer robustness, and package discovery.
- Removed `.tmp_stage11_benchmark_suite/` from Git tracking with
  `git rm --cached` while leaving files on disk.

Commands run:
- `python -m pip install -r requirements-dev.txt`
- `python -m pytest -q tests\test_two_site_preconditioning.py tests\test_runner_algorithm_identity.py tests\test_hard_sector_initializer_robustness.py tests\test_packaging_discovery.py`
- `python -m pytest -q -p no:cacheprovider tests\test_runner_schema.py tests\test_model_registry.py tests\test_runner_algorithm_identity.py tests\test_two_site_preconditioning.py tests\test_hard_sector_initializer_robustness.py tests\test_packaging_discovery.py`
- `python scripts\run_ad_model_benchmark.py --model heisenberg --N 4 --chi 2 --sweeps 1 --device cpu --dtype complex128 --optimizer adam --local-steps 1 --lr 0.01 --init neel --output tmp\heis_cli.json --no-ed`
- `python scripts\run_ad_model_benchmark.py --model spinless_tv --N 4 --chi 2 --sweeps 1 --device cpu --dtype complex128 --optimizer adam --local-steps 1 --lr 0.01 --init spinless_cdw --target-n 2 --sector-mode hard --output tmp\spinless_hard_cli.json --no-ed`
- `python scripts\run_ad_model_benchmark.py --model heisenberg --N 4 --chi 4 --method ad_global --sweeps 1 --device cpu --dtype complex128 --optimizer adam --local-steps 1 --lr 0.01 --init random --stabilization none --grad-clip 0.5 --output tmp\heis_random_global_cli.json --no-ed`
- `python scripts\validation_score.py --fast`
- `python -B -c "from pathlib import Path; import setuptools.build_meta as bm; out=Path('tmp/wheelhouse'); out.mkdir(parents=True, exist_ok=True); print(bm.build_wheel(str(out)))"`
- Isolated built-tree import from `build/lib` for
  `latticetn.benchmarks.exact_reference`.

Result:
- New targeted tests: 12 passed.
- Runner/model/precondition/package subset: 20 passed, 1 skipped.
- CLI dispatch smoke:
  - Heisenberg sector-none default ran `ad_two_site`
    (`optimizer_path=two_site_ad_local_theta`).
  - spinless hard sector ran `ad_global`
    (`optimizer_path=global_ad_hard_charge_mask`) with zero forbidden amplitude
    and exact particle number.
  - explicit random global run reported actual bond dims `[2, 4, 2]`, honored
    `projection=none`, `grad_clip=0.5`, and serialized global-step metadata.
- Fast validation: Score PASS.
- Setuptools package discovery copied `latticetn/benchmarks` into `build/lib`,
  and importing `latticetn.benchmarks.exact_reference` from that built tree
  succeeded.

Current failing test or bottleneck:
- The sandbox could not write/remove generated build metadata and pytest temp
  roots. After escalated cleanup of untracked generated artifacts and escalated
  runs for the affected commands:
  - Wheel build succeeded and produced
    `tmp2/wheelhouse/latticetn-0.1.0-py3-none-any.whl`.
  - Isolated import directly from the wheel succeeded for `latticetn`,
    `latticetn.benchmarks`, and `latticetn.benchmarks.exact_reference`.
  - Tmp-path based tests
    `tests/test_hard_sector_ad_runner.py tests/test_ad_model_benchmark_cli.py`
    passed with one CUDA skip.
  - Re-run targeted subset:
    `tests/test_runner_schema.py tests/test_model_registry.py
    tests/test_runner_algorithm_identity.py tests/test_two_site_preconditioning.py
    tests/test_hard_sector_initializer_robustness.py
    tests/test_packaging_discovery.py` -> 20 passed, 1 skipped.
  - Re-run `python scripts/validation_score.py --fast` -> Score PASS.
- Remaining limitation: no long N=40/N=80 manual acceptance jobs were run.

Next action:
Run a final requirement-by-requirement audit, inspect the final diff for
generated artifacts, and do not add new Stage 11 cases during this
stabilization sprint.

## Checkpoint Stage 11D-21: Bounded Chi=32 Large-N Evidence

Goal:
Move the requested-size large-N chi tables another rung toward the Stage 11
convergence target by adding `chi=32` rows while preserving CPU-small quick
validation and the distinction from production-depth convergence.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - widened Heisenberg large-N chi tables from `chi=4/8/16` to
    `chi=4/8/16/32`;
  - widened spinless and Hubbard hard-sector chi tables from `chi=4/8/16` to
    `chi=4/8/16/32`;
  - renamed chi-table records to `chi4_8_16_32`;
  - updated large-N evidence review reasons to reflect the wider bounded table.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to require
  the renamed records and four chi rows.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to report bounded `chi=4/8/16/32` evidence
  while keeping production-depth convergence as REVIEW REQUIRED.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 53/53 PASS.
- Updated fast validation: 52 passed, Score PASS.
- Representative requested-size `chi=32` rows:
  - Heisenberg `N=80`: final energy `-29.6968503865`, energy/site
    `-0.3712106298`, max bond `32`, finite local/connected `Sz`, no
    ED/DMRG/Lanczos/dense Hamiltonian.
  - spinless t-V `N=80`, `target_n=40`: final energy `-9.8906964799`,
    energy/site `-0.1236337060`, max bond `32`, sector abs error `0.0`,
    `local_density_mid=0.9999999830`.
  - Hubbard `N=40`, `(N_up,N_down)=(20,20)`: final energy
    `-40.0133399656`, energy/site `-1.0003334991`, max bond `32`,
    sector abs errors `3.55e-15` and `1.07e-14`,
    `local_density_mid=0.9999999927`,
    `double_occupancy_mid=1.2794610278e-08`,
    `local_sz_mid=0.4999999828`.
- Heisenberg requested-size rows remain monotonic through `chi=32`. Spinless
  and Hubbard hard-sector rows remain finite, sector-clean, and physical but
  are not claimed as monotonic under the tiny two-local-step budget.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are production-depth convergence
  over larger optimization budgets and `chi=64`, plus literature-grade
  interacting large-N observable/correlation trend tables.

Next action:
Continue only with bounded CPU-small evidence, or request explicit resource
bounds for production-depth convergence/literature trend jobs before claiming
Stage 11 completion.

## Checkpoint Stabilization-02: Runner Projection Diagnostics

Goal:
Close the remaining runner-level conditioning gap without expanding Stage 11
benchmark scope: verify `ad_global` honors the configured projection mode and
records finite norm/gradient diagnostics through the unified runner API.

Files changed:
- Extended `tests/test_runner_algorithm_identity.py` with a parameterized
  CPU-small `ad_global` test for `projection in {none,tensor_norm,canonical}`.
- The test spies on the runner projection call, checks the selected projection
  is applied once per global optimizer step, and asserts finite final energy,
  best energy, gradient norm, and per-step state norm diagnostics.

Commands run:
- `python -m pytest -q -p no:cacheprovider tests\test_runner_algorithm_identity.py`
- `python -m pytest -q -p no:cacheprovider tests\test_runner_schema.py tests\test_model_registry.py tests\test_runner_algorithm_identity.py tests\test_two_site_preconditioning.py tests\test_hard_sector_initializer_robustness.py tests\test_packaging_discovery.py`
- `python scripts\validation_score.py --fast`

Result:
- 9 passed.
- Focused stabilization subset: 23 passed, 1 skipped. The only warnings were
  the expected `ad_dmrg` deprecation warnings in legacy compatibility tests.
- Fast validation: Score PASS. Pytest emitted the existing tensor-to-float
  warning in `tests/test_mps_dense.py`, expected `ad_dmrg` deprecation
  warnings, and a non-fatal `.pytest_cache` write warning.

Current failing test or bottleneck:
- No new failure found. Stage 11 expansion remains paused; this checkpoint is
  bounded stabilization coverage, not new large-system convergence evidence.

Next action:
Perform final diff/status audit and summarize remaining sprint scope.

## Checkpoint Stabilization-03: CI Smoke Coverage

Goal:
Cover the sprint's CI quick-smoke item without adding long physics jobs or GPU
requirements.

Files changed:
- Added `.github/workflows/ci.yml`.
- The workflow installs CPU test dependencies, installs the package editable,
  runs `python scripts/validation_score.py --fast`, builds a no-deps wheel, and
  imports `latticetn.benchmarks.exact_reference` from that wheel.

Commands run:
- `python -c "from pathlib import Path; text=Path('.github/workflows/ci.yml').read_text(); assert 'python scripts/validation_score.py --fast' in text; assert 'python -m pip wheel . --no-deps --wheel-dir dist' in text; assert 'latticetn.benchmarks.exact_reference' in text; print('ci workflow smoke ok')"`
- `python -m pytest -q -p no:cacheprovider tests\test_runner_schema.py tests\test_model_registry.py tests\test_runner_algorithm_identity.py tests\test_two_site_preconditioning.py tests\test_hard_sector_initializer_robustness.py tests\test_packaging_discovery.py`

Result:
- CI workflow smoke check passed.
- Focused stabilization subset after CI addition: 23 passed, 1 skipped, with
  expected `ad_dmrg` deprecation warnings in legacy compatibility tests.

Current failing test or bottleneck:
- The workflow itself was not executed by GitHub Actions in this local
  workspace. Local equivalents for fast validation and wheel import passed in
  Stabilization-01/02.

Next action:
Summarize final stabilization status and remaining limitations.

## Checkpoint Stage 11D-20: Bounded Chi=16 Large-N Evidence

Goal:
Move the existing requested-size large-N chi tables one notch closer to the
Stage 11 convergence envelope by adding `chi=16` rows while keeping the suite
CPU-small and not claiming production-depth convergence.

Files changed:
- Updated `scripts/run_physics_benchmark_suite.py`:
  - widened Heisenberg large-N chi tables from `chi=4/8` to `chi=4/8/16`;
  - widened spinless and Hubbard hard-sector chi tables from `chi=4/8` to
    `chi=4/8/16`;
  - renamed chi-table records to `chi4_8_16`;
  - generalized the Heisenberg monotonic check to
    `higher_chi_energy_not_worse`;
  - kept hard-sector tables as finite/sector-clean bounded evidence without
    asserting monotonic energy convergence from the tiny two-local-step budget.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to require
  the renamed records and three chi rows.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to report bounded `chi=4/8/16` evidence.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 53/53 PASS.
- Updated fast validation: 52 passed, Score PASS.
- Representative requested-size `chi=16` rows:
  - Heisenberg `N=80`: final energy `-25.5082156387`, energy/site
    `-0.3188526955`, max bond `16`, no ED/DMRG/Lanczos/dense Hamiltonian.
  - spinless t-V `N=80`, `target_n=40`: final energy `-9.8924158348`,
    energy/site `-0.1236551979`, max bond `16`, sector abs error
    `7.11e-15`, `local_density_mid=0.9999999525`.
  - Hubbard `N=40`, `(N_up,N_down)=(20,20)`: final energy
    `-40.0127510985`, energy/site `-1.0003187775`, max bond `16`, sector abs
    errors `7.11e-15`, `local_density_mid=0.9999999864`,
    `double_occupancy_mid=7.4721999461e-09`,
    `local_sz_mid=0.4999999855`.
- Heisenberg and spinless requested-size rows improve through `chi=16`.
  Hubbard remains finite, sector-clean, and physical but is not claimed as
  monotonic under this tiny hard-sector smoke budget.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are production-depth convergence
  over larger chi/step budgets, especially `chi=32/64`, and
  literature-grade interacting large-N observable/correlation trend tables.

Next action:
Continue only with bounded CPU-small evidence, or request explicit resource
bounds for production-depth convergence/literature trend jobs before claiming
Stage 11 completion.

## Checkpoint Stage 11D-19: Hard-Sector Local Observable Smoke

Goal:
Strengthen spinless and Hubbard large-N hard-sector evidence with non-dense
one-site observables exposed through the Stage 10 runner API.

Files changed:
- Updated `latticetn/runner.py`:
  - added normalized final-MPS one-site expectation contractions;
  - exposed `local_density_mid` for spinless/Hubbard hard-sector runs;
  - exposed Hubbard `double_occupancy_mid` and `local_sz_mid`.
- Updated `scripts/run_physics_benchmark_suite.py`:
  - requested the new local observables for large-N hard-sector smoke and
    chi-table records;
  - included physical range checks in record pass criteria.
- Updated `tests/physics/test_stage11_benchmark_suite_runner.py` to assert
  finite local observables and physical ranges for spinless and Hubbard
  large-N hard-sector records.
- Regenerated quick benchmark outputs under `outputs/physics_benchmarks/`.
- Updated `docs/STAGE11_ACCEPTANCE_AUDIT.md` and
  `docs/STAGE11_FINAL_REPORT.md` to report the new local-observable evidence.
- Updated this progress log.

Commands run:
- `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q tests/physics/test_stage11_benchmark_suite_runner.py`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick`
- `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast`

Result:
- Stage 11 runner test: 2 passed.
- Quick benchmark suite: 53/53 PASS.
- Updated fast validation: 52 passed, Score PASS.
- Representative hard-sector local-observable samples:
  - spinless t-V `N=80`, `chi=8`: `local_density_mid`
    `0.9999999399732036`, sector abs error `2.13e-14`.
  - Hubbard `N=40`, `chi=8`: `local_density_mid`
    `0.9999999804780064`, `double_occupancy_mid`
    `3.6724629845826875e-08`, `local_sz_mid`
    `0.49999995074569953`, sector abs errors `3.55e-15`.
- All added local observable fields are produced from final MPS contractions;
  no ED, classical DMRG, Lanczos, or dense Hamiltonian construction was added
  to large-N hard-sector records.

Current failing test or bottleneck:
- No quick/full-pytest failure is known with writable workspace basetemp.
  Remaining Stage 11 REVIEW REQUIRED items are production-depth convergence
  over larger chi/step budgets and literature-grade interacting large-N
  observable/correlation trend tables.

Next action:
Continue only with bounded CPU-small evidence, or request explicit resource
bounds for production-depth convergence/literature trend jobs before claiming
Stage 11 completion.
## Checkpoint Stage 12B-01: One-site TDVP Baseline

Goal:
Implement the traditional fixed-bond one-site TDVP baseline with matrix-free
MPO environments, a Hermitian Lanczos exponential action, symmetric
projector-splitting sweeps, and small-system ED validation.

Files changed:
- Added `latticetn/tdvp/effective_hamiltonian.py` with one-site, zero-site, and
  two-site matrix-free effective-Hamiltonian contractions.
- Added `latticetn/tdvp/krylov.py` with a device-preserving, fully
  reorthogonalized Lanczos exponential action.
- Added `latticetn/tdvp/tdvp.py` and package exports with the `TDVP` /
  `TDVPResult` API and the symmetric one-site projector-splitting integrator.
- Added focused Krylov, effective-Hamiltonian, norm/energy, canonical-gauge,
  fixed-bond, N=8 ED-fidelity, and local-observable tests.

Commands run:
- `python -m pytest -q -p no:cacheprovider tests/test_mps_canonicalization.py tests/test_dmrg_environments.py tests/test_dmrg_matrix_free_heff.py tests/test_dmrg_two_site_update.py`
- `python scripts/validation_score.py --fast`
- `python -m pytest -q -p no:cacheprovider tests/test_tdvp_krylov.py tests/test_tdvp_effective_hamiltonian.py tests/test_tdvp_one_site.py`

Result:
- Existing canonical/DMRG baseline: 15 passed.
- Pre-change fast validation: 52 passed, Score PASS.
- New Stage 12B-1 focused validation: 7 passed.
- A full-bond N=8 Neel state evolved to t=0.05 matches dense exact evolution
  with fidelity above `1 - 1e-11`; norm drift is below `1e-12` and local Sz
  agrees with the dense reference below `1e-11`.

Current failing test or bottleneck:
- No Stage 12B-1 focused failure.  The new TDVP tests still need a full fast
  regression checkpoint, and Stage 12B-2 two-site evolution is not yet wired
  into the driver.

Next action:
Run fast validation, then implement two-site projector splitting with SVD
truncation and adaptive bond growth.
## Checkpoint Stage 12B-02: Two-site TDVP and Final Validation

Goal:
Complete Stage 12B with two-site projector splitting, adaptive bond growth and
truncation, a Heisenberg Neel-quench benchmark, documentation, and full
repository regression evidence.

Files changed:
- Extended `latticetn/tdvp/tdvp.py` with symmetric two-site sweeps, overlapping
  one-site backward evolution, adaptive SVD rank selection, per-update
  discarded-weight diagnostics, and normalized post-truncation states.
- Added `tests/test_tdvp_two_site.py` and opt-in `tests/test_tdvp_gpu.py`.
- Added `scripts/run_tdvp_heisenberg_quench.py` and included all CPU TDVP tests
  plus the quench in the formal validation score.
- Added `docs/STAGE12B_TDVP_REPORT.md` and updated API, index, numerical report,
  user guide, README, roadmap, and repository status.
- Added `scripts/__init__.py` so the repository's local scripts package wins
  over an unrelated installed top-level `scripts` package during full pytest
  collection.

Commands run:
- `python -m pytest -q -p no:cacheprovider tests/test_tdvp_krylov.py tests/test_tdvp_effective_hamiltonian.py tests/test_tdvp_one_site.py tests/test_tdvp_two_site.py`
- `python scripts/run_tdvp_heisenberg_quench.py --N 8 --dt 0.02 --steps 10 --chi-max 8 --truncation-tol 1e-10 --device cpu`
- `python -m pytest -q -p no:cacheprovider`
- `python scripts/validation_score.py --full`
- `git diff --check`

Result:
- Focused TDVP tests: 10 passed.
- Full repository pytest: PASS at 100%; opt-in GPU tests clean-skipped and no
  failures occurred.
- Formal full validation: 62 fast tests passed; the N=6 AD reference solve
  passed; the N=8 two-site TDVP quench passed.
- N=8 quench at t=0.2, chi_max=8: norm drift `8.882e-16`, energy drift
  `5.356e-10`, ED fidelity `0.999999993776`, midpoint Sz `0.5 ->
  0.480264616391`, midpoint entropy `0 -> 0.055401154702` nats.
- Full-chi N=8 one-site and two-site tests both achieve fidelity above
  `1 - 1e-11`; adaptive two-site bonds grow from chi=1 and obey the requested
  cap and discarded-weight tolerance.

Current failing test or bottleneck:
- None found. GPU execution is intentionally not performed by default; the
  device path is covered by an opt-in parity test under the repository GPU
  selector.

Next action:
Perform the final requirement/diff audit, then commit and push Stage 12B.
