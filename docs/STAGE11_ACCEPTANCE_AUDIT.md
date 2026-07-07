# Stage 11 Acceptance Audit

This document maps the Stage 11 physics-benchmark objective to current
evidence. It is an audit aid, not a claim that Stage 11 is complete.

Last checkpoint evidence:

| Gate | Evidence | Status |
|---|---|---|
| Fast validation | `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast` -> 52 passed, Score PASS | PASS |
| Full pytest | `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q --basetemp D:\AI\latticeTN\.tmp_pytest_full_stage11_latest` -> exit 0 | PASS |
| Quick physics benchmark suite | `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick` -> 53/53 PASS | PASS |
| Physics report | `outputs/physics_benchmarks/PHYSICS_VALIDATION_REPORT.md` generated | PASS |
| Requirement-by-requirement Stage 11 report | `docs/STAGE11_FINAL_REPORT.md` generated; status REVIEW REQUIRED for full large-N convergence breadth | PASS as report artifact |
| JSON/CSV/Markdown summaries | `outputs/physics_benchmarks/benchmark_summary.{json,csv,md}` generated | PASS |
| Focused large-N evidence audit | `outputs/physics_benchmarks/large_n_evidence.{json,md}` generated with 20 current evidence records and 4 explicit REVIEW REQUIRED items | PASS as audit artifact |
| Main branch | `git status --short --branch` reports `## main...origin/main` | PASS |

## Current Coverage

| Requirement | Evidence | Status |
|---|---|---|
| Physics conventions documented | `docs/PHYSICS_CONVENTIONS.md`; `docs/PHYSICS_SPEC.md`; Stage 11 Hamiltonian tests | PASS |
| Heisenberg MPO vs independent dense reference | `tests/physics/test_stage11_hamiltonian_audit.py`; `hamiltonian_heisenberg_N4` report record | PASS |
| TFI MPO vs independent dense reference | `tests/physics/test_stage11_hamiltonian_audit.py`; `hamiltonian_tfi_N4` report record | PASS |
| spinless JW sign and hopping convention | `tests/physics/test_stage11_hamiltonian_audit.py`; `tests/physics/test_stage11_observables_correlations.py`; corrected `latticetn/operators.py`, `latticetn/mpo.py`, `latticetn/observables.py` | PASS |
| Hubbard up/down JW sign convention | `tests/physics/test_stage11_hamiltonian_audit.py`; `hamiltonian_hubbard_N3` report record | PASS |
| ModelSpec -> MPO physical equivalence | `tests/test_model_builder_mpo_dense.py`; `build_model_spec` path used by `scripts/run_physics_benchmark_suite.py` Hamiltonian records | PASS |
| Small-N ED benchmarks | `latticetn/benchmarks/exact_reference.py`; `tests/physics/test_stage11_small_n_energy_benchmarks.py`; small ED report records | PASS |
| AD variational bound sanity | `tests/test_heisenberg_variational_smoke.py`; full-stack AD-vs-ED records for Heisenberg, TFI, spinless t-V, and hard-sector Hubbard | PASS for small-N |
| Classical DMRG vs ED | full-stack DMRG-vs-ED records for Heisenberg, TFI, spinless t-V, and Hubbard | PASS for small-N |
| AD vs classical DMRG | full-stack AD-vs-DMRG records for Heisenberg, TFI, spinless t-V, and Hubbard | PASS for small-N |
| Local observables | `tests/physics/test_stage11_observables_correlations.py`; observable report records | PASS |
| Two-point and connected correlations | `dense_connected_correlation`, `mps_connected_correlation`; Stage 11 observable tests | PASS |
| Entanglement entropy known-state tests | `test_entanglement_entropy_known_states`; `entanglement_bell_pair` report record | PASS |
| Literature/reference metadata | `benchmarks/references/reference_registry.json`; registry test | PASS |
| Literature and analytic trend checks | Heisenberg Bethe finite-OBC trend, TFI magnetization trend, spinless free-fermion limit, spinless free-fermion large-N N=40/80 observable/correlation trend, Hubbard double-occupancy trend, Hubbard U=0 large-N N=40/80 observable/correlation trend | PASS for current trend suite |
| Automated benchmark runner | `scripts/run_physics_benchmark_suite.py` with `quick`, `exact`, `observables`, `fullstack`, `literature`, and `full` suites | PASS |
| CPU strict tests | Included in `validation_score.py --fast` | PASS |
| Full repository tests | Full pytest passes with workspace-local `--basetemp`; default temp root is not writable on this machine | PASS with basetemp |
| CUDA quick-suite behavior | `policy_cuda_quick_suite_cpu_only` record; current machine reports CUDA available but not used by Stage 11 quick validation | PASS |
| Large-N AD policy | `policy_large_n_ad_runner_no_dense_or_classical_reference` record states no ED, no classical DMRG, no Lanczos, and no dense Hamiltonian construction for large-N AD benchmark records | PASS as policy record |
| Large-N AD smoke | Large-N AD records for Heisenberg N=20/40/80, spinless t-V N=20/40/80 hard sector, and Hubbard N=10/20/40 hard sector; no ED/DMRG/Lanczos/dense Hamiltonian | PASS as smoke |
| Bounded large-N chi tables | Heisenberg N=20/40/80 chi=4/8/16/32 AD tables record finite energies, decreasing per-chi histories, higher-chi non-worse final energies, finite local/connected `Sz`, finite nonnegative mid-bond entanglement entropy, and no ED/DMRG/Lanczos/dense Hamiltonian; spinless N=20/40/80 and Hubbard N=10/20/40 hard-sector chi=4/8/16/32 tables record finite energies, sector-clean additive observables, non-dense local observables, forbidden-entry cleanliness, and no ED/DMRG/Lanczos/dense Hamiltonian | PASS as bounded evidence |
| Large-N observable smoke | Non-dense MPS observables in large-N records: Heisenberg local/connected `Sz` and mid-bond entropy; spinless hard-sector mid-site density; Hubbard hard-sector mid-site density, double occupancy, and local `Sz` | PASS as smoke |
| Focused large-N evidence artifact | `large_n_evidence.json` extracts large-N AD smoke including requested-size N=80/N=40 interacting smokes, bounded chi tables, analytic N=40/80 free-limit trends, and explicit remaining interacting large-N review items | PASS as audit artifact |

## Remaining Gaps Before Stage 11 Completion

| Gap | Why It Remains |
|---|---|
| Full large-N convergence tables | Current suite includes ED-free AD smokes and bounded chi=4/8/16/32 tables across Heisenberg N=20/40/80, spinless N=20/40/80, and Hubbard N=10/20/40, but not production-depth convergence over larger chi and optimization budgets. |
| Literature-grade interacting large-N observable/correlation trends | Current report has non-dense large-N observable smokes, small-N trend/literature checks, and independent spinless V=0 and Hubbard U=0 analytic N=40/80 observable/correlation trends, but not interacting Heisenberg/spinless/Hubbard large-N observable/correlation trend tables. |

## Current Interpretation

Stage 11 is materially stronger than before this audit: the quick report now
contains strict Hamiltonian checks, small ED ground states, observables,
entanglement, literature/trend checks, and small full-stack AD/DMRG/ED
comparisons for Heisenberg, TFI, spinless t-V, and hard-sector Hubbard, plus
explicit quick-suite policy records for CUDA/large-N safeguards and ED-free
large-N AD smokes for Heisenberg, spinless t-V, and Hubbard, including
non-dense observable fields. The current hard-sector samples include spinless
N=80 mid density near `1.0` and Hubbard N=40 mid density near `1.0`, double
occupancy near `0`, and local `Sz` near `0.5`, all from final MPS contractions.
The suite also includes bounded chi=4/8/16/32 tables for all three large-N model
families and independent analytic spinless V=0 and Hubbard U=0 N=40/80
observable/correlation trends. The remaining work is full interacting large-N
chi-convergence and literature-grade interacting large-N observable/correlation
trend breadth.
