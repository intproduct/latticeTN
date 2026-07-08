# Stage 11 Physics Validation Report

Status: REVIEW REQUIRED, with all current automated Stage 11 quick checks
passing.

Stabilization note: Stage 11 expansion is paused during the current
semantic-integrity sprint. The large-N 2-3 step cases below are smoke and
integration evidence only; they must not be cited as convergence evidence.

This report summarizes the current Stage 11 evidence for physics benchmarks,
literature/trend reproduction, observable validation, and full-stack scientific
audit. It does not replace `docs/STAGE11_ACCEPTANCE_AUDIT.md`; that file is the
live requirement-to-evidence map.

## Commands

| Command | Result |
|---|---|
| `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/validation_score.py --fast` | 52 passed, Score PASS |
| `C:\Apps\Miniforge3\envs\comfyui\python.exe scripts/run_physics_benchmark_suite.py --suite quick` | 53/53 PASS |
| `C:\Apps\Miniforge3\envs\comfyui\python.exe -m pytest -q --basetemp D:\AI\latticeTN\.tmp_pytest_full_stage11_latest` | exit 0 |

Default pytest temp root note: full pytest without `--basetemp` fails during
fixture setup because the default Windows temp root
`C:\Users\Frank F\AppData\Local\Temp\pytest-of-Frank F` is not writable.
Using a workspace-local basetemp avoids that environment issue.

## Physics Conventions

| Model | Convention |
|---|---|
| Heisenberg | `H = J sum_i S_i . S_{i+1}`, `S = sigma/2`, open boundary |
| TFI | `H = -J sum_i Sz_i Sz_{i+1} - h sum_i Sx_i`, spin convention |
| spinless t-V | local basis `|0>, |1>`; nearest-neighbor hopping uses the adjacent JW-reduced product |
| Hubbard | local basis `|0>, |up>, |down>, |up down>`; site-major mode ordering; spin-resolved hopping keeps the surviving site parity |

Detailed convention record: `docs/PHYSICS_CONVENTIONS.md`.

## Independent References

| Reference Type | Evidence |
|---|---|
| Independent dense Hamiltonians | `tests/physics/test_stage11_hamiltonian_audit.py` |
| Small-N ED | `latticetn/benchmarks/exact_reference.py` and `tests/physics/test_stage11_small_n_energy_benchmarks.py` |
| Fixed sectors | spinless `target_n`; Hubbard `(target_nup,target_ndown)` sector restriction |
| Literature / analytic metadata | `benchmarks/references/reference_registry.json` |

## Hamiltonian Audit

All reported by `outputs/physics_benchmarks/PHYSICS_VALIDATION_REPORT.md`.

| Case | Result |
|---|---|
| Heisenberg MPO vs independent dense, N=4 | PASS |
| TFI MPO vs independent dense, N=4 | PASS |
| spinless t-V MPO/JW dense, N=4 | PASS |
| Hubbard MPO/JW dense, N=3 | PASS |
| ModelSpec -> build_mpo equivalence | PASS via Stage 11 runner and model-builder tests |

## Ground-State and Full-Stack Audit

| Case | Result |
|---|---|
| Heisenberg small ED, N=6 | PASS |
| TFI small ED, N=6 | PASS |
| spinless t-V sector ED, N=6, target_n=3 | PASS |
| Hubbard sector ED, N=3, target_nup=1,target_ndown=1 | PASS |
| Heisenberg AD vs ED, N=4 | PASS, abs error `6.15e-05` |
| Heisenberg DMRG vs ED, N=4 | PASS, abs error `0.0` |
| Heisenberg AD vs DMRG, N=4 | PASS, abs diff `6.15e-05` |
| TFI AD vs ED, N=4 | PASS, abs error `1.96e-04` |
| TFI DMRG vs ED, N=4 | PASS, abs error `0.0` |
| TFI AD vs DMRG, N=4 | PASS, abs diff `1.96e-04` |
| spinless t-V AD vs ED, N=4 | PASS, abs error `1.80e-05` |
| spinless t-V DMRG vs ED, N=4 | PASS, abs error `1.33e-15` |
| spinless t-V AD vs DMRG, N=4 | PASS, abs diff `1.80e-05` |
| Hubbard hard-sector AD vs sector ED, N=2 | PASS, abs error `8.15e-12`, sector errors zero |
| Hubbard DMRG vs ED, N=2 | PASS, abs error `8.88e-16` |
| Hubbard AD vs DMRG, N=2 | PASS, abs diff `8.15e-12` |

## Observables and Entanglement

| Area | Evidence | Result |
|---|---|---|
| Spin product observables | `observable_heisenberg_product_neel` | PASS |
| spinless density observables | `observable_spinless_cdw_density` | PASS |
| Hubbard double occupancy | `observable_hubbard_double_occupancy` | PASS |
| Connected correlations | `tests/physics/test_stage11_observables_correlations.py` | PASS |
| Entanglement entropy | product-state zero and Bell-pair `ln(2)` tests; `entanglement_bell_pair` record | PASS |

## Literature and Trend Audit

| Case | Result |
|---|---|
| Heisenberg Bethe thermodynamic-limit metadata | PASS |
| Heisenberg finite-OBC small-N trend toward `1/4-ln(2)` | PASS |
| TFI transverse magnetization trend across `h=0.5,1.0,1.5` | PASS |
| spinless open-chain free-fermion analytic limit | PASS |
| spinless open-chain free-fermion large-N observables, N=40/80 | PASS: energy/site moves closer to `-2/pi`; mid density remains `0.5`; connected midbond density correlation finite |
| Hubbard large-U double-occupancy trend | PASS |
| Hubbard U=0 large-N observables, N=40/80 | PASS: energy/site moves closer to `-4/pi`; mid density remains `1.0`; double occupancy remains `0.25`; connected midbond density correlation finite |

These are small-N trend checks. Finite open chains are not required to equal
thermodynamic-limit values exactly.

## Large-N AD and Policy Audit

| Case | Result |
|---|---|
| CUDA quick-suite behavior | PASS: CUDA is available on this machine but Stage 11 quick validation remains CPU-only |
| Large-N reference policy | PASS: large-N AD records must not use ED, classical DMRG, Lanczos, or dense Hamiltonian construction |
| Focused large-N evidence artifact | PASS: `outputs/physics_benchmarks/large_n_evidence.{json,md}` generated with 20 bounded/analytic evidence records and 4 explicit REVIEW REQUIRED items |
| Heisenberg N=20, chi=8 AD-only smoke | PASS: energy decreased from `-0.3307489912` to `-3.8076497833` in 3 steps; no ED/DMRG/Lanczos/dense Hamiltonian |
| Heisenberg N=20, chi=4/8/16/32 bounded chi table | PASS: all four chi rows are finite, each energy decreased, higher chi is non-worse, mid-bond entropies finite/nonnegative; no ED/DMRG/Lanczos/dense Hamiltonian |
| Heisenberg N=40, chi=8 bounded AD smoke | PASS: energy decreased from `-0.0503191825` to `-7.2436011523`; finite local/connected `Sz`; no ED/DMRG/Lanczos/dense Hamiltonian |
| Heisenberg N=40, chi=4/8/16/32 bounded chi table | PASS: all four chi rows are finite, each energy decreased, higher chi is non-worse, mid-bond entropies finite/nonnegative; no ED/DMRG/Lanczos/dense Hamiltonian |
| Heisenberg N=80, chi=4/8/16/32 bounded chi table | PASS: chi=4 final energy `-8.9464081375`, chi=8 final energy `-16.0064711774`, chi=16 final energy `-25.5082156387`, chi=32 final energy `-29.6968503865`; mid-bond entropies finite/nonnegative; each energy decreased; no ED/DMRG/Lanczos/dense Hamiltonian |
| spinless t-V N=20, chi=8 hard-sector AD smoke | PASS: sector target_n=10 preserved, no ED/DMRG/Lanczos/dense Hamiltonian |
| spinless t-V N=40, chi=8 hard-sector AD smoke | PASS: sector target_n=20 preserved exactly; no ED/DMRG/Lanczos/dense Hamiltonian |
| spinless t-V N=40, chi=4/8/16/32 bounded chi table | PASS: all chi values preserve `target_n=20`; no ED/DMRG/Lanczos/dense Hamiltonian |
| spinless t-V N=80, chi=4/8/16/32 bounded chi table | PASS: all chi values preserve `target_n=40`; chi=32 final energy `-9.8906964799`; no ED/DMRG/Lanczos/dense Hamiltonian |
| spinless t-V N=20, chi=4/8/16/32 bounded chi table | PASS: all chi values preserve `target_n=10`; no monotonic energy claim is made for this tiny two-local-step budget |
| Hubbard N=10, chi=8 hard-sector AD smoke | PASS: sector `(N_up,N_down)=(5,5)` preserved, no ED/DMRG/Lanczos/dense Hamiltonian |
| Hubbard N=20, chi=8 hard-sector AD smoke | PASS: sector `(N_up,N_down)=(10,10)` preserved exactly; no ED/DMRG/Lanczos/dense Hamiltonian |
| Hubbard N=10, chi=4/8/16/32 bounded chi table | PASS: all chi values preserve `(N_up,N_down)=(5,5)`; no ED/DMRG/Lanczos/dense Hamiltonian |
| Hubbard N=20, chi=4/8/16/32 bounded chi table | PASS: all chi values preserve `(N_up,N_down)=(10,10)`; no ED/DMRG/Lanczos/dense Hamiltonian |
| Hubbard N=40, chi=4/8/16/32 bounded chi table | PASS: all chi values preserve `(N_up,N_down)=(20,20)`; chi=32 local density `0.9999999927`, double occupancy `1.2794610278e-08`; no monotonic energy claim is made for this two-local-step hard-sector budget |
| Heisenberg N=20/40/80 local/connected `Sz` and entropy smoke | PASS: non-dense MPS local/connected two-point contractions and canonical mid-bond entropies finite/nonnegative |
| spinless/Hubbard large-N local observables | PASS: non-dense hard-sector additive observables are sector-clean; spinless mid-site density is finite in `[0,1]`; Hubbard mid-site density, double occupancy, and local `Sz` are finite and in physical ranges |

## REVIEW REQUIRED

The current Stage 11 quick suite is scientifically much broader than the
original Heisenberg-only validation and all automated gates above pass. The
remaining limitation is breadth of large-N convergence:

| Item | Status |
|---|---|
| Heisenberg production-depth chi-convergence beyond bounded N=20/40/80 chi=4/8/16/32 smoke-budget evidence | REVIEW REQUIRED |
| interacting spinless t-V production-depth chi-convergence beyond bounded N=20/40/80 chi=4/8/16/32 smoke-budget evidence and V=0 analytic N=40/80 observables | REVIEW REQUIRED |
| Hubbard production-depth chi-convergence beyond bounded N=10/20/40 chi=4/8/16/32 smoke-budget evidence and U=0 analytic N=40/80 observables | REVIEW REQUIRED |
| Literature-grade large-N observable/correlation trends for interacting Heisenberg/spinless/Hubbard cases beyond current spinless V=0 and Hubbard U=0 analytic N=40/80 references | REVIEW REQUIRED |

These are not hidden failures; the quick suite now includes bounded chi=4/8/16/32
tables and larger-size smokes for Heisenberg, spinless t-V, and Hubbard. The
large-N hard-sector records include local MPS-contracted observables; for
example the latest spinless N=80 chi=8 smoke has mid density
`0.9999999399732036`, and the Hubbard N=40 chi=8 smoke has mid density
`0.9999999804780064`, double occupancy `3.6724629845826875e-08`, and local
`Sz` `0.49999995074569953`. The quick suite also includes large-N analytic
spinless V=0 and Hubbard U=0 observable/correlation trends at N=40/80. The
larger interacting production-scale tables are not yet run as part of the
CPU-small Stage 11 quick suite.
