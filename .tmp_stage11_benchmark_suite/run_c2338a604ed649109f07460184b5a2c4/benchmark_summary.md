# Physics Benchmark Summary

| case_id | suite | model | method | pass |
|---|---|---|---|:---:|
| hamiltonian_heisenberg_N4 | exact | heisenberg | MPO vs independent dense reference | True |
| hamiltonian_tfi_N4 | exact | tfi | MPO vs independent dense reference | True |
| hamiltonian_spinless_tv_N4 | exact | spinless_tv | MPO vs independent dense reference | True |
| hamiltonian_hubbard_N3 | exact | hubbard | MPO vs independent dense reference | True |
| small_ed_heisenberg_N6 | exact | heisenberg | small-N exact diagonalization | True |
| small_ed_tfi_N6 | exact | tfi | small-N exact diagonalization | True |
| small_ed_spinless_tv_N6 | exact | spinless_tv | small-N exact diagonalization | True |
| small_ed_hubbard_N3 | exact | hubbard | small-N exact diagonalization | True |
| observable_heisenberg_product_neel | observables | heisenberg | analytic product-state observable | True |
| observable_spinless_cdw_density | observables | spinless_tv | analytic product-state observable | True |
| observable_hubbard_double_occupancy | observables | hubbard | analytic product-state observable | True |
| entanglement_bell_pair | observables | heisenberg | known entangled state | True |
| fullstack_heisenberg_ad_vs_ed_N4 | fullstack | heisenberg | AD-MPS Rayleigh optimization vs ED | True |
| fullstack_heisenberg_dmrg_vs_ed_N4 | fullstack | heisenberg | classical two-site DMRG reference vs ED | True |
| fullstack_heisenberg_ad_vs_dmrg_N4 | fullstack | heisenberg | AD-MPS vs classical DMRG baseline | True |
| fullstack_tfi_ad_vs_ed_N4 | fullstack | tfi | AD-MPS Rayleigh optimization vs ED | True |
| fullstack_tfi_dmrg_vs_ed_N4 | fullstack | tfi | classical two-site DMRG reference vs ED | True |
| fullstack_tfi_ad_vs_dmrg_N4 | fullstack | tfi | AD-MPS vs classical DMRG baseline | True |
| fullstack_spinless_tv_ad_vs_ed_N4 | fullstack | spinless_tv | AD-MPS Rayleigh optimization vs ED | True |
| fullstack_spinless_tv_dmrg_vs_ed_N4 | fullstack | spinless_tv | classical two-site DMRG reference vs ED | True |
| fullstack_spinless_tv_ad_vs_dmrg_N4 | fullstack | spinless_tv | AD-MPS vs classical DMRG baseline | True |
| fullstack_hubbard_ad_hard_sector_vs_ed_N2 | fullstack | hubbard | hard-sector AD runner vs fixed-sector ED | True |
| fullstack_hubbard_dmrg_vs_ed_N2 | fullstack | hubbard | classical two-site DMRG reference vs ED | True |
| fullstack_hubbard_ad_vs_dmrg_N2 | fullstack | hubbard | hard-sector AD runner vs classical DMRG baseline | True |
| policy_cuda_quick_suite_cpu_only | policy | all | runtime policy check | True |
| policy_large_n_ad_runner_no_dense_or_classical_reference | policy | all | large-N validation policy record | True |
| large_n_ad_heisenberg_N20_chi8_no_reference | large_n_ad | heisenberg | large-N AD-only smoke | True |
| large_n_chi_table_heisenberg_N20_chi4_8_no_reference | large_n_ad | heisenberg | resource-bounded large-N AD chi table | True |
| large_n_ad_spinless_tv_N20_chi8_hard_sector_no_reference | large_n_ad | spinless_tv | large-N hard-sector AD-only smoke | True |
| large_n_ad_hubbard_N10_chi8_hard_sector_no_reference | large_n_ad | hubbard | large-N hard-sector AD-only smoke | True |
| large_n_chi_table_spinless_tv_N20_chi4_8_hard_sector_no_reference | large_n_ad | spinless_tv | resource-bounded large-N hard-sector AD chi table | True |
| large_n_chi_table_hubbard_N10_chi4_8_hard_sector_no_reference | large_n_ad | hubbard | resource-bounded large-N hard-sector AD chi table | True |
| literature_heisenberg_bethe_energy | literature | heisenberg | reference metadata | True |
| literature_spinless_free_fermion_open_chain | literature | spinless_tv | reference metadata | True |
| literature_hubbard_free_fermion_limit | literature | hubbard | reference metadata | True |
| trend_heisenberg_bethe_finite_obc | literature | heisenberg | small-N ED finite-size trend | True |
| trend_tfi_transverse_magnetization | literature | tfi | small-N ED parameter trend | True |
| trend_spinless_free_fermion_limit | literature | spinless_tv | analytic free-fermion limit | True |
| trend_spinless_free_fermion_large_n_observables | literature | spinless_tv | analytic large-N free-fermion observable trend | True |
| trend_hubbard_double_occupancy_large_u | literature | hubbard | small-N ED interaction trend | True |
| trend_hubbard_free_fermion_large_n_observables | literature | hubbard | analytic large-N U=0 Hubbard observable trend | True |

PASS: 41/41
