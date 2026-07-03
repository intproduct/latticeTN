from latticetn.charge_sectors import (
    spinless_hard_sector_product_mps,
    hubbard_hard_sector_product_mps,
    max_forbidden_abs,
)
from latticetn.sector_observables import (
    total_particle_number,
    particle_number_variance,
    total_nup,
    total_ndown,
    total_ntot,
    total_sz,
    variance_nup,
    variance_ndown,
    variance_ntot,
)


def test_spinless_hard_product_mps_has_exact_sector_and_zero_forbidden():
    camps = spinless_hard_sector_product_mps(N=6, target_n=3, chi=4, pattern="cdw")
    assert abs(float(total_particle_number(camps.mps)) - 3.0) < 1e-10
    assert float(particle_number_variance(camps.mps)) < 1e-10
    assert max_forbidden_abs(camps.mps, camps.masks) == 0.0


def test_hubbard_hard_product_mps_has_exact_sector_and_zero_forbidden():
    camps = hubbard_hard_sector_product_mps(
        N=6, target_nup=3, target_ndown=3, chi=6, pattern="neel"
    )
    assert abs(float(total_nup(camps.mps)) - 3.0) < 1e-10
    assert abs(float(total_ndown(camps.mps)) - 3.0) < 1e-10
    assert abs(float(total_ntot(camps.mps)) - 6.0) < 1e-10
    assert abs(float(total_sz(camps.mps))) < 1e-10
    assert float(variance_nup(camps.mps)) < 1e-10
    assert float(variance_ndown(camps.mps)) < 1e-10
    assert float(variance_ntot(camps.mps)) < 1e-10
    assert max_forbidden_abs(camps.mps, camps.masks) == 0.0
