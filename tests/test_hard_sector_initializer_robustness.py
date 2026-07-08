from latticetn.charge_sectors import (
    hubbard_hard_sector_product_mps,
    max_forbidden_abs,
    spinless_hard_sector_product_mps,
)
from latticetn.sector_observables import hubbard_sector_leakage_report, sector_leakage_report


def test_spinless_hard_sector_preserves_cdw_path_at_minimal_chi():
    camps = spinless_hard_sector_product_mps(4, target_n=2, chi=1, pattern="cdw")
    assert max_forbidden_abs(camps.mps, camps.masks) == 0.0
    assert sector_leakage_report(camps.mps, target_n=2)["abs_error"] < 1e-10


def test_hubbard_hard_sector_preserves_neel_path_at_minimal_chi():
    camps = hubbard_hard_sector_product_mps(4, target_nup=2, target_ndown=2, chi=1, pattern="neel")
    assert max_forbidden_abs(camps.mps, camps.masks) == 0.0
    report = hubbard_sector_leakage_report(camps.mps, target_nup=2, target_ndown=2)
    assert report["n_up_abs_error"] < 1e-10
    assert report["n_down_abs_error"] < 1e-10
