import math

import torch as tc

from latticetn.initial_states import spinless_fixed_number_state, hubbard_fixed_sector_state
from latticetn.sector_observables import (
    particle_number_variance,
    sector_leakage_report,
    variance_nup,
    variance_ndown,
    variance_ntot,
    hubbard_sector_leakage_report,
)


def test_spinless_sector_report_is_finite_and_targets_match():
    mps = spinless_fixed_number_state(5, 2, pattern="centered")
    report = sector_leakage_report(mps, target_n=2)
    assert report["n_target"] == 2
    assert report["abs_error"] < 1e-12
    assert report["variance"] < 1e-12
    assert all(math.isfinite(float(v)) for v in report.values())
    assert tc.isfinite(particle_number_variance(mps)).all()


def test_hubbard_sector_report_is_finite_and_targets_match():
    mps = hubbard_fixed_sector_state(4, n_up=3, n_down=1, pattern="balanced")
    report = hubbard_sector_leakage_report(mps, target_nup=3, target_ndown=1)
    assert report["n_up_target"] == 3
    assert report["n_down_target"] == 1
    assert report["n_up_abs_error"] < 1e-12
    assert report["n_down_abs_error"] < 1e-12
    assert report["n_tot"] == 4.0
    assert report["sz"] == 1.0
    assert all(math.isfinite(float(v)) for v in report.values())
    assert float(variance_nup(mps)) < 1e-12
    assert float(variance_ndown(mps)) < 1e-12
    assert float(variance_ntot(mps)) < 1e-12
