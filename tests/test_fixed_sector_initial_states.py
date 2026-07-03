import pytest

from latticetn.initial_states import (
    spinless_fixed_number_state,
    spinless_half_filled_cdw_state,
    hubbard_fixed_sector_state,
    hubbard_half_filled_neel_state,
)
from latticetn.sector_observables import (
    total_particle_number,
    total_nup,
    total_ndown,
    total_ntot,
    total_sz,
)


def test_spinless_fixed_number_product_state_has_requested_total():
    for pattern in ("left", "cdw", "centered"):
        mps = spinless_fixed_number_state(6, 3, pattern=pattern)
        assert abs(float(total_particle_number(mps)) - 3.0) < 1e-12
        assert max(t.shape[0] for t in mps.tensors) == 1
        assert max(t.shape[2] for t in mps.tensors) == 1


def test_spinless_half_filled_cdw_rejects_odd_length():
    with pytest.raises(ValueError, match="even N"):
        spinless_half_filled_cdw_state(5)


def test_hubbard_fixed_sector_product_state_has_requested_charges():
    mps = hubbard_fixed_sector_state(6, n_up=4, n_down=3, pattern="balanced")
    assert abs(float(total_nup(mps)) - 4.0) < 1e-12
    assert abs(float(total_ndown(mps)) - 3.0) < 1e-12
    assert abs(float(total_ntot(mps)) - 7.0) < 1e-12
    assert abs(float(total_sz(mps)) - 0.5) < 1e-12


def test_hubbard_half_filled_neel_sector():
    mps = hubbard_half_filled_neel_state(4)
    assert abs(float(total_nup(mps)) - 2.0) < 1e-12
    assert abs(float(total_ndown(mps)) - 2.0) < 1e-12


def test_hubbard_rejects_impossible_sector():
    with pytest.raises(ValueError, match="0 <= n <= N"):
        hubbard_fixed_sector_state(4, n_up=5, n_down=0)
