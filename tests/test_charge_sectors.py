import pytest

from latticetn.charge_sectors import (
    build_spinless_bond_sectors,
    build_hubbard_bond_sectors,
)


def test_spinless_bond_sectors_are_reachable_for_n4_target2():
    sectors = build_spinless_bond_sectors(N=4, target_n=2)
    charges = [s.charges for s in sectors]
    assert charges[0] == [0]
    assert charges[-1] == [2]
    assert charges == [[0], [0, 1], [0, 1, 2], [1, 2], [2]]


def test_hubbard_bond_sectors_are_reachable_for_n4_half_filling():
    sectors = build_hubbard_bond_sectors(N=4, target_nup=2, target_ndown=2)
    charges = [s.charges for s in sectors]
    assert charges[0] == [(0, 0)]
    assert charges[-1] == [(2, 2)]
    for i, bond in enumerate(charges):
        for up, down in bond:
            assert 0 <= up <= min(i, 2)
            assert 0 <= down <= min(i, 2)
            assert 0 <= 2 - up <= 4 - i
            assert 0 <= 2 - down <= 4 - i


def test_unreachable_target_rejected():
    with pytest.raises(ValueError, match="target_n"):
        build_spinless_bond_sectors(N=4, target_n=5)
    with pytest.raises(ValueError, match="target_nup"):
        build_hubbard_bond_sectors(N=4, target_nup=5, target_ndown=2)
