import torch as tc

from latticetn.charges import (
    spinless_charge_metadata,
    hubbard_charge_metadata,
    local_number_operator,
    local_nup_operator,
    local_ndown_operator,
    local_ntot_operator,
    local_sz_operator,
)


def test_spinless_metadata_and_number_operator():
    assert spinless_charge_metadata() == {"n": [0, 1], "parity": [0, 1]}
    n = local_number_operator("spinless")
    assert tc.allclose(n, tc.diag(tc.tensor([0, 1], dtype=tc.complex128)))


def test_hubbard_metadata_and_diagonal_operators():
    meta = hubbard_charge_metadata()
    assert meta["n_up"] == [0, 1, 0, 1]
    assert meta["n_down"] == [0, 0, 1, 1]
    assert meta["n_tot"] == [0, 1, 1, 2]
    assert meta["sz2"] == [0, 1, -1, 0]
    assert meta["parity"] == [0, 1, 1, 0]
    assert tc.allclose(local_nup_operator(), tc.diag(tc.tensor(meta["n_up"], dtype=tc.complex128)))
    assert tc.allclose(local_ndown_operator(), tc.diag(tc.tensor(meta["n_down"], dtype=tc.complex128)))
    assert tc.allclose(local_ntot_operator(), tc.diag(tc.tensor(meta["n_tot"], dtype=tc.complex128)))
    assert tc.allclose(local_sz_operator(), tc.diag(tc.tensor([0, 0.5, -0.5, 0], dtype=tc.complex128)))
