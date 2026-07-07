from __future__ import annotations

import math

import torch as tc

from latticetn.fermion_operators import fermion_operators, hubbard_local_operators
from latticetn.initial_states import neel_spin_state
from latticetn.observables import (
    dense_connected_correlation,
    dense_entanglement_entropy,
    dense_expect_local,
    dense_expect_two_site,
    dense_fermion_density_density,
    dense_fermion_local_density,
    dense_fermion_nn_hopping,
    dense_hubbard_double_occ,
    dense_hubbard_local_density,
    dense_hubbard_local_sz,
    mps_connected_correlation,
)
from latticetn.operators import spin_operators


DTYPE = tc.complex128


def _basis_state(dim: int, states: list[int]) -> tc.Tensor:
    idx = 0
    for state in states:
        idx = idx * dim + state
    psi = tc.zeros(dim ** len(states), dtype=DTYPE)
    psi[idx] = 1.0
    return psi


def test_spin_product_observables_and_connected_correlations_are_exact():
    N = 4
    psi = _basis_state(2, [0, 1, 0, 1])  # up, down, up, down
    ops = spin_operators(dtype=DTYPE)
    expected_sz = [0.5, -0.5, 0.5, -0.5]

    for site, expected in enumerate(expected_sz):
        assert abs(float(dense_expect_local(psi, ops["Sz"], site, N).real) - expected) < 1e-12

    assert abs(float(dense_expect_two_site(psi, ops["Sz"], 0, ops["Sz"], 1, N).real) + 0.25) < 1e-12
    assert abs(complex(dense_connected_correlation(psi, ops["Sz"], 0, ops["Sz"], 1, N))) < 1e-12

    mps = neel_spin_state(N, dtype=DTYPE)
    assert abs(complex(mps_connected_correlation(mps, ops["Sz"], 0, ops["Sz"], 1))) < 1e-12


def test_spinless_product_density_and_connected_correlations_are_exact():
    N = 4
    psi = _basis_state(2, [1, 0, 1, 0])
    expected_n = [1.0, 0.0, 1.0, 0.0]
    for site, expected in enumerate(expected_n):
        assert abs(float(dense_fermion_local_density(psi, site, N)) - expected) < 1e-12
    assert abs(float(dense_fermion_density_density(psi, 0, 2, N)) - 1.0) < 1e-12

    ops = fermion_operators(dtype=DTYPE)
    connected = dense_connected_correlation(psi, ops["n"], 0, ops["n"], 2, N)
    assert abs(complex(connected)) < 1e-12


def test_spinless_nn_hopping_has_no_left_parity_string():
    N = 3
    psi = (_basis_state(2, [1, 1, 0]) + _basis_state(2, [1, 0, 1])) / math.sqrt(2.0)
    # The occupied site 0 is left of the hopping bond (1, 2). The correct
    # adjacent JW-reduced Hamiltonian observable is +1, not -1.
    assert abs(float(dense_fermion_nn_hopping(psi, 1, N)) - 1.0) < 1e-12


def test_hubbard_product_observables_and_connected_correlations_are_exact():
    N = 4
    psi = _basis_state(4, [1, 2, 3, 0])  # up, down, updown, empty
    expected = [
        (1.0, 0.0, 0.5, 0.0),
        (0.0, 1.0, -0.5, 0.0),
        (1.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 0.0),
    ]
    for site, (nup, ndown, sz, docc) in enumerate(expected):
        assert abs(float(dense_hubbard_local_density(psi, site, N, "up")) - nup) < 1e-12
        assert abs(float(dense_hubbard_local_density(psi, site, N, "down")) - ndown) < 1e-12
        assert abs(float(dense_hubbard_local_sz(psi, site, N)) - sz) < 1e-12
        assert abs(float(dense_hubbard_double_occ(psi, site, N)) - docc) < 1e-12

    ops = hubbard_local_operators(dtype=DTYPE)
    connected = dense_connected_correlation(psi, ops["nup"], 0, ops["ndown"], 2, N)
    assert abs(complex(connected)) < 1e-12


def test_entanglement_entropy_known_states():
    product = _basis_state(2, [0, 0, 0, 0])
    for cut in range(1, 4):
        assert abs(float(dense_entanglement_entropy(product, cut, 4))) < 1e-12

    bell = (tc.tensor([1, 0, 0, 1], dtype=DTYPE)) / math.sqrt(2.0)
    assert abs(float(dense_entanglement_entropy(bell, 1, 2)) - math.log(2.0)) < 1e-12
