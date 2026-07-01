"""Stage 2 observable tests.

These tests define the expected observable API. They may fail before Stage 2 is
implemented; Claude Code should make them pass without changing the physics
conventions from Stage 1.
"""

from __future__ import annotations

import torch as tc

from latticetn.mps import MPS
from latticetn.operators import spin_operators, heisenberg_dense, exact_ground_energy


def _normalized_state_from_mps(mps: MPS) -> tc.Tensor:
    psi = mps.to_dense()
    return psi / tc.linalg.norm(psi)


def test_dense_local_and_two_site_observables_on_exact_state():
    from latticetn.observables import dense_expect_local, dense_expect_two_site

    N = 4
    ops = spin_operators()
    H = heisenberg_dense(N)
    _, psi = exact_ground_energy(H)

    # SU(2)-symmetric finite ground state should have near-zero local Sz.
    for site in range(N):
        val = dense_expect_local(psi, ops["Sz"], site, N)
        assert abs(complex(val)) < 1e-10

    # Correlations should be finite real scalars and symmetric under argument order.
    c01 = dense_expect_two_site(psi, ops["Sz"], 0, ops["Sz"], 1, N)
    c10 = dense_expect_two_site(psi, ops["Sz"], 1, ops["Sz"], 0, N)
    assert abs(complex(c01 - c10)) < 1e-10
    assert abs(complex(c01).imag) < 1e-10


def test_mps_local_observable_matches_dense_state_reference():
    from latticetn.observables import dense_expect_local, mps_expect_local

    tc.manual_seed(0)
    N = 4
    mps = MPS(N, 2, chi=4)
    psi = _normalized_state_from_mps(mps)
    ops = spin_operators()

    for site in range(N):
        dense_val = dense_expect_local(psi, ops["Sz"], site, N)
        mps_val = mps_expect_local(mps, ops["Sz"], site)
        assert tc.allclose(tc.as_tensor(mps_val), tc.as_tensor(dense_val), atol=1e-8, rtol=1e-8)


def test_mps_two_site_observable_matches_dense_state_reference():
    from latticetn.observables import dense_expect_two_site, mps_expect_two_site

    tc.manual_seed(1)
    N = 5
    mps = MPS(N, 2, chi=4)
    psi = _normalized_state_from_mps(mps)
    ops = spin_operators()

    pairs = [(0, 1), (1, 3), (2, 4)]
    for i, j in pairs:
        dense_val = dense_expect_two_site(psi, ops["Sz"], i, ops["Sz"], j, N)
        mps_val = mps_expect_two_site(mps, ops["Sz"], i, ops["Sz"], j)
        assert tc.allclose(tc.as_tensor(mps_val), tc.as_tensor(dense_val), atol=1e-8, rtol=1e-8)


def test_mps_bond_energy_matches_dense_state_reference():
    from latticetn.observables import dense_bond_energy_heisenberg, mps_bond_energy_heisenberg

    tc.manual_seed(2)
    N = 4
    mps = MPS(N, 2, chi=4)
    psi = _normalized_state_from_mps(mps)

    for i in range(N - 1):
        dense_val = dense_bond_energy_heisenberg(psi, i, N)
        mps_val = mps_bond_energy_heisenberg(mps, i)
        assert tc.allclose(tc.as_tensor(mps_val), tc.as_tensor(dense_val), atol=1e-8, rtol=1e-8)
