"""Stage 2 entanglement entropy tests."""

from __future__ import annotations

import math

import torch as tc

from latticetn.mps import MPS


def test_dense_entanglement_entropy_product_state_is_zero():
    from latticetn.observables import dense_entanglement_entropy

    N = 4
    psi = tc.zeros(2 ** N, dtype=tc.complex128)
    psi[0] = 1.0
    for cut in range(1, N):
        S = dense_entanglement_entropy(psi, cut, N)
        assert abs(float(S)) < 1e-12


def test_dense_entanglement_entropy_bell_pair_is_log2():
    from latticetn.observables import dense_entanglement_entropy

    # N=2, state (|00> + |11>) / sqrt(2), cut between the two sites.
    psi = tc.zeros(4, dtype=tc.complex128)
    psi[0] = 1.0 / math.sqrt(2.0)
    psi[3] = 1.0 / math.sqrt(2.0)
    S = dense_entanglement_entropy(psi, cut=1, N=2)
    assert abs(float(S) - math.log(2.0)) < 1e-12


def test_mps_entanglement_entropy_matches_dense_reference():
    from latticetn.observables import dense_entanglement_entropy, mps_entanglement_entropy

    tc.manual_seed(3)
    N = 5
    mps = MPS(N, 2, chi=4)
    psi = mps.to_dense()
    psi = psi / tc.linalg.norm(psi)

    for cut in range(1, N):
        dense_S = dense_entanglement_entropy(psi, cut, N)
        mps_S = mps_entanglement_entropy(mps, cut)
        assert abs(float(dense_S) - float(mps_S)) < 1e-8
