"""Stage 7C: spinful Hubbard MPO -> dense must match the dense reference.

``MPO.generate_hubbard`` builds a bond-dim-6 fermionic Hubbard MPO (local
d=4) whose ``to_dense`` must match ``operators.hubbard_dense`` across N=2..4
and several (t, U, mu, h) parameter sets. The MPO is NOT a spin /
hard-core-boson MPO: the surviving site-``i`` parity ``@ P`` / ``P @`` on the
hop left factors (and the intra-site ``F_up`` inside ``cdown``/``cdagdown``)
are what make it fermionic.
"""

from __future__ import annotations

import torch as tc

from latticetn.mpo import MPO
from latticetn.operators import hubbard_dense, exact_ground_energy

DTYPE = tc.complex128

CASES = [
    (1.0, 4.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.7, 2.0, 0.3, 0.1),
    (1.0, 4.0, 0.2, -0.3),
    (1.0, 8.0, 0.0, 0.0),
    (0.5, 1.0, -0.2, 0.4),
    (0.0, 4.0, 0.0, 0.0),
    (2.0, 0.5, 0.1, 0.0),
    (1.0, 4.0, 0.5, 0.0),
    (1.5, 3.0, -0.1, 0.2),
]


def test_mpo_dense_matches_reference():
    for N in [2, 3, 4]:
        for (t, U, mu, h) in CASES:
            mpo = MPO.from_bonds(N, 4, dtype=DTYPE).generate_hubbard(
                t=t, U=U, mu=mu, h=h)
            H_mpo = mpo.to_dense()
            H_ref = hubbard_dense(N, t=t, U=U, mu=mu, h=h, dtype=DTYPE)
            assert tc.allclose(H_mpo, H_ref, atol=1e-11), (
                N, t, U, mu, h, float((H_mpo - H_ref).abs().max()))


def test_mpo_ground_energy_matches_ed():
    for N in [2, 4]:
        for (t, U, mu, h) in [(1.0, 4.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0),
                              (0.5, 2.0, -0.3, 0.2)]:
            mpo = MPO.from_bonds(N, 4, dtype=DTYPE).generate_hubbard(
                t=t, U=U, mu=mu, h=h)
            E0_mpo, _ = exact_ground_energy(mpo.to_dense())
            E0_ref, _ = exact_ground_energy(
                hubbard_dense(N, t=t, U=U, mu=mu, h=h, dtype=DTYPE))
            assert abs(E0_mpo - E0_ref) < 1e-9, (N, t, U, mu, h, E0_mpo, E0_ref)


def test_mpo_open_boundary_shapes_and_bond_dim():
    N = 5
    mpo = MPO.from_bonds(N, 4, dtype=DTYPE).generate_hubbard(
        t=1.0, U=4.0, mu=0.0, h=0.0)
    assert mpo.tensors[0].shape[0] == 1            # left boundary
    assert mpo.tensors[-1].shape[1] == 1           # right boundary
    # bulk bond dimension is 6 for the Hubbard MPO
    assert mpo.tensors[1].shape[0] == 6 and mpo.tensors[1].shape[1] == 6
    # local physical dimension is 4
    assert mpo.tensors[1].shape[2] == 4 and mpo.tensors[1].shape[3] == 4


def test_mpo_t_scaling():
    for t in [0.5, 1.0, 2.0]:
        N = 4
        mpo = MPO.from_bonds(N, 4, dtype=DTYPE).generate_hubbard(t=t)
        H_ref = hubbard_dense(N, t=t, dtype=DTYPE)
        assert tc.allclose(mpo.to_dense(), H_ref, atol=1e-11)


def test_mpo_u_scaling():
    for U in [0.0, 2.0, 4.0, 8.0]:
        N = 4
        mpo = MPO.from_bonds(N, 4, dtype=DTYPE).generate_hubbard(U=U)
        H_ref = hubbard_dense(N, U=U, dtype=DTYPE)
        assert tc.allclose(mpo.to_dense(), H_ref, atol=1e-11)


def test_mpo_is_hermitian():
    # N up to 4 (d=4 MPO.to_dense is expensive at N=5,6).
    for N in [2, 3, 4]:
        mpo = MPO.from_bonds(N, 4, dtype=DTYPE).generate_hubbard(
            t=1.0, U=4.0, mu=0.3, h=-0.2)
        H = mpo.to_dense()
        assert tc.allclose(H, H.conj().T, atol=1e-11), N
