"""Stage 7A: spinless fermion MPO -> dense must match the dense reference.

``MPO.generate_spinless_fermion`` builds a bond-dim-5 fermionic Hamiltonian MPO
where the adjacent hopping terms use the JW-reduced local product. Its ``to_dense`` must
match ``operators.spinless_fermion_dense`` (which builds the global operators
with the full JW string) across N=2..6 and several (t, V, mu) parameter sets.

This is the fermionic analogue of ``test_heisenberg_mpo_dense``. Nonlocal
one-body observables still require explicit JW strings; adjacent Hamiltonian
hops do not carry a leftover left string.
"""

from __future__ import annotations

import torch as tc

from latticetn.mpo import MPO
from latticetn.operators import spinless_fermion_dense, exact_ground_energy

DTYPE = tc.complex128

CASES = [
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.5, 0.3),
    (0.7, 1.2, -0.4),
    (1.0, -0.5, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 0.5),
    (1.0, 2.0, -0.7),
    (0.5, 0.8, 0.1),
]


def test_mpo_dense_matches_reference():
    for N in [2, 3, 4, 5, 6]:
        for (t, V, mu) in CASES:
            mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_spinless_fermion(
                t=t, V=V, mu=mu)
            H_mpo = mpo.to_dense()
            H_ref = spinless_fermion_dense(N, t=t, V=V, mu=mu, dtype=DTYPE)
            assert tc.allclose(H_mpo, H_ref, atol=1e-12), (
                N, t, V, mu, float((H_mpo - H_ref).abs().max()))


def test_mpo_ground_energy_matches_ed():
    for N in [2, 4, 6]:
        for (t, V, mu) in [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.3)]:
            mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_spinless_fermion(
                t=t, V=V, mu=mu)
            E0_mpo, _ = exact_ground_energy(mpo.to_dense())
            E0_ref, _ = exact_ground_energy(
                spinless_fermion_dense(N, t=t, V=V, mu=mu, dtype=DTYPE))
            assert abs(E0_mpo - E0_ref) < 1e-9, (N, t, V, mu, E0_mpo, E0_ref)


def test_mpo_open_boundary_shapes_and_bond_dim():
    N = 5
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_spinless_fermion(
        t=1.0, V=1.0, mu=0.0)
    assert mpo.tensors[0].shape[0] == 1       # left boundary
    assert mpo.tensors[-1].shape[1] == 1      # right boundary
    # bulk bond dimension is 5 for the JW-reduced nearest-neighbor Hamiltonian MPO
    assert mpo.tensors[1].shape[0] == 5 and mpo.tensors[1].shape[1] == 5


def test_mpo_t_scaling():
    for t in [0.5, 1.0, 2.0]:
        N = 5
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_spinless_fermion(t=t)
        H_ref = spinless_fermion_dense(N, t=t, dtype=DTYPE)
        assert tc.allclose(mpo.to_dense(), H_ref, atol=1e-12)


def test_mpo_is_hermitian():
    for N in [2, 3, 4, 5, 6]:
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_spinless_fermion(
            t=1.0, V=0.5, mu=0.3)
        H = mpo.to_dense()
        assert tc.allclose(H, H.conj().T, atol=1e-12), N
