"""Stage 6: Heisenberg energy via MPO equals dense energy; variational
principle (any E >= E0) holds; the ground-state vector achieves E0.
"""

from __future__ import annotations

import numpy as np
import torch as tc

from latticetn.mpo import MPO
from latticetn.mps import MPS
from latticetn.operators import heisenberg_dense, exact_ground_energy

DTYPE = tc.complex128


def test_heisenberg_energy_mpo_equals_dense_random_mps():
    tc.manual_seed(11)
    for N in [2, 4, 6]:
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
        H = heisenberg_dense(N)
        mps = MPS(N, 2, 8, dtype=DTYPE)
        psi = mps.to_dense()
        e_mpo = float(mps.energy_with_MPO(mpo))
        e_dense = float((psi.conj() @ H @ psi / (psi.conj() @ psi)).real)
        assert abs(e_mpo - e_dense) < 1e-9, (N, e_mpo, e_dense)


def test_variational_principle_random_states_above_ground():
    # For ANY normalized state E >= E0 (E0 is the global minimum). Check random MPS
    # energies are not below E0 (within a tiny tolerance).
    tc.manual_seed(13)
    for N in [2, 4, 6]:
        H = heisenberg_dense(N, J=1.0)
        E0, _ = exact_ground_energy(H)
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
        for _ in range(5):
            mps = MPS(N, 2, 8, dtype=DTYPE)
            e = float(mps.energy_with_MPO(mpo))
            assert e >= E0 - 1e-7, (N, e, E0)  # must not dip below ground state


def test_exact_ground_state_mps_gives_e0():
    # Encode the exact ground state as a dense vector, build a dense "MPS" via
    # large chi, and confirm energy == E0 (within chi representability).
    for N in [2, 4]:
        H = heisenberg_dense(N, J=1.0)
        E0, vec = exact_ground_energy(H)
        # put vec into MPS tensors of bond dim 2^(N//2) (exact-representable)
        chi = 2 ** ((N + 1) // 2)
        mps = MPS(N, 2, chi, dtype=DTYPE)
        # reshape vec [d^N] into (d, d, ..., d) then left-SVD-free: stuff by
        # permuting. Easiest: set tensors so to_dense reproduces vec.
        _set_mps_to_vector(mps, vec)
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
        e = float(mps.energy_with_MPO(mpo))
        assert abs(e - E0) < 1e-8, (N, e, E0)


def _set_mps_to_vector(mps: MPS, vec: tc.Tensor):
    """Stuff a dense state vector into an MPS by successive reshaping/SVD.

    Builds a left-canonical MPS whose to_dense() reproduces `vec`. Operates out
    of the autograd path (sets requires_grad=False) since it is test setup only.
    """
    N = mps.N
    d = mps.dim
    psi = vec.detach().to(tc.complex128).reshape(d, d ** (N - 1))
    tensors = []
    for i in range(N - 1):
        l = 1 if i == 0 else tensors[-1].shape[2]
        mat = psi.reshape(l * d, -1)
        u, s, vh = tc.linalg.svd(mat, full_matrices=False)
        # trim
        chi = u.shape[1]
        A = u.reshape(l, d, chi)
        tensors.append(A)
        psi = (tc.diag(s.to(tc.complex128)) @ vh)
    tensors.append(psi.reshape(psi.shape[0], d, 1))
    for i, A in enumerate(tensors):
        mps.tensors[i] = tc.nn.Parameter(A.clone())
