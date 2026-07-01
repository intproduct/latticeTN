"""Stage 2 (MPO): MPO -> dense Hamiltonian conversion.

Documents and checks the MPO tensor index order:
    W_i : (left_bond, right_bond, phys_in, phys_out)
and the dense-matrix axis order:
    H[s0..s_{N-1}, s0'..s'_{N-1}] -> reshaped (d**N, d**N),
    with phys_in (rows) = composite ket index, phys_out (cols) = bra index.
"""

from __future__ import annotations

import numpy as np
import torch as tc

from latticetn.mpo import MPO
from latticetn.operators import spin_operators

DTYPE = tc.complex128


def test_identity_mpo_dense_is_identity():
    # Build a trivially identity MPO (D=1) and check to_dense == I_{d^N}.
    for N in [2, 3, 4]:
        I2 = tc.eye(2, dtype=DTYPE)
        tensors = []
        for i in range(N):
            W = I2.reshape(1, 1, 2, 2)  # (l=1, r=1, s_in, s_out) = I
            tensors.append(W)
        mpo = MPO(tensors, length=N, dim=2)
        Hd = mpo.to_dense()
        assert Hd.shape == (2 ** N, 2 ** N)
        assert tc.allclose(Hd, tc.eye(2 ** N, dtype=DTYPE), atol=1e-12)


def test_single_site_mpo_places_operator_on_correct_site():
    # An MPO that places Sz on site k and identity elsewhere should equal
    # I \otimes ... \otimes Sz \otimes ... \otimes I (on site k).
    ops = spin_operators(dtype=DTYPE)
    Sz = ops["Sz"]
    I2 = ops["I"]
    N = 3
    for k in range(N):
        tensors = []
        for i in range(N):
            op = Sz if i == k else I2
            tensors.append(op.reshape(1, 1, 2, 2))
        mpo = MPO(tensors, length=N, dim=2)
        Hd = mpo.to_dense().numpy()
        # reference: kron over sites, placing Sz at position k (site 0 = leftmost)
        ref = np.eye(1, dtype=complex)
        for i in range(N):
            op = (Sz.numpy() if i == k else I2.numpy())
            ref = np.kron(ref, op)
        assert np.allclose(Hd, ref, atol=1e-12), k


def test_mpo_phys_in_phys_out_order_two_site():
    # Verify the (phys_in, phys_out) ordering with a single bond Sz_i Sz_{i+1}.
    # to_dense rows index should be the ket, cols the bra; H|psi> = matmul rows.
    ops = spin_operators(dtype=DTYPE)
    Sz = ops["Sz"]
    I2 = ops["I"]
    N = 2
    # MPO: Sz on both sites (D=1, operator = Sz x Sz).
    tensors = [Sz.reshape(1, 1, 2, 2), Sz.reshape(1, 1, 2, 2)]
    mpo = MPO(tensors, length=N, dim=2)
    Hd = mpo.to_dense()
    # reference Sz x Sz (site 0 most significant)
    ref = tc.kron(Sz, Sz)
    assert tc.allclose(Hd, ref, atol=1e-12)
    # H@psi: rows must contract with psi (ket). Check a known vector.
    psi = tc.tensor([1, 0, 0, 1], dtype=DTYPE) / np.sqrt(2)
    # <Sz Sz> for Bell-like (|00>+|11>) = +1/4 (both aligned)
    expect = float((psi.conj() @ Hd @ psi).real)
    assert abs(expect - 0.25) < 1e-12, expect


def test_mpo_to_dense_hermitian_for_heisenberg():
    from latticetn.operators import heisenberg_dense

    for N in [2, 4, 6]:
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
        H = mpo.to_dense()
        assert tc.allclose(H, H.conj().T, atol=1e-12)
        # matches the dense reference
        assert tc.allclose(H, heisenberg_dense(N, J=1.0), atol=1e-12)
