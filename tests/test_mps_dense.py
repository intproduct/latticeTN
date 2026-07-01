"""Stage 2: MPS -> dense state conversion.

Verifies the MPS dense expansion, its index order (site 0 = most significant
physical bit), and the bond structure.
"""

from __future__ import annotations

import numpy as np
import torch as tc

from latticetn.mps import MPS

DTYPE = tc.complex128


def test_mps_dense_shape():
    for N in [2, 3, 4, 6]:
        mps = MPS(N, 2, 4, dtype=DTYPE)
        psi = mps.to_dense()
        assert psi.shape == (2 ** N,)
        assert psi.dtype == DTYPE


def test_mps_product_state_expansion():
    # chi=1 -> product state. |b0 b1 b2> with site 0 most significant.
    N = 4
    bits = [1, 0, 1, 1]
    mps = MPS(N, 2, 1, dtype=DTYPE)
    for i in range(N):
        t = tc.zeros((1, 2, 1), dtype=DTYPE)
        t[0, bits[i], 0] = 1.0
        mps.tensors[i] = t.clone().requires_grad_(True)
    psi = mps.to_dense().detach().numpy()
    expected = np.zeros(2 ** N, dtype=complex)
    expected[int("".join(map(str, bits)), 2)] = 1.0
    assert np.allclose(psi, expected), (np.round(psi, 3), np.round(expected, 3))


def test_mps_bond_consistency():
    # bond between site i and i+1 must match; chi large enough is exact.
    N = 6
    mps = MPS(N, 2, 8, dtype=DTYPE)
    for i in range(N - 1):
        assert mps.tensors[i].shape[2] == mps.tensors[i + 1].shape[0]
    # open boundary bonds are 1
    assert mps.tensors[0].shape[0] == 1
    assert mps.tensors[-1].shape[2] == 1


def test_mps_norm_matches_dense():
    tc.manual_seed(0)
    for N in [2, 3, 4, 6]:
        for chi in [2, 4, 8]:
            mps = MPS(N, 2, chi, dtype=DTYPE)
            psi = mps.to_dense()
            n_dense = float((psi.conj() @ psi).real)
            n_mps = float(mps.norm_sq().real)
            assert abs(n_dense - n_mps) / max(1.0, n_dense) < 1e-10, (N, chi, n_dense, n_mps)


def test_mps_overlap_matches_dense():
    tc.manual_seed(2)
    N = 5
    a = MPS(N, 2, 4, dtype=DTYPE)
    b = MPS(N, 2, 4, dtype=DTYPE)
    pa = a.to_dense()
    pb = b.to_dense()
    ov_dense = float((pa.conj() @ pb))
    ov_mps = complex(a.overlap(b))
    assert abs(ov_dense - ov_mps) / max(1.0, abs(ov_dense)) < 1e-10, (ov_dense, ov_mps)
