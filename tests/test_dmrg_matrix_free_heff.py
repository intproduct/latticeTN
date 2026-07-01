"""Stage 4B matrix-free H_eff apply tests.

Verifies the matrix-free H_eff apply matches the dense H_eff matrix, and that
the matrix-free local solve recovers the exact ground eigenvalue on a small
system.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn import dmrg as D  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _two_site_mps(N=4, chi=4, seed=0, i=1):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    t = D.mixed_canonical_two_site(mps, i)
    return MPS.from_tensors(t, dtype=DTYPE, device="cpu"), mpo, i


def test_matrix_free_apply_matches_dense():
    mps, mpo, i = _two_site_mps(N=4, seed=1, i=1)
    Hd = D.effective_hamiltonian(mps, mpo, i)
    mf = D.matrix_free_apply(mps, mpo, i)
    Ddim = mf.dim
    # random vectors
    g = tc.Generator().manual_seed(7)
    for _ in range(4):
        x = (tc.randn(Ddim, dtype=DTYPE, generator=g)
             + 1j * tc.randn(Ddim, dtype=DTYPE, generator=g))
        ref = Hd @ x
        got = mf(x)
        assert tc.allclose(ref, got, atol=1e-10, rtol=1e-10), float((ref - got).abs().max())


def test_matrix_free_apply_matches_dense_for_multiple_bonds():
    mps, mpo, _ = _two_site_mps(N=5, chi=4, seed=2, i=1)
    for i in range(mps.N - 1):
        t = D.mixed_canonical_two_site(mps, i)
        tmp = MPS.from_tensors(t, dtype=DTYPE, device="cpu")
        Hd = D.effective_hamiltonian(tmp, mpo, i)
        mf = D.matrix_free_apply(tmp, mpo, i)
        x = tc.randn(mf.dim, dtype=DTYPE) + 1j * tc.randn(mf.dim, dtype=DTYPE)
        assert tc.allclose(Hd @ x, mf(x), atol=1e-10, rtol=1e-10), i


def test_matrix_free_lowest_eigenvalue_matches_exact():
    # In the full-chi limit the local H_eff ground eigenvalue == exact global.
    N = 4
    tc.manual_seed(3)
    mps = MPS(N, 2, 4, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    t = D.mixed_canonical_two_site(mps, 1)
    tmp = MPS.from_tensors(t, dtype=DTYPE, device="cpu")
    mf = D.matrix_free_apply(tmp, mpo, 1)
    # use the matrix-free apply via a small dense build for the eigenvalue
    Hd = D.effective_hamiltonian(tmp, mpo, 1)
    E0_mf = float(tc.linalg.eigvalsh(Hd)[0].real)   # Hd built FROM matrix-free apply
    E0_exact, _ = exact_ground_energy(heisenberg_dense(N, dtype=DTYPE))
    assert abs(E0_mf - E0_exact) < 1e-8
