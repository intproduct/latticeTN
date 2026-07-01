"""Stage 4B Lanczos solver tests.

Verifies the Lanczos lowest-eigenpair against torch.linalg.eigh on small
systems, and that the Lanczos-driven DMRG returns the exact ground energy.
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
from latticetn import lanczos as LZ  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _two_site_mps(N=4, chi=4, seed=0, i=1):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    t = D.mixed_canonical_two_site(mps, i)
    return MPS.from_tensors(t, dtype=DTYPE, device="cpu"), mpo, i


def test_lanczos_lowest_eigenvalue_matches_dense_eigh():
    mps, mpo, i = _two_site_mps(N=4, seed=1, i=1)
    Hd = D.effective_hamiltonian(mps, mpo, i)
    mf = D.matrix_free_apply(mps, mpo, i)
    E_lanczos, _ = LZ.lanczos_lowest_eigenpair(
        mf, mf.dim, dtype=DTYPE, device="cpu", max_iter=30, tol=1e-12, seed=0)
    E_dense = float(tc.linalg.eigvalsh(Hd)[0].real)
    assert abs(float(E_lanczos.real) - E_dense) < 1e-9


def test_lanczos_recovered_vector_is_eigenvector():
    mps, mpo, i = _two_site_mps(N=5, chi=4, seed=2, i=2)
    mf = D.matrix_free_apply(mps, mpo, i)
    E, V = LZ.lanczos_lowest_eigenpair(
        mf, mf.dim, dtype=DTYPE, device="cpu", max_iter=40, tol=1e-12, seed=1)
    # Rayleigh quotient of the returned vector should match E.
    rq = float(LZ.ritz_quotient(mf, V).real)
    assert abs(rq - float(E.real)) < 1e-7


def test_lanczos_dmrg_recovers_exact_n4():
    N = 4
    tc.manual_seed(0)
    mps = MPS(N, 2, 8, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    res = D.run_dmrg(mps, mpo, chi=8, num_sweeps=4, solver="lanczos")
    E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=DTYPE))
    assert abs(res["final_energy"] - E0) < 1e-6
    assert not res["below_ground"]


def test_lanczos_dmrg_recovers_exact_n6():
    N = 6
    tc.manual_seed(0)
    mps = MPS(N, 2, 8, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    res = D.run_dmrg(mps, mpo, chi=8, num_sweeps=4, solver="lanczos")
    E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=DTYPE))
    assert abs(res["final_energy"] - E0) < 1e-6
    assert not res["below_ground"]


def test_lanczos_dmrg_matches_dense_dmrg_small():
    N = 5
    tc.manual_seed(0)
    mps = MPS(N, 2, 8, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    r_dense = D.run_dmrg(mps, mpo, chi=8, num_sweeps=3, solver="dense")
    tc.manual_seed(0)
    mps2 = MPS(N, 2, 8, dtype=DTYPE)
    r_lanczos = D.run_dmrg(mps2, mpo, chi=8, num_sweeps=3, solver="lanczos")
    assert abs(r_dense["final_energy"] - r_lanczos["final_energy"]) < 1e-6
