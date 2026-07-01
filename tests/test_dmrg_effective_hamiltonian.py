"""Stage 4A effective-Hamiltonian tests.

Verifies H_eff is Hermitian and that its lowest eigenvalue equals the exact
ground energy on the effective two-site space of a two-site mixed-canonical MPS
(the local-vs-full alignment: since everything outside the block is canonical,
the local eigenvalue IS the global variational minimum for that block).
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
from latticetn import contractions as K  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _two_site_mps(N=4, chi=4, seed=0, i=1):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    t = D.mixed_canonical_two_site(mps, i)
    return MPS.from_tensors(t, dtype=DTYPE, device="cpu"), mpo, i


def test_heff_is_hermitian():
    mps, mpo, i = _two_site_mps(N=4, i=1)
    H = D.effective_hamiltonian(mps, mpo, i)
    herm_err = float((H - H.conj().t()).abs().max())
    assert herm_err < 1e-10


def test_heff_hermitian_for_multiple_bonds():
    mps, mpo, _ = _two_site_mps(N=5, chi=4, i=1)
    for i in range(mps.N - 1):
        t = D.mixed_canonical_two_site(mps, i)
        tmp = MPS.from_tensors(t, dtype=DTYPE, device="cpu")
        H = D.effective_hamiltonian(tmp, mpo, i)
        assert float((H - H.conj().t()).abs().max()) < 1e-10


def test_heff_lowest_eigenvalue_geq_exact_ground():
    # On a random MPS the local two-site optimum is a variational upper bound
    # on the global ground energy (>= exact E0) — it must NOT undershoot.
    mps, mpo, i = _two_site_mps(N=4, chi=4, seed=2, i=1)
    H = D.effective_hamiltonian(mps, mpo, i)
    ev = tc.linalg.eigvalsh(H)
    E0_exact, _ = exact_ground_energy(heisenberg_dense(mps.N, dtype=DTYPE))
    assert float(ev[0].real) >= E0_exact - 1e-8


def test_heff_local_energy_recovers_exact_in_full_chi_limit():
    # When the block spans the whole effective space (small N, full chi), the
    # local two-site ground energy equals the exact global ground energy.
    N = 4
    tc.manual_seed(3)
    mps = MPS(N, 2, 4, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    # bring to two-site form at bond 1
    t = D.mixed_canonical_two_site(mps, 1)
    tmp = MPS.from_tensors(t, dtype=DTYPE, device="cpu")
    H = D.effective_hamiltonian(tmp, mpo, 1)
    ev = float(tc.linalg.eigvalsh(H)[0].real)
    E0_exact, _ = exact_ground_energy(heisenberg_dense(N, dtype=DTYPE))
    assert abs(ev - E0_exact) < 1e-8


def test_heff_trace_real_and_finite():
    mps, mpo, i = _two_site_mps(N=4, i=0)
    H = D.effective_hamiltonian(mps, mpo, i)
    assert tc.isfinite(H).all()
    assert abs(float(H.trace().imag)) < 1e-9
