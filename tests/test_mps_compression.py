"""Stage 3A MPS SVD compression tests.

Verifies that SVD compression caps bond dimension at the target chi, preserves
the state when chi is large enough (high fidelity / controlled energy error),
and reports truncation errors. CPU-only, small systems.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mpo import MPO  # noqa: E402
from latticetn.mps import MPS  # noqa: E402
from latticetn import canonical as C  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _fidelity(mps_a: MPS, mps_b: MPS) -> float:
    a = mps_a.to_dense()
    b = mps_b.to_dense()
    a = a / tc.linalg.norm(a)
    b = b / tc.linalg.norm(b)
    return abs(tc.vdot(b, a)).item()


def _bond_dims(mps: MPS) -> list[int]:
    return [int(mps.tensors[i].shape[2]) for i in range(mps.N - 1)]


def test_compression_caps_bond_dimension():
    tc.manual_seed(0)
    mps = MPS(6, 2, 8, dtype=DTYPE)         # large initial bonds
    for chi in (1, 2, 4):
        comp, info = C.svd_compress(mps, chi)
        for d in _bond_dims(comp):
            assert d <= chi
        assert info["max_bond_dim"] <= chi
        # truncation errors are in [0, 1] and non-decreasing as chi shrinks
        for e in info["truncation_errors"]:
            assert -1e-12 <= e <= 1.0 + 1e-12


def test_compression_no_truncation_recovers_state():
    tc.manual_seed(1)
    mps = MPS(6, 2, 4, dtype=DTYPE)
    comp, info = C.svd_compress(mps, chi=16)   # chi >= existing bonds -> exact
    assert _fidelity(mps, comp) > 1.0 - 1e-10
    assert max(info["truncation_errors"]) < 1e-10


def test_compression_heisenberg_energy_error_controlled():
    # Exact Heisenberg ground state -> MPS (full chi) -> compress to full bond.
    N = 6
    H = heisenberg_dense(N, dtype=DTYPE)
    E0, gs = exact_ground_energy(H)
    mps_full = C.from_dense(gs, N, chi=None)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    e_full = float(mps_full.energy_with_MPO(mpo))
    # full-bond compression (chi=8 = 2^(N/2) for N=6) is exact
    comp, info = C.svd_compress(mps_full, chi=8)
    e_comp = float(comp.energy_with_MPO(mpo))
    assert abs(e_full - E0) < 1e-9          # full MPS reproduces exact energy
    assert abs(e_comp - E0) < 1e-9          # compression does not spoil it
    assert info["max_bond_dim"] <= 8
    # variational principle: compressed energy must not undershoot ground state
    assert e_comp >= E0 - 1e-6


def test_compression_truncated_energy_stays_physical():
    # A more aggressive truncation: still must satisfy the variational bound.
    N = 6
    H = heisenberg_dense(N, dtype=DTYPE)
    E0, gs = exact_ground_energy(H)
    mps_full = C.from_dense(gs, N, chi=None)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    comp, info = C.svd_compress(mps_full, chi=2)
    e_comp = float(comp.energy_with_MPO(mpo))
    assert info["max_bond_dim"] <= 2
    # truncated energy >= exact ground (no below-ground violation)
    assert e_comp >= E0 - 1e-6
    # and the energy error is bounded (not divergent)
    assert abs(e_comp - E0) < 1.0
