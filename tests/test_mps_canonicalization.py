"""Stage 3A MPS canonicalization tests.

Verifies left/right/mixed canonical forms on small random MPS: orthonormality
structure, dense-state fidelity up to a global phase, and norm consistency.
CPU-only, complex128, small systems.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn import canonical as C  # noqa: E402


def _fidelity(mps_a: MPS, mps_b: MPS) -> float:
    a = mps_a.to_dense()
    b = mps_b.to_dense()
    a = a / tc.linalg.norm(a)
    b = b / tc.linalg.norm(b)
    return abs(tc.vdot(b, a)).item()


def _make(N=5, chi=4, seed=0):
    tc.manual_seed(seed)
    return MPS(N, 2, chi, dtype=tc.complex128)


def test_left_canonical_preserves_state_and_orthonormality():
    mps = _make()
    L = C.left_canonical(mps)
    # sites 0..N-2 left-orthonormal
    assert C.left_orthonormal_all(L) < 1e-12
    # dense state preserved up to global phase
    assert _fidelity(mps, L) > 1.0 - 1e-10


def test_right_canonical_preserves_state_and_orthonormality():
    mps = _make(seed=1)
    R = C.right_canonical(mps)
    # sites 1..N-1 right-orthonormal
    assert C.right_orthonormal_all(R) < 1e-12
    assert _fidelity(mps, R) > 1.0 - 1e-10


def test_mixed_canonical_structure_for_each_center():
    mps = _make(N=6, chi=4, seed=2)
    for center in range(1, mps.N - 1):
        M = C.mixed_canonical(mps, center)
        # left of center: left-orthonormal
        assert C.left_orthonormal_all(M, upto=center) < 1e-12
        # right of center: right-orthonormal
        assert C.right_orthonormal_all(M, from_=center) < 1e-12
        # state preserved
        assert _fidelity(mps, M) > 1.0 - 1e-10


def test_canonical_norm_matches_dense_norm():
    mps = _make(seed=3)
    dense_norm = float(tc.linalg.norm(mps.to_dense()))
    for fn in (C.left_canonical, C.right_canonical):
        canon = fn(mps)
        assert abs(C.canonical_norm(canon) - dense_norm) < 1e-9
    for center in range(1, mps.N - 1):
        M = C.mixed_canonical(mps, center)
        assert abs(C.canonical_norm(M) - dense_norm) < 1e-9
        # independent cross-check: whole norm sits in the center tensor
        assert abs(C.center_frob_norm(M, center) - dense_norm) < 1e-9
