"""Stage 3B native norm contraction tests.

Verifies the native (no-to_dense) MPS norm contraction against the dense norm
and against the Stage 1 overlap path, for small random MPS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn import contractions as K  # noqa: E402

DTYPE = tc.complex128


def _dense_norm_sq(mps: MPS) -> float:
    psi = mps.to_dense().detach()
    return float((psi.conj() @ psi).real)


def test_native_norm_matches_dense_for_random_mps():
    tc.manual_seed(0)
    mps = MPS(6, 2, 4, dtype=DTYPE)
    diff = abs(_dense_norm_sq(mps) - float(K.native_norm_sq(mps).real))
    assert diff < 1e-9


def test_native_norm_matches_stage1_overlap_path():
    tc.manual_seed(1)
    mps = MPS(5, 2, 4, dtype=DTYPE)
    overlap_norm_sq = float(mps.overlap(mps).real)
    native = float(K.native_norm_sq(mps).real)
    assert abs(overlap_norm_sq - native) < 1e-10


def test_native_norm_does_not_change_mps_tensors():
    tc.manual_seed(2)
    mps = MPS(4, 2, 4, dtype=DTYPE)
    before = [t.detach().clone() for t in mps.tensors]
    _ = K.native_norm_sq(mps)
    for b, a in zip(before, mps.tensors):
        assert tc.allclose(b, a.detach())


def test_native_norm_gradient_flows():
    tc.manual_seed(3)
    mps = MPS(4, 2, 4, dtype=DTYPE)
    n = K.native_norm(mps)        # sqrt(<psi|psi>), differentiable
    n.backward()
    assert all(p.grad is not None for p in mps.tensors)
