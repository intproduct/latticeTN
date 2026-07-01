"""Stage 5A AD gauge projection tests.

Verifies the canonical gauge projection:
- preserves the Rayleigh energy (gauge, not physics),
- preserves the dense state up to a global phase (fidelity ~1),
- reduces the canonical (left-orthonormality) error,
- keeps the MPS tensors as trainable leaf parameters afterward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_variational import (  # noqa: E402
    ADVariationalMPS, _canonical_error, _project,
)
from latticetn.operators import heisenberg_dense  # noqa: E402

DTYPE = tc.complex128


def _make(N=5, chi=4, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return ADVariationalMPS(mps, mpo)


def _fidelity(ad_a, ad_b) -> float:
    a = ad_a.mps.to_dense().detach()
    b = ad_b.mps.to_dense().detach()
    a = a / tc.linalg.norm(a)
    b = b / tc.linalg.norm(b)
    return abs(tc.vdot(b, a)).item()


def test_canonical_projection_preserves_rayleigh_energy():
    ad = _make(seed=1)
    e_before = float(ad.energy())
    _project(ad.mps, "canonical")
    e_after = float(ad.energy())
    assert abs(e_before - e_after) < 1e-9


def test_canonical_projection_fidelity_near_one():
    ad = _make(seed=2)
    # snapshot state before projection
    snap = ad.mps.clone()
    snap_ad = ADVariationalMPS(snap, ad.mpo)
    _project(ad.mps, "canonical")
    fid = _fidelity(snap_ad, ad)
    assert fid > 1.0 - 1e-8


def test_canonical_projection_reduces_canonical_error():
    ad = _make(seed=3)
    err_before = _canonical_error(ad.mps)
    _project(ad.mps, "canonical")
    err_after = _canonical_error(ad.mps)
    assert err_after < err_before
    assert err_after < 1e-10          # essentially left-canonical


def test_projection_keeps_params_trainable_leaves():
    ad = _make(seed=4)
    for proj in ("none", "tensor_norm", "canonical"):
        _project(ad.mps, proj)
        for p in ad.parameters():
            assert p.requires_grad, proj
            assert p.is_leaf, proj


def test_projection_none_is_identity():
    ad = _make(seed=5)
    snap = [p.detach().clone() for p in ad.parameters()]
    _project(ad.mps, "none")
    for s, p in zip(snap, ad.parameters()):
        assert tc.allclose(s, p.detach())


def test_projection_invalid_raises():
    ad = _make()
    try:
        _project(ad.mps, "bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid projection")
