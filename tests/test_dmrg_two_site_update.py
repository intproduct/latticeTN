"""Stage 4A two-site update (SVD split + truncation) tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn import dmrg as D  # noqa: E402
from latticetn import canonical as C  # noqa: E402

DTYPE = tc.complex128


def _random_theta(l=3, s=2, s1=2, r=3, seed=0):
    g = tc.Generator().manual_seed(seed)
    return tc.randn(l, s, s1, r, dtype=DTYPE, generator=g)


def test_two_site_update_bond_dim_capped_at_chi():
    Theta = _random_theta(l=4, r=4, seed=0)
    for chi in (1, 2, 3, 4, 8):
        [A, B], trunc, k = D.two_site_update(Theta, chi=chi, direction="right")
        assert A.shape[2] == B.shape[0] == k
        assert k <= chi


def test_two_site_right_update_left_canonical_site():
    # direction='right': A must be left-canonical (A^H A = I on the new bond).
    Theta = _random_theta(l=4, r=4, seed=1)
    [A, B], trunc, k = D.two_site_update(Theta, chi=4, direction="right")
    l, s, r = A.shape
    M = A.reshape(l * s, r)
    eye = tc.eye(r, dtype=DTYPE)
    assert float((M.conj().t() @ M - eye).abs().max()) < 1e-10


def test_two_site_left_update_right_canonical_site():
    # direction='left': B must be right-canonical (B B^H = I on the new bond).
    Theta = _random_theta(l=4, r=4, seed=2)
    [A, B], trunc, k = D.two_site_update(Theta, chi=4, direction="left")
    l, s, r = B.shape
    M = B.reshape(l, s * r)
    eye = tc.eye(l, dtype=DTYPE)
    assert float((M @ M.conj().t() - eye).abs().max()) < 1e-10


def test_truncation_error_nonneg_finite_and_zero_at_full_chi():
    Theta = _random_theta(l=4, r=4, seed=3)
    for chi in (1, 2, 4):
        _, trunc, k = D.two_site_update(Theta, chi=chi, direction="right")
        assert 0.0 <= trunc <= 1.0 + 1e-12
        assert trunc == trunc  # finite, not NaN
    # full chi (>= min(l*s, s1*r)) -> zero truncation
    _, trunc, _ = D.two_site_update(Theta, chi=16, direction="right")
    assert trunc < 1e-12


def test_two_site_update_preserves_state_at_full_chi():
    # Without truncation the split reconstructs Theta exactly (up to SVD signs).
    Theta = _random_theta(l=3, r=3, seed=4)
    [A, B], trunc, _ = D.two_site_update(Theta, chi=16, direction="right")
    rec = tc.einsum("lsc,cer->lser", A, B)
    assert tc.allclose(rec, Theta, atol=1e-10)
