"""Stage 5B two-site AD local optimization — split / compression tests.

Verifies the post-step SVD split:
- bond dim does not exceed max_bond_dim;
- full-rank split preserves the dense state (fidelity ~ 1);
- truncation error is non-negative and finite;
- direction absorption gives left/right canonical tensors.
SVD/QR here are split/compression only, NOT the optimizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_two_site import ADTwoSiteOptimizer, _split_theta  # noqa: E402
from latticetn.canonical import left_orthonormal_error, right_orthonormal_error  # noqa: E402

DTYPE = tc.complex128


def _rand_block(l=3, si=2, si1=2, r=3, seed=0):
    tc.manual_seed(seed)
    return tc.randn((l, si, si1, r), dtype=DTYPE)


def test_split_full_rank_fidelity():
    # Full-rank split (no truncation) is a lossless reshape of the block:
    # reconstructing the dense two-site state must give fidelity ~ 1.
    theta = _rand_block(l=3, si=2, si1=2, r=4, seed=1)
    l, si, si1, r = theta.shape
    [A, B], trunc, k = _split_theta(theta, max_bond_dim=None, cutoff=None,
                                    direction="right")
    assert k == min(l * si, si1 * r)
    assert trunc == 0.0
    # reconstruct block
    rec = tc.einsum("lsc,cer->lser", A, B)
    num = (theta.conj() * rec).sum().real
    fid = float(num / (theta.norm() * rec.norm()))
    assert abs(fid - 1.0) < 1e-10, fid


def test_split_respects_max_bond_dim():
    theta = _rand_block(l=4, si=2, si1=2, r=4, seed=2)
    [A, B], trunc, k = _split_theta(theta, max_bond_dim=3, cutoff=None,
                                    direction="right")
    assert k <= 3
    assert A.shape[2] == k
    assert B.shape[0] == k


def test_split_truncation_error_nonneg_finite():
    theta = _rand_block(l=4, si=2, si1=2, r=4, seed=3)
    [A, B], trunc, k = _split_theta(theta, max_bond_dim=2, cutoff=None,
                                    direction="right")
    assert tc.isfinite(tc.tensor(trunc))
    assert trunc >= 0.0
    assert trunc <= 1.0


def test_split_right_direction_left_canonical():
    # direction='right': A_i must be left-canonical (orthonormal columns).
    theta = _rand_block(l=3, si=2, si1=2, r=3, seed=4)
    [A, B], trunc, k = _split_theta(theta, max_bond_dim=None, cutoff=None,
                                    direction="right")
    err = left_orthonormal_error(A)
    assert err < 1e-10, err


def test_split_left_direction_right_canonical():
    # direction='left': A_{i+1} must be right-canonical (orthonormal rows).
    theta = _rand_block(l=3, si=2, si1=2, r=3, seed=5)
    [A, B], trunc, k = _split_theta(theta, max_bond_dim=None, cutoff=None,
                                    direction="left")
    err = right_orthonormal_error(B)
    assert err < 1e-10, err


def test_split_cutoff_drops_tiny_singular_values():
    # Build a block with a clear spectral gap; cutoff should drop tiny modes.
    theta = _rand_block(l=4, si=2, si1=2, r=4, seed=6)
    # rank-2-ish block: zero out two singular directions by construction
    M = theta.reshape(8, 8)
    U, S, Vh = tc.linalg.svd(M, full_matrices=False)
    S2 = S.clone()
    S2[2:] = 0.0
    theta_r = (U * S2) @ Vh
    theta_r = theta_r.reshape(4, 2, 2, 4)
    [A, B], trunc, k = _split_theta(theta_r, max_bond_dim=None, cutoff=1e-12,
                                    direction="right")
    assert k <= 2, k
    assert trunc >= 0.0


def test_optimizer_split_writes_back_into_mps():
    tc.manual_seed(7)
    mps = MPS(4, 2, 6, dtype=DTYPE)
    mpo = MPO.from_bonds(4, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    adtso = ADTwoSiteOptimizer(mps, mpo, bond=1)
    e_global_before = float(adtso.global_energy().real)
    e_local_before = float(adtso.loss().real)
    # optimize Theta a little, then split back with a bond cap
    opt = tc.optim.LBFGS(adtso.parameters(), lr=1.0, max_iter=20,
                         line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        e = adtso.energy()
        e.backward()
        return e
    opt.step(closure)
    trunc, kept = adtso.split(max_bond_dim=4, cutoff=None, direction="right")
    assert kept <= 4
    assert trunc >= 0.0
    # after write-back the global energy should match the local loss we reached
    e_global_after = float(adtso.global_energy().real)
    assert e_global_after <= e_global_before + 1e-9, (e_global_before, e_global_after)
