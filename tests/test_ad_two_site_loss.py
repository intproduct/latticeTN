"""Stage 5B two-site AD local optimization — loss integrity tests.

Verifies the two-site local loss is a differentiable scalar, finite,
requires_grad, scale-invariant in Theta, and that only the two-site center
tensor Theta is trainable (the rest of the chain is frozen).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_two_site import ADTwoSiteOptimizer  # noqa: E402

DTYPE = tc.complex128


def _make(N=4, chi=8, seed=0, bond=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return ADTwoSiteOptimizer(mps, mpo, bond=bond)


def test_loss_is_real_scalar_with_requires_grad():
    adtso = _make()
    e = adtso.loss()
    assert e.dim() == 0
    assert tc.isfinite(e).all()
    assert e.requires_grad


def test_only_two_site_theta_is_trainable():
    adtso = _make(N=5, bond=2)
    params = list(adtso.parameters())
    assert len(params) == 1, "exactly the two-site center tensor should be trainable"
    theta = params[0]
    assert theta.requires_grad
    # all MPS site tensors are frozen
    for t in adtso.mps.tensors:
        assert not t.requires_grad


def test_theta_shape_is_two_site_block():
    # Theta must be (l, s_i, s_{i+1}, r): four axes.
    adtso = _make(N=5, bond=2)
    assert adtso.theta.dim() == 4


def test_loss_is_scale_invariant_in_theta():
    # Scaling Theta must not change the Rayleigh quotient.
    adtso = _make(N=4, seed=1, bond=1)
    e_before = float(adtso.loss().real)
    with tc.no_grad():
        adtso.theta = tc.nn.Parameter(adtso.theta.detach() * 3.7,
                                      requires_grad=True)
    e_after = float(adtso.loss().real)
    assert abs(e_before - e_after) < 1e-9, (e_before, e_after)


def test_loss_equals_global_rayleigh_in_mixed_canonical():
    # In exact two-site mixed-canonical form the local loss E(Theta)
    # = <Theta|H_eff|Theta>/<Theta|Theta> equals the global Rayleigh quotient.
    adtso = _make(N=4, seed=2, bond=1)
    e_local = float(adtso.loss().real)
    e_global = float(adtso.global_energy().real)
    assert abs(e_local - e_global) < 1e-7, (e_local, e_global)


def test_reset_bond_preserves_global_energy():
    # Re-canonicalizing at a different bond is a gauge operation; the global
    # Rayleigh quotient is invariant.
    adtso = _make(N=5, seed=3, bond=0)
    e0 = float(adtso.global_energy().real)
    adtso.reset_bond(3)
    e1 = float(adtso.global_energy().real)
    adtso.reset_bond(1)
    e2 = float(adtso.global_energy().real)
    assert abs(e0 - e1) < 1e-7, (e0, e1)
    assert abs(e0 - e2) < 1e-7, (e0, e2)
