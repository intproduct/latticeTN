"""Stage 5A AD local-tensor optimization — loss integrity tests.

Verifies the local loss is a differentiable scalar, scale-invariant, and
autograd-clean; only the center tensor is trainable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_local import ADLocalOptimizer  # noqa: E402

DTYPE = tc.complex128


def _make(N=4, chi=8, seed=0, center=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return ADLocalOptimizer(mps, mpo, center=center)


def test_loss_is_real_scalar_with_requires_grad():
    adlo = _make()
    e = adlo.loss()
    assert e.dim() == 0
    assert tc.isfinite(e).all()
    assert e.requires_grad


def test_only_center_tensor_is_trainable():
    adlo = _make(N=5, center=2)
    params = list(adlo.parameters())
    assert len(params) == 1, "exactly the center tensor should be trainable"
    for i, t in enumerate(adlo.mps.tensors):
        if i == adlo.center:
            assert t.requires_grad, f"center site {i} must require grad"
        else:
            assert not t.requires_grad, f"non-center site {i} must be frozen"


def test_loss_is_scale_invariant_in_center():
    # Scaling the center tensor must not change the Rayleigh quotient.
    adlo = _make(N=4, seed=1)
    e_before = float(adlo.loss().real)
    with tc.no_grad():
        adlo.mps.tensors[adlo.center] = tc.nn.Parameter(
            adlo.mps.tensors[adlo.center].detach() * 3.7, requires_grad=True)
    e_after = float(adlo.loss().real)
    assert abs(e_before - e_after) < 1e-9, (e_before, e_after)


def test_move_center_preserves_energy():
    # Center movement is a gauge operation; the Rayleigh quotient is invariant.
    adlo = _make(N=5, seed=2, center=0)
    e0 = float(adlo.loss().real)
    adlo.move_center(4)
    adlo.move_center(1)
    adlo.move_center(3)
    e1 = float(adlo.loss().real)
    assert abs(e0 - e1) < 1e-9, (e0, e1)


def test_loss_does_not_change_under_environment_rescaling():
    # Rescaling a FROZEN environment tensor must not change the Rayleigh quota
    # (scale-invariant), confirming the environment is treated as a constant.
    adlo = _make(N=4, seed=3, center=1)
    e_before = float(adlo.loss().real)
    with tc.no_grad():
        i = (adlo.center + 1) % adlo.N  # a non-center site
        t = adlo.mps.tensors[i]
        adlo.mps.tensors[i] = tc.nn.Parameter(
            t.detach() * 2.1, requires_grad=False)
    e_after = float(adlo.loss().real)
    assert abs(e_before - e_after) < 1e-9, (e_before, e_after)
