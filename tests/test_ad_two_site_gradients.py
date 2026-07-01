"""Stage 5B two-site AD local optimization — gradient tests.

Verifies backward() populates a non-None, finite gradient on Theta and that a
single local optimizer step does not increase the energy (LBFGS line search).
Also compares the autograd gradient against a finite-difference estimate.
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


def _make(N=4, chi=8, seed=0, bond=1):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return ADTwoSiteOptimizer(mps, mpo, bond=bond)


def test_backward_populates_finite_theta_grad():
    adtso = _make()
    for p in adtso.parameters():
        if p.grad is not None:
            p.grad = None
    e = adtso.loss()
    e.backward()
    g = adtso.theta.grad
    assert g is not None, "Theta grad is None after backward"
    assert tc.isfinite(g).all(), "Theta grad not finite"
    assert g.abs().sum() > 0, "Theta grad is exactly zero"


def test_one_lbfgs_step_does_not_increase_energy():
    adtso = _make(N=4, seed=1, bond=1)
    e_before = float(adtso.loss().real)
    opt = tc.optim.LBFGS(adtso.parameters(), lr=1.0, max_iter=20,
                         line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        e = adtso.energy()
        e.backward()
        return e
    opt.step(closure)
    e_after = float(adtso.loss().real)
    # LBFGS line search on a Rayleigh quotient should not raise the energy.
    assert e_after <= e_before + 1e-9, (e_before, e_after)


def test_one_adam_step_does_not_strongly_increase_energy():
    adtso = _make(N=4, seed=2, bond=1)
    e_before = float(adtso.loss().real)
    opt = tc.optim.Adam(adtso.parameters(), lr=5e-2)
    for _ in range(5):
        opt.zero_grad()
        e = adtso.energy()
        e.backward()
        opt.step()
    e_after = float(adtso.loss().real)
    # Allow a tiny transient but require no strong increase.
    assert e_after <= e_before + 1e-6, (e_before, e_after)


def test_gradient_matches_finite_difference():
    # Real-part finite-difference check on a realified view of Theta.
    adtso = _make(N=3, chi=4, seed=4, bond=0)
    for p in adtso.parameters():
        if p.grad is not None:
            p.grad = None
    e = adtso.loss()
    e.backward()
    ana = adtso.theta.grad.detach().clone()

    eps = 1e-6
    th = adtso.theta.detach().clone()
    num = tc.zeros_like(th)
    flat = th.reshape(-1)
    # perturb the real part of a few entries
    idxs = [0, flat.numel() // 2, flat.numel() - 1]
    for k in idxs:
        for sgn in (+1, -1):
            th2 = th.clone().reshape(-1)
            th2[k] = th2[k] + eps * sgn
            th2 = th2.reshape(th.shape)
            with tc.no_grad():
                adtso.theta = tc.nn.Parameter(th2, requires_grad=True)
                ep = float(adtso.loss().real)
            if sgn == +1:
                ep_plus = ep
            else:
                ep_minus = ep
        fd = (ep_plus - ep_minus) / (2 * eps)
        flat_ana = ana.reshape(-1)
        # dE/d(real theta_k) = 2 * Re(grad_k) for the standard complex convention;
        # check sign-consistent magnitude order.
        flat_num = num.reshape(-1)
        flat_num[k] = fd
        num = flat_num.reshape(th.shape)
    # restore theta
    with tc.no_grad():
        adtso.theta = tc.nn.Parameter(th, requires_grad=True)

    for k in idxs:
        # PyTorch complex autograd convention: dE/d(Re theta_k) = Re(grad_k).
        a = float(ana.reshape(-1)[k].real)
        n = float(num.reshape(-1)[k])
        # same sign and within a loose relative tolerance (FD on complex loss)
        rel = abs(a - n) / (abs(a) + abs(n) + 1e-12)
        assert rel < 5e-2 or abs(a - n) < 1e-5, (k, a, n, rel)
