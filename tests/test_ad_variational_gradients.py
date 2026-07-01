"""Stage 4R AD variational gradient tests.

Verifies loss.backward() populates all trainable MPS parameter gradients
(non-None, finite) — the core autograd requirement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS  # noqa: E402

DTYPE = tc.complex128


def _make(N=4, chi=8, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return ADVariationalMPS(mps, mpo)


def test_backward_populates_all_param_grads():
    ad = _make(N=5, chi=4, seed=1)
    e = ad.loss()
    e.backward()
    for i, p in enumerate(ad.parameters()):
        assert p.grad is not None, f"site {i} grad is None"
        assert tc.isfinite(p.grad).all(), f"site {i} grad not finite"
        # gradient should not be identically zero for a random init
        assert float(p.grad.abs().max()) > 0.0, f"site {i} grad all zero"


def test_params_are_leaf_and_require_grad():
    ad = _make()
    for p in ad.parameters():
        assert p.requires_grad
        assert p.is_leaf


def test_grad_matches_autograd_via_torch_gradcheck_finite_diff():
    # Lightweight finite-difference check on a single parameter element: the
    # analytic autograd grad should be consistent with a numerical directional
    # derivative of the real loss.
    ad = _make(N=4, chi=4, seed=2)
    e0 = ad.loss()
    e0.backward()
    p = ad.parameters()[0]
    g_analytic = p.grad.detach().clone()
    assert g_analytic is not None
    # pick a real direction (loss is real)
    eps = 1e-6
    with tc.no_grad():
        p.add_(eps)
    e_plus = float(ad.loss())
    with tc.no_grad():
        p.sub_(2 * eps)
    e_minus = float(ad.loss())
    with tc.no_grad():
        p.add_(eps)  # restore
    # directional derivative estimate (direction = unit along data add we did,
    # which added eps uniformly -> direction is all-ones scaled). Since we
    # added eps to every element, the finite-diff directional derivative along
    # the all-ones direction = (e_plus - e_minus)/(2*eps); analytic along same
    # direction = sum(g_analytic.real).
    fd = (e_plus - e_minus) / (2 * eps)
    ana = float(g_analytic.real.sum())
    denom = max(1.0, abs(fd), abs(ana))
    assert abs(fd - ana) / denom < 1e-3, (fd, ana)
