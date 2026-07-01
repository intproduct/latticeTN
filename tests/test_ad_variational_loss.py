"""Stage 4R AD variational loss tests.

Verifies the loss is a finite scalar with requires_grad, and that the loss
source path is autograd-clean (no detach/.data/no_grad wrapping the loss).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS  # noqa: E402
import latticetn.ad_variational as advmod  # noqa: E402
import latticetn.contractions as K  # noqa: E402

DTYPE = tc.complex128


def _make(N=4, chi=8, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return ADVariationalMPS(mps, mpo)


def test_loss_is_finite_scalar_with_requires_grad():
    ad = _make()
    e = ad.loss()
    assert e.dim() == 0
    assert tc.isfinite(e).all()
    assert e.requires_grad


def test_loss_uses_native_rayleigh_and_is_real():
    ad = _make()
    e = ad.loss()
    # rayleigh_energy_native returns .real -> real-valued tensor
    assert not e.is_complex()
    assert abs(float(e.imag) if e.is_complex() else 0.0) < 1e-12
    # equals the native contraction directly
    e2 = K.rayleigh_energy_native(ad.mps, ad.mpo)
    assert tc.allclose(e, e2, atol=1e-12)


def test_loss_source_has_no_forbidden_patterns():
    # The differentiable loss lives in ADVariationalMPS.energy/loss, which call
    # contractions.rayleigh_energy_native. Ensure none of those method bodies
    # wrap the loss in torch.no_grad or invoke .data / .detach / .item inside
    # the energy path. We strip docstrings and comments before grepping so the
    # check reflects executable code only.
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(advmod)))
    flagged = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("energy", "loss"):
            # collect only executable statements, skipping docstrings/comments
            stmts = []
            for st in ast.iter_child_nodes(node):
                stmts.append(ast.get_source_segment(textwrap.dedent(inspect.getsource(advmod)), st) or "")
            body = "\n".join(stmts)
            flagged[node.name] = body

    for name, body in flagged.items():
        assert "no_grad" not in body, name
        assert ".detach()" not in body, name
        assert ".data" not in body, name
        assert ".item()" not in body, name

    # rayleigh_energy_native itself must be clean too
    treeK = ast.parse(textwrap.dedent(inspect.getsource(K)))
    for node in ast.walk(treeK):
        if isinstance(node, ast.FunctionDef) and node.name == "rayleigh_energy_native":
            stmts = [ast.get_source_segment(textwrap.dedent(inspect.getsource(K)), st) or ""
                     for st in ast.iter_child_nodes(node)]
            body = "\n".join(stmts)
            assert "no_grad" not in body
            assert ".detach()" not in body
            assert ".data" not in body


def test_loss_does_not_change_under_mps_scaling():
    # Rayleigh quotient is scale-invariant -> scaling the MPS by a constant
    # should leave the loss unchanged (autograd path).
    ad = _make()
    e_before = float(ad.loss())
    with tc.no_grad():
        for p in ad.parameters():
            p.mul_(2.5 + 0.0j)
    e_after = float(ad.loss())
    assert abs(e_before - e_after) < 1e-9
