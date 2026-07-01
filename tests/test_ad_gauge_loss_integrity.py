"""Stage 5A AD gauge loss-integrity tests.

Verifies the loss is a differentiable scalar, backward populates all grads,
and the loss path is structurally free of dmrg/lanczos/projection/detach/
.data/no_grad/unnecessary item (AST inspection).
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
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


def test_loss_scalar_finite_requires_grad():
    ad = _make()
    e = ad.loss()
    assert e.dim() == 0
    assert tc.isfinite(e).all()
    assert e.requires_grad


def test_backward_all_grads_not_none_and_finite():
    ad = _make(N=5, chi=4, seed=1)
    e = ad.loss()
    e.backward()
    for i, p in enumerate(ad.parameters()):
        assert p.grad is not None, i
        assert tc.isfinite(p.grad).all(), i


def _stmts(mod, fname):
    src = textwrap.dedent(inspect.getsource(mod))
    tree = ast.parse(src)
    seg = inspect.getsource(mod)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            return [ast.get_source_segment(seg, st) or ""
                    for st in ast.iter_child_nodes(node)]
    return []


def test_loss_path_has_no_dmrg_lanczos_projection_or_forbidden():
    # energy()/loss() are the loss path; must not call projection, dmrg, lanczos,
    # no_grad, .detach(), .data, or .item().
    for fname in ("energy", "loss"):
        body = "\n".join(_stmts(advmod, fname))
        for bad in ("no_grad", ".detach()", ".data", ".item()",
                    "_project", "_renormalize", "left_canonical",
                    "dmrg", "lanczos", ".run_dmrg"):
            assert bad not in body, f"{fname}: found {bad!r}"

    # rayleigh_energy_native (the underlying loss) must be clean too.
    srcK = textwrap.dedent(inspect.getsource(K))
    treeK = ast.parse(srcK)
    for node in ast.walk(treeK):
        if isinstance(node, ast.FunctionDef) and node.name == "rayleigh_energy_native":
            body = "\n".join(ast.get_source_segment(srcK, st) or ""
                             for st in ast.iter_child_nodes(node))
            for bad in ("no_grad", ".detach()", ".data", ".item()",
                        "dmrg", "lanczos", "_project"):
                assert bad not in body, f"rayleigh: found {bad!r}"
