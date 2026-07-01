"""Stage 5A AD local-tensor optimization — policy guard tests.

AST/structural enforcement of docs/AD_MAINLINE_POLICY.md:
- the AD-local module imports NEITHER dmrg NOR lanczos;
- the loss path (energy/loss + the underlying rayleigh_energy_native) contains
  no calls to `torch.no_grad`, `.detach()`, `.data` access, `.item()`, nor any
  `eigh`/`svd`/`qr`/dmrg/lanczos call;
- the local optimizer step path does not call dmrg/lanczos/eigh;
- SVD/QR/canonicalization, if present, live only in explicitly-marked post-step
  / center-movement helpers, never in the loss.

These checks use real AST traversal of Call/Attribute nodes (not raw substring
matching), so docstrings mentioning e.g. "Rayleigh"/"eigensolver" do not trigger
false positives.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import latticetn.ad_local as adlocmod  # noqa: E402
import latticetn.contractions as K  # noqa: E402

SRC = textwrap.dedent(inspect.getsource(adlocmod))
SRC_K = textwrap.dedent(inspect.getsource(K))


def _module_docstrings_and_strings(tree):
    strs = set()

    class StrVisitor(ast.NodeVisitor):
        def visit_Constant(self, node):  # noqa: N802
            if isinstance(node.value, str):
                strs.add(node.value)
            self.generic_visit(node)

    StrVisitor().visit(tree)
    return strs


def _collect_calls(tree):
    """Return (call_names, attr_names, no_grad_calls, eigh_calls, svd_calls,
    qr_calls, detach_calls, data_access, item_calls) found anywhere in tree."""
    calls = []
    attrs = []
    no_grad = eigh = svd = qr = detach = data = item = 0

    class V(ast.NodeVisitor):
        def visit_Call(self, node):  # noqa: N802
            f = node.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                name = f.attr
            calls.append(name)
            if name == "no_grad":
                nonlocal no_grad; no_grad += 1
            if name == "eigh":
                nonlocal eigh; eigh += 1
            if name == "svd":
                nonlocal svd; svd += 1
            if name == "qr":
                nonlocal qr; qr += 1
            if name == "item":
                nonlocal item; item += 1
            self.generic_visit(node)

        def visit_Attribute(self, node):  # noqa: N802
            attrs.append(node.attr)
            if node.attr == "detach":
                nonlocal detach; detach += 1
            if node.attr == "data":
                nonlocal data; data += 1
            self.generic_visit(node)

    V().visit(tree)
    return dict(calls=calls, attrs=attrs, no_grad=no_grad, eigh=eigh, svd=svd,
                qr=qr, detach=detach, data=data, item=item)


def _func_subtree(mod_src_full, mod_obj, fname):
    tree = ast.parse(mod_src_full)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            return node
    return None


# ---------------------------------------------------------------------------
# import guard (substring on import statements is fine — no English words here)
# ---------------------------------------------------------------------------

def test_module_does_not_import_dmrg_or_lanczos():
    # Only inspect actual import/alias statements, not docstrings.
    tree = ast.parse(SRC)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
            imported.extend(a.name for a in node.names)
    joined = " ".join(imported)
    assert "dmrg" not in joined
    assert "lanczos" not in joined
    # confirm no bare textual references to the classical baselines/solvers
    for bad in ("run_dmrg", "two_site_sweep", "lanczos_lowest_eigenpair"):
        assert bad not in SRC


# ---------------------------------------------------------------------------
# loss-path cleanliness (AST, docstring-stripped implicitly — we only inspect
# Call/Attribute nodes, never string literals)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", ["energy", "loss"])
def test_loss_path_clean(fname):
    node = _func_subtree(SRC, adlocmod, fname)
    assert node is not None, fname
    info = _collect_calls(node)
    assert info["no_grad"] == 0, f"{fname}: torch.no_grad in loss"
    assert info["detach"] == 0, f"{fname}: .detach in loss"
    assert info["data"] == 0, f"{fname}: .data in loss"
    assert info["item"] == 0, f"{fname}: .item in loss"
    assert info["eigh"] == 0, f"{fname}: eigh in loss"
    assert info["svd"] == 0, f"{fname}: svd in loss"
    assert info["qr"] == 0, f"{fname}: qr in loss"
    for c in info["calls"]:
        assert c not in ("dmrg", "lanczos", "run_dmrg", "_move_center_right",
                         "_move_center_left", "_stabilize", "left_canonical",
                         "mixed_canonical"), f"{fname}: forbidden call {c}"


def test_rayleigh_energy_native_clean():
    node = _func_subtree(SRC_K, K, "rayleigh_energy_native")
    assert node is not None
    info = _collect_calls(node)
    for key in ("no_grad", "detach", "data", "item", "eigh", "svd", "qr"):
        assert info[key] == 0, f"rayleigh: {key} present"
    for c in info["calls"]:
        assert c not in ("dmrg", "lanczos", "run_dmrg", "_move_center_right",
                         "_move_center_left", "_stabilize")


def test_native_mpo_numerator_and_norm_sq_clean():
    for fname in ("native_mpo_numerator", "native_norm_sq", "native_mpo_expectation"):
        node = _func_subtree(SRC_K, K, fname)
        assert node is not None, fname
        info = _collect_calls(node)
        for key in ("no_grad", "detach", "data", "item", "eigh", "svd", "qr"):
            assert info[key] == 0, f"{fname}: {key} present"
        for c in info["calls"]:
            assert c not in ("dmrg", "lanczos", "run_dmrg", "_stabilize",
                             "_move_center_right", "_move_center_left")


# ---------------------------------------------------------------------------
# QR/SVD/eigh scope discipline
# ---------------------------------------------------------------------------

def test_eigh_nowhere_in_module():
    node = ast.parse(SRC)
    info = _collect_calls(node)
    assert info["eigh"] == 0, "torch.linalg.eigh must never be called"


def test_qr_svds_only_in_marked_helpers():
    # QR/SVD calls may only live in _move_center_right/_move_center_left/
    # _stabilize (post-step / center-movement), never in energy/loss/__init__/
    # set_center/parameters/norm/train_ad_local's gradient region.
    tree = ast.parse(SRC)
    allowed = {"_move_center_right", "_move_center_left", "_stabilize"}
    forbidden = {"energy", "loss", "__init__", "set_center", "parameters",
                 "norm", "max_bond_dim", "move_center"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            info = _collect_calls(node)
            if node.name in forbidden:
                assert info["qr"] == 0, f"{node.name}: qr must not appear"
                assert info["svd"] == 0, f"{node.name}: svd must not appear"
            elif node.name in allowed:
                pass  # QR/SVD allowed here
            elif node.name == "train_ad_local":
                # the driver itself must not call qr/svd directly (it delegates)
                assert info["qr"] == 0, "train_ad_local: qr must not appear"
                assert info["svd"] == 0, "train_ad_local: svd must not appear"


def test_optimizer_step_uses_torch_optimizer_and_backward():
    node = _func_subtree(SRC, adlocmod, "train_ad_local")
    assert node is not None
    info = _collect_calls(node)
    # the driver must drive optimization via backward + a torch optimizer step
    assert "backward" in info["calls"], "must call backward()"
    assert "step" in info["calls"], "must call optimizer.step()"
    assert info["eigh"] == 0
    for c in info["calls"]:
        assert c not in ("dmrg", "lanczos", "run_dmrg", "two_site_sweep"), \
            f"train_ad_local: forbidden call {c}"
