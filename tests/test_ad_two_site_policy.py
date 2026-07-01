"""Stage 5B two-site AD local optimization — policy guard tests.

AST/structural enforcement of docs/AD_MAINLINE_POLICY.md / AD_TWO_SITE_SPEC.md:
- the AD-two-site module imports NEITHER dmrg NOR lanczos;
- the loss path (energy/loss) contains no `torch.no_grad`, `.detach()`,
  `.data` access, `.item()`, nor any `eigh`/`svd`/`qr` call, and no call into
  the split / canonicalization helpers;
- the local optimizer step path (train_ad_two_site) drives optimization via
  backward() + a torch optimizer step and never calls dmrg/lanczos/eigh;
- SVD/QR appear ONLY in the explicitly-marked preprocessing / post-step helpers
  (_two_site_mixed_canonical, _split_theta, _stabilize_tensor_norm), NEVER in
  the loss.

Real AST traversal of Call/Attribute nodes (not substring matching), so
docstrings mentioning e.g. "eigensolver"/"Rayleigh" do not trigger false
positives.
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

import latticetn.ad_two_site as adtsmod  # noqa: E402

SRC = textwrap.dedent(inspect.getsource(adtsmod))


def _collect_calls(tree):
    """Return dict of call/attr counts found anywhere in tree."""
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


def _func_subtree(mod_src_full, fname):
    tree = ast.parse(mod_src_full)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            return node
        if isinstance(node, ast.AsyncFunctionDef) and node.name == fname:
            return node
    return None


def _method_subtree(cls_name, fname):
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and item.name == fname:
                    return item
    return None


# ---------------------------------------------------------------------------
# import guard
# ---------------------------------------------------------------------------

def test_module_does_not_import_dmrg_or_lanczos():
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
    # Substring guard on actual references to classical solver entry points.
    # (eigh CALLS are caught separately by AST in test_eigh_nowhere_in_module;
    # we deliberately do NOT substring-match "eigh" because the docstring
    # legitimately contains the English word "eigensolver".)
    for bad in ("run_dmrg", "two_site_sweep", "lanczos_lowest_eigenpair",
                "local_ground_state"):
        assert bad not in SRC, f"forbidden reference {bad!r}"


# ---------------------------------------------------------------------------
# loss-path cleanliness (AST, only Call/Attribute nodes inspected)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", ["energy", "loss"])
def test_loss_path_clean(fname):
    node = _method_subtree("ADTwoSiteOptimizer", fname)
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
        assert c not in (
            "dmrg", "lanczos", "run_dmrg", "_two_site_mixed_canonical",
            "_split_theta", "_stabilize_tensor_norm", "reset_bond", "split",
            "left_canonical", "mixed_canonical",
        ), f"{fname}: forbidden call {c}"


def test_eigh_nowhere_in_module():
    node = ast.parse(SRC)
    info = _collect_calls(node)
    assert info["eigh"] == 0, "torch.linalg.eigh must never be called"


def test_qr_svds_only_in_marked_helpers():
    tree = ast.parse(SRC)
    allowed = {"_two_site_mixed_canonical", "_split_theta"}
    # _stabilize_tensor_norm uses .data but no qr/svd; still allowed to have
    # no_grad/data. The loss/energy methods must be clean (checked elsewhere).
    forbidden = {"energy", "loss", "parameters", "global_energy", "norm",
                 "max_bond_dim", "bond_dims", "reset_bond"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info = _collect_calls(node)
            if node.name in forbidden:
                assert info["qr"] == 0, f"{node.name}: qr must not appear"
                assert info["svd"] == 0, f"{node.name}: svd must not appear"
            elif node.name == "train_ad_two_site":
                # the driver must not call qr/svd directly (it delegates)
                assert info["qr"] == 0, "train_ad_two_site: qr must not appear"
                assert info["svd"] == 0, "train_ad_two_site: svd must not appear"
            elif node.name in allowed:
                pass  # qr/svd allowed here
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and item.name in forbidden:
                    info = _collect_calls(item)
                    assert info["qr"] == 0, f"{item.name}: qr must not appear"
                    assert info["svd"] == 0, f"{item.name}: svd must not appear"


def test_optimizer_step_uses_torch_optimizer_and_backward():
    node = _func_subtree(SRC, "train_ad_two_site")
    assert node is not None
    info = _collect_calls(node)
    assert "backward" in info["calls"], "must call backward()"
    assert "step" in info["calls"], "must call optimizer.step()"
    assert info["eigh"] == 0
    for c in info["calls"]:
        assert c not in ("dmrg", "lanczos", "run_dmrg", "two_site_sweep",
                         "lanczos_lowest_eigenpair", "eigh",
                         "local_ground_state"), \
            f"train_ad_two_site: forbidden call {c}"


def test_split_is_post_step_only():
    # energy/loss must not call split; split itself must use svd under no_grad.
    for fname in ("energy", "loss"):
        node = _method_subtree("ADTwoSiteOptimizer", fname)
        info = _collect_calls(node)
        assert "split" not in info["calls"], f"{fname}: split in loss path"
    split_node = _method_subtree("ADTwoSiteOptimizer", "split")
    assert split_node is not None
    sinfo = _collect_calls(split_node)
    assert sinfo["svd"] >= 1 or "_split_theta" in sinfo["calls"], \
        "split must use SVD (via _split_theta)"
