"""Stage 5A AD gauge vs baseline tests.

The canonical projection's final energy must not be materially worse than the
tensor_norm baseline (Stage 4R). Same seed/system, compare final energies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _run_proj(N, chi, steps, projection, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    ad = ADVariationalMPS(mps, mpo)
    return train_ad_mps(ad, num_steps=steps, lr=1e-2, projection=projection)


def test_canonical_not_worse_than_tensor_norm_n6():
    # Run both from the SAME seed; canonical should be at least as good (both
    # target the same variational minimum). Allow a tiny numerical margin.
    r_tn = _run_proj(6, 8, 200, "tensor_norm")
    r_can = _run_proj(6, 8, 200, "canonical")
    E0, _ = exact_ground_energy(heisenberg_dense(6, dtype=DTYPE))
    # both variational -> >= exact
    assert r_tn["final_energy"] >= E0 - 1e-6
    assert r_can["final_energy"] >= E0 - 1e-6
    # canonical not materially worse (within 1e-4 of tensor_norm)
    assert r_can["final_energy"] <= r_tn["final_energy"] + 1e-4, (
        r_tn["final_energy"], r_can["final_energy"])


def test_none_projection_is_weakest_or_comparable():
    # Without any gauge/scale stabilization, none should not beat the
    # stabilized variants by more than a tiny margin (it usually converges
    # worse on this problem). We assert none is not better-than-canonical by
    # more than a small tolerance (sanity on gauge benefit).
    r_none = _run_proj(6, 8, 200, "none")
    r_can = _run_proj(6, 8, 200, "canonical")
    # canonical should be at least as good as none (within tol)
    assert r_can["final_energy"] <= r_none["final_energy"] + 1e-3, (
        r_none["final_energy"], r_can["final_energy"])
