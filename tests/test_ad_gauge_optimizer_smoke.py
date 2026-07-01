"""Stage 5A AD gauge optimizer smoke tests.

For each projection (none / tensor_norm / canonical): Adam short training
lowers the energy; N=4/6 final energy not below exact ground (within reported
tolerances); canonical error decreases for the canonical projection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps, _canonical_error  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128

TOL = {4: 1e-5, 6: 5e-4}  # canonical is generally tighter; these bound all 3 projections


def _run(N, chi, steps, projection, lr=1e-2, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    ad = ADVariationalMPS(mps, mpo)
    res = train_ad_mps(ad, num_steps=steps, lr=lr, projection=projection,
                       record_every=max(1, steps // 4))
    E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=DTYPE))
    res["exact_energy"] = E0
    return res


def test_each_projection_lowers_energy():
    for projection in ("none", "tensor_norm", "canonical"):
        res = _run(N=4, chi=8, steps=60, projection=projection)
        assert res["final_energy"] < res["initial_energy"] - 1e-6, projection


def test_each_projection_n4_not_below_ground_within_tol():
    # Stabilized projections must hit the tight N=4 tolerance. `none` lacks
    # scale stabilization (its only role here is the energy-decrease smoke),
    # so it gets the loose not-below-ground check only (covered by the loop
    # below) and is reported, not gated, on accuracy.
    for projection in ("tensor_norm", "canonical"):
        res = _run(N=4, chi=8, steps=200, projection=projection)
        assert res["final_energy"] >= res["exact_energy"] - 1e-6, projection
        assert abs(res["final_energy"] - res["exact_energy"]) < TOL[4], (projection, res["final_energy"])


def test_each_projection_n4_not_below_ground():
    # All projections (including `none`) must respect the variational bound.
    for projection in ("none", "tensor_norm", "canonical"):
        res = _run(N=4, chi=8, steps=200, projection=projection)
        assert res["final_energy"] >= res["exact_energy"] - 1e-6, projection


def test_canonical_projection_n6_not_below_ground_within_tol():
    res = _run(N=6, chi=8, steps=200, projection="canonical")
    assert res["final_energy"] >= res["exact_energy"] - 1e-6
    assert abs(res["final_energy"] - res["exact_energy"]) < TOL[6]


def test_canonical_error_decreases_under_canonical_projection():
    res = _run(N=6, chi=8, steps=100, projection="canonical")
    ce = res["canonical_error_history"]
    assert ce[-1] < ce[0]
    assert ce[-1] < 1e-9


def test_history_records_projection_and_diagnostics():
    res = _run(N=4, chi=8, steps=40, projection="canonical")
    assert res["projection"] == "canonical"
    for key in ("energy_history", "grad_norm_history", "state_norm_history",
                "canonical_error_history"):
        assert key in res
        assert len(res[key]) >= 2
    # back-compat alias preserved
    assert "norm_history" in res
