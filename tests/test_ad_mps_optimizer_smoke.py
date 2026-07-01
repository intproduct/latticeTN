"""Stage 4R AD-MPS optimizer smoke tests.

Adam short training lowers the energy; N=4/6 final energy vs exact
(not-below-ground; within a reported tolerance).
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

# Tolerances recorded in the Stage 4R report: Adam with 200 steps reaches
# these accuracies on N=4/N=6 (seed 0). Reported here so the test is the
# contract; if these relax the report must be updated with justification.
TOL = {4: 1e-6, 6: 1e-3}


def _run(N, chi, steps, lr=1e-2, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    ad = ADVariationalMPS(mps, mpo)
    res = train_ad_mps(ad, num_steps=steps, lr=lr, optimizer="adam")
    E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=DTYPE))
    res["exact_energy"] = E0
    return res


def test_adam_short_training_lowers_energy():
    res = _run(N=4, chi=8, steps=60, lr=1e-2)
    assert res["final_energy"] < res["initial_energy"] - 1e-6


def test_adam_n4_within_tolerance_and_not_below_ground():
    res = _run(N=4, chi=8, steps=200, lr=1e-2)
    E0 = res["exact_energy"]
    assert abs(res["final_energy"] - E0) < TOL[4]
    assert res["final_energy"] >= E0 - 1e-6


def test_adam_n6_within_tolerance_and_not_below_ground():
    res = _run(N=6, chi=8, steps=200, lr=1e-2)
    E0 = res["exact_energy"]
    assert abs(res["final_energy"] - E0) < TOL[6]
    assert res["final_energy"] >= E0 - 1e-6


def test_training_records_history_and_metadata():
    res = _run(N=4, chi=8, steps=50, lr=1e-2)
    assert len(res["energy_history"]) >= 2
    assert len(res["grad_norm_history"]) == len(res["energy_history"])
    assert len(res["norm_history"]) == len(res["energy_history"])
    assert res["optimizer"] == "adam"
    assert res["max_bond"] >= 1
    assert tc.isfinite(tc.as_tensor(res["final_energy"]))


def test_lbfgs_optimizer_runs_and_lowers_energy():
    tc.manual_seed(0)
    mps = MPS(4, 2, 8, dtype=DTYPE)
    mpo = MPO.from_bonds(4, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    ad = ADVariationalMPS(mps, mpo)
    res = train_ad_mps(ad, num_steps=15, lr=1.0, optimizer="lbfgs", lbfgs_iters=20)
    assert res["final_energy"] < res["initial_energy"] - 1e-6
