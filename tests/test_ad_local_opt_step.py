"""Stage 5A AD local-tensor optimization — step & convergence tests.

A local optimizer step lowers the energy; a full sweep lowers it further; the
final energy is at or above the exact ground energy (no variational cheating).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_local import ADLocalOptimizer, train_ad_local  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _mpo(N):
    return MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)


def test_single_local_step_lowers_energy():
    tc.manual_seed(0)
    mps = MPS(4, 2, 8, dtype=DTYPE)
    adlo = ADLocalOptimizer(mps, _mpo(4), center=0)
    e0 = float(adlo.loss().real)
    opt = tc.optim.Adam(adlo.parameters(), lr=1e-2)
    opt.zero_grad()
    adlo.loss().backward()
    opt.step()
    e1 = float(adlo.loss().real)
    assert e1 < e0, (e0, e1)


def test_single_lbfgs_local_opt_lowers_energy():
    tc.manual_seed(0)
    mps = MPS(4, 2, 8, dtype=DTYPE)
    adlo = ADLocalOptimizer(mps, _mpo(4), center=1)
    e0 = float(adlo.loss().real)
    opt = tc.optim.LBFGS(adlo.parameters(), lr=1.0, max_iter=20,
                         line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        e = adlo.loss()
        e.backward()
        return e
    opt.step(closure)
    e1 = float(adlo.loss().real)
    assert e1 <= e0 + 1e-9, (e0, e1)


def test_full_sweep_converges_n4():
    tc.manual_seed(0)
    mps = MPS(4, 2, 8, dtype=DTYPE)
    mpo = _mpo(4)
    E0, _ = exact_ground_energy(heisenberg_dense(4, dtype=DTYPE, device="cpu"))
    res = train_ad_local(mps, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                         optimizer="lbfgs", stabilization="none")
    assert res["final_energy"] >= E0 - 1e-6, (res["final_energy"], E0)
    assert abs(res["final_energy"] - E0) < 1e-6, (res["final_energy"], E0)
    # energy strictly decreased across sweeps
    hist = res["energy_history"]
    assert hist[-1] < hist[0]


def test_full_sweep_n6_within_tolerance():
    tc.manual_seed(0)
    mps = MPS(6, 2, 8, dtype=DTYPE)
    mpo = _mpo(6)
    E0, _ = exact_ground_energy(heisenberg_dense(6, dtype=DTYPE, device="cpu"))
    res = train_ad_local(mps, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                         optimizer="lbfgs", stabilization="qr")
    assert res["final_energy"] >= E0 - 1e-6, (res["final_energy"], E0)
    assert abs(res["final_energy"] - E0) < 1e-5, (res["final_energy"], E0)


def test_no_below_ground_across_stabilizations():
    for stab in ("none", "tensor_norm", "qr", "canonical"):
        tc.manual_seed(0)
        mps = MPS(4, 2, 8, dtype=DTYPE)
        mpo = _mpo(4)
        E0, _ = exact_ground_energy(heisenberg_dense(4, dtype=DTYPE, device="cpu"))
        res = train_ad_local(mps, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                             optimizer="lbfgs", stabilization=stab)
        assert res["final_energy"] >= E0 - 1e-6, (stab, res["final_energy"], E0)


def test_history_records_are_populated():
    tc.manual_seed(0)
    mps = MPS(4, 2, 8, dtype=DTYPE)
    res = train_ad_local(mps, _mpo(4), num_sweeps=2, local_steps=10, lr=1.0,
                         optimizer="lbfgs", stabilization="none")
    for key in ("energy_history", "grad_norm_history", "state_norm_history",
                "canonical_error_history", "sweeps", "initial_energy",
                "final_energy", "max_bond", "stabilization"):
        assert key in res, key
    assert len(res["energy_history"]) >= 2
    assert len(res["sweeps"]) == 2
