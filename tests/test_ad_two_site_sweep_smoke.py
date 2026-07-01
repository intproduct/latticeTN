"""Stage 5B two-site AD local optimization — sweep smoke tests (N=4, N=6).

Verifies a full two-site AD sweep lowers the energy, the final energy is not
below the exact ground energy beyond tolerance, and sweep direction alternates.
CPU-only, small systems, fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_two_site import train_ad_two_site  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _exact(N):
    H = heisenberg_dense(N, dtype=DTYPE, device="cpu")
    return exact_ground_energy(H)[0]


def _mpo(N):
    return MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)


def _run(N, chi=8, num_sweeps=4, local_steps=20, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    return train_ad_two_site(mps, _mpo(N), num_sweeps=num_sweeps,
                             local_steps=local_steps, lr=1.0,
                             optimizer="lbfgs", max_bond_dim=chi)


def test_sweep_n4_energy_decreases():
    res = _run(4, chi=8, num_sweeps=3, local_steps=20)
    assert res["final_energy"] <= res["initial_energy"] + 1e-9, (
        res["initial_energy"], res["final_energy"])


def test_sweep_n6_energy_decreases():
    res = _run(6, chi=8, num_sweeps=4, local_steps=20)
    assert res["final_energy"] <= res["initial_energy"] + 1e-9, (
        res["initial_energy"], res["final_energy"])


def test_sweep_n4_final_near_exact_not_below():
    res = _run(4, chi=8, num_sweeps=4, local_steps=20)
    e0 = _exact(4)
    err = abs(res["final_energy"] - e0)
    assert err < 1e-6, (res["final_energy"], e0, err)
    # variational: must not be below ground beyond a tiny tolerance
    assert res["final_energy"] >= e0 - 1e-8, (res["final_energy"], e0)


def test_sweep_n6_final_near_exact_not_below():
    res = _run(6, chi=8, num_sweeps=5, local_steps=20)
    e0 = _exact(6)
    err = abs(res["final_energy"] - e0)
    assert err < 1e-4, (res["final_energy"], e0, err)
    assert res["final_energy"] >= e0 - 1e-6, (res["final_energy"], e0)


def test_sweep_direction_alternates():
    res = _run(4, chi=8, num_sweeps=4, local_steps=5)
    dirs = [s["direction"] for s in res["sweeps"]]
    assert dirs[0] == "right"
    assert dirs == ["right", "left", "right", "left"], dirs


def test_sweep_truncation_errors_nonneg_finite():
    res = _run(6, chi=4, num_sweeps=2, local_steps=5)
    for s in res["sweeps"]:
        for t in s["per_bond_trunc"]:
            assert t == t, "nan truncation"
            assert t >= 0.0, t


def test_sweep_bond_dim_respects_cap():
    res = _run(6, chi=4, num_sweeps=2, local_steps=5)
    assert res["max_bond"] <= 4
