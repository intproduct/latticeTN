"""Stage 4A DMRG sweep smoke tests.

Small systems: DMRG final energy vs exact (N<=6), not-below-ground, energy
monotonic-ish. Medium smoke (N=8): finite, energy decreases vs initial, bond
dims <= chi, fast runtime. No dense ED at N=8.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn import dmrg as D  # noqa: E402
from latticetn import contractions as K  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _run(N, chi, sweeps=4, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return D.run_dmrg(mps, mpo, chi=chi, num_sweeps=sweeps, seed=seed)


def test_dmrg_n4_matches_exact_within_tol():
    res = _run(N=4, chi=8, sweeps=4)
    E0 = res["exact_energy"]
    assert abs(res["final_energy"] - E0) < 1e-6
    assert not res["below_ground"]


def test_dmrg_n6_matches_exact_within_tol():
    res = _run(N=6, chi=8, sweeps=4)
    E0 = res["exact_energy"]
    assert abs(res["final_energy"] - E0) < 1e-6
    assert not res["below_ground"]


def test_dmrg_energy_does_not_undershoot_exact():
    for N in (4, 5, 6):
        res = _run(N=N, chi=8, sweeps=3, seed=1)
        E0 = res["exact_energy"]
        assert res["final_energy"] >= E0 - 1e-6, (N, res["final_energy"], E0)


def test_dmrg_energy_history_nonincreasing_or_reported():
    # Energy should overall not rise; allow tiny numerical wiggles (< 1e-6).
    res = _run(N=6, chi=8, sweeps=4)
    es = [h["energy"] for h in res["history"]]
    for a, b in zip(es, es[1:]):
        assert b <= a + 1e-6, es
    assert es[-1] <= es[0] + 1e-6


def test_dmrg_n8_smoke_finite_energy_down_and_bond_capped():
    t0 = time.perf_counter()
    # N=8 smoke: no dense ED; only finite + energy decrease + bond caps + runtime.
    tc.manual_seed(0)
    N, chi = 8, 8
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    e_init = float(K.rayleigh_energy_native(mps, mpo))
    res = D.run_dmrg(mps, mpo, chi=chi, num_sweeps=3, seed=0)
    elapsed = time.perf_counter() - t0
    assert tc.isfinite(tc.as_tensor(res["final_energy"]))
    assert res["final_energy"] < e_init + 1e-6     # energy decreased
    assert res["final_max_bond"] <= chi
    # per-sweep truncation errors finite & nonneg
    for h in res["history"]:
        for e in h["truncation_errors"]:
            assert 0.0 <= e <= 1.0 + 1e-12
    assert elapsed < 60.0                            # not a "long training" job
