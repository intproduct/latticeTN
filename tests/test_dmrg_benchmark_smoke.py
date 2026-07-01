"""Stage 4B DMRG benchmark smoke tests.

- N<=6: dense and lanczos DMRG final energy vs exact, not-below-ground.
- chi sweep: energy must not materially worsen as chi grows.
- N=10 CPU smoke (no dense ED): finite, energy decreases, bond dims <= chi,
  reasonable runtime.
CPU-only, small systems, fast.
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


def _dmrg(N, chi, sweeps, solver, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return D.run_dmrg(mps, mpo, chi=chi, num_sweeps=sweeps, solver=solver, seed=seed)


def _exact(N):
    H = heisenberg_dense(N, dtype=DTYPE, device="cpu")
    E0, _ = exact_ground_energy(H)
    return E0


def test_dense_dmrg_n4_n6_vs_exact():
    for N in (4, 6):
        res = _dmrg(N, chi=8, sweeps=4, solver="dense")
        E0 = _exact(N)
        assert abs(res["final_energy"] - E0) < 1e-6, (N, res["final_energy"], E0)
        assert not res["below_ground"]


def test_lanczos_dmrg_n4_n6_vs_exact():
    for N in (4, 6):
        res = _dmrg(N, chi=8, sweeps=4, solver="lanczos")
        E0 = _exact(N)
        assert abs(res["final_energy"] - E0) < 1e-6, (N, res["final_energy"], E0)
        assert not res["below_ground"]


def test_dmrg_not_below_exact_ground():
    for solver in ("dense", "lanczos"):
        res = _dmrg(N=6, chi=8, sweeps=3, solver=solver, seed=1)
        E0 = _exact(6)
        assert res["final_energy"] >= E0 - 1e-6, (solver, res["final_energy"], E0)


def test_chi_sweep_energy_does_not_worsen():
    # Larger chi should not materially worsen the energy (allow tiny numerical
    # wiggles). All convergent at full chi for N=6.
    N = 6
    E0 = _exact(N)
    energies = []
    for chi in (4, 8, 16):
        res = _dmrg(N, chi=chi, sweeps=4, solver="dense")
        energies.append(res["final_energy"])
        # each chi must still respect the variational bound
        assert res["final_energy"] >= E0 - 1e-6
    # energy is non-increasing as chi grows (within a small tolerance)
    for a, b in zip(energies, energies[1:]):
        assert b <= a + 1e-6, energies


def test_n10_smoke_finite_energy_down_bond_capped():
    t0 = time.perf_counter()
    tc.manual_seed(0)
    N, chi = 10, 16
    mps = MPS(N, 2, 8, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    e_init = float(K.rayleigh_energy_native(mps, mpo))
    res = D.run_dmrg(mps, mpo, chi=chi, num_sweeps=3, solver="lanczos")
    elapsed = time.perf_counter() - t0
    assert tc.isfinite(tc.as_tensor(res["final_energy"]))
    assert res["final_energy"] < e_init + 1e-6      # energy decreased
    assert res["final_max_bond"] <= chi
    assert elapsed < 90.0                            # not a long-training job


def test_solver_option_recorded_in_history():
    res = _dmrg(N=4, chi=8, sweeps=2, solver="lanczos")
    assert res["solver"] == "lanczos"
    assert all(h["solver"] == "lanczos" for h in res["history"])
