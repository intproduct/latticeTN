"""Stage 7A: AD mainline solvers on the spinless fermion t-V chain.

The three AD mainline solvers (unchanged) must work on the NEW fermion
Hamiltonian/MPO:

- global AD-MPS            (ad_variational.train_ad_mps, Adam)
- one-site AD local opt    (ad_local.train_ad_local, LBFGS)
- two-site AD local opt    (ad_two_site.train_ad_two_site, LBFGS)

Each must LOWER the energy from its initial value and must NOT undershoot the
exact ground energy beyond tolerance (``below_ground`` guard). The exact ground
energy (``exact_ground_energy``) is the reference baseline; no DMRG/Lanczos is
used. The AD loss path is unchanged — only the Hamiltonian/MPO layer is new.

Conventions: open boundary, complex128, CPU-only, small N (4, 6).
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
from latticetn.ad_local import train_ad_local  # noqa: E402
from latticetn.ad_two_site import train_ad_two_site  # noqa: E402
from latticetn.operators import spinless_fermion_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128
BELOW_GROUND_TOL = 1e-6


def _exact_e0(N, t, V, mu):
    H = spinless_fermion_dense(N, t=t, V=V, mu=mu, dtype=DTYPE)
    return float(exact_ground_energy(H)[0])


def _mps_mpo(N, chi, t, V, mu, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_spinless_fermion(
        t=t, V=V, mu=mu)
    return mps, mpo


def test_global_ad_lowers_energy_and_not_below_ground():
    for N, chi in [(4, 4), (6, 8)]:
        t, V, mu = 1.0, 0.5, 0.0
        e0_exact = _exact_e0(N, t, V, mu)
        mps, mpo = _mps_mpo(N, chi, t, V, mu, seed=1)
        ad = ADVariationalMPS(mps, mpo)
        r = train_ad_mps(ad, num_steps=120, lr=1e-2, optimizer="adam",
                         projection="tensor_norm")
        assert r["final_energy"] < r["initial_energy"], (N, r)
        assert r["final_energy"] >= e0_exact - BELOW_GROUND_TOL, (N, r, e0_exact)


def test_one_site_ad_lowers_energy_and_not_below_ground():
    for N, chi in [(4, 4), (6, 8)]:
        t, V, mu = 1.0, 0.5, 0.0
        e0_exact = _exact_e0(N, t, V, mu)
        mps, mpo = _mps_mpo(N, chi, t, V, mu, seed=2)
        r = train_ad_local(mps, mpo, num_sweeps=2, local_steps=10, lr=1.0,
                           optimizer="lbfgs", stabilization="qr")
        assert r["final_energy"] < r["initial_energy"], (N, r)
        assert r["final_energy"] >= e0_exact - BELOW_GROUND_TOL, (N, r, e0_exact)


def test_two_site_ad_lowers_energy_and_not_below_ground():
    for N, chi in [(4, 4), (6, 8)]:
        t, V, mu = 1.0, 0.5, 0.0
        e0_exact = _exact_e0(N, t, V, mu)
        mps, mpo = _mps_mpo(N, chi, t, V, mu, seed=3)
        r = train_ad_two_site(mps, mpo, num_sweeps=2, local_steps=10, lr=1.0,
                              optimizer="lbfgs", max_bond_dim=chi, cutoff=None)
        assert r["final_energy"] < r["initial_energy"], (N, r)
        assert r["final_energy"] >= e0_exact - BELOW_GROUND_TOL, (N, r, e0_exact)


def test_global_ad_attractive_regime():
    # V < 0 (attractive) — different regime, still must lower + not below ground.
    N, chi = 6, 8
    t, V, mu = 1.0, -1.0, 0.0
    e0_exact = _exact_e0(N, t, V, mu)
    mps, mpo = _mps_mpo(N, chi, t, V, mu, seed=5)
    ad = ADVariationalMPS(mps, mpo)
    r = train_ad_mps(ad, num_steps=120, lr=1e-2, optimizer="adam",
                     projection="tensor_norm")
    assert r["final_energy"] < r["initial_energy"]
    assert r["final_energy"] >= e0_exact - BELOW_GROUND_TOL, (r, e0_exact)


def test_two_site_ad_approaches_exact_for_small_N():
    # For very small N, two-site AD with enough sweeps should get close to ED.
    N, chi = 4, 8
    t, V, mu = 1.0, 0.0, 0.0  # free fermion
    e0_exact = _exact_e0(N, t, V, mu)
    mps, mpo = _mps_mpo(N, chi, t, V, mu, seed=7)
    r = train_ad_two_site(mps, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                          optimizer="lbfgs", max_bond_dim=chi, cutoff=None)
    assert abs(r["final_energy"] - e0_exact) < 1e-3, (r["final_energy"], e0_exact)
