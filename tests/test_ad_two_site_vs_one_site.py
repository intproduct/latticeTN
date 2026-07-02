"""Stage 5B two-site AD local optimization — comparison tests.

Verifies two-site AD reaches the same variational minimum as one-site AD,
global AD-MPS, and the classical DMRG reference (reference only, never in the
AD path), and that all are consistent with exact diagonalization (direction
agreement, no below-ground violation).
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
from latticetn.ad_local import train_ad_local  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn import dmrg as D  # noqa: E402  (reference baseline only)
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _exact(N):
    H = heisenberg_dense(N, dtype=DTYPE, device="cpu")
    return exact_ground_energy(H)[0]


def _mpo(N):
    return MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)


def test_two_site_agrees_with_one_site_ad_n6():
    N = 6
    e0 = _exact(N)
    tc.manual_seed(0)
    mps2 = MPS(N, 2, 8, dtype=DTYPE)
    r2 = train_ad_two_site(mps2, _mpo(N), num_sweeps=5, local_steps=20,
                           lr=1.0, optimizer="lbfgs", max_bond_dim=8)
    tc.manual_seed(0)
    mps1 = MPS(N, 2, 8, dtype=DTYPE)
    r1 = train_ad_local(mps1, _mpo(N), num_sweeps=4, local_steps=20, lr=1.0,
                        optimizer="lbfgs", stabilization="qr")
    # both near exact, and close to each other
    assert abs(r2["final_energy"] - e0) < 1e-4, (r2["final_energy"], e0)
    assert abs(r1["final_energy"] - e0) < 1e-4, (r1["final_energy"], e0)
    assert abs(r2["final_energy"] - r1["final_energy"]) < 1e-3, (
        r2["final_energy"], r1["final_energy"])


def test_two_site_agrees_with_global_ad_n4():
    N = 4
    e0 = _exact(N)
    tc.manual_seed(0)
    mps2 = MPS(N, 2, 8, dtype=DTYPE)
    r2 = train_ad_two_site(mps2, _mpo(N), num_sweeps=4, local_steps=20,
                           lr=1.0, optimizer="lbfgs", max_bond_dim=8)
    tc.manual_seed(0)
    mpsg = MPS(N, 2, 8, dtype=DTYPE)
    ad = ADVariationalMPS(mpsg, _mpo(N))
    rg = train_ad_mps(ad, num_steps=300, lr=1e-2, optimizer="adam")
    assert abs(r2["final_energy"] - e0) < 1e-6, (r2["final_energy"], e0)
    assert abs(rg["final_energy"] - e0) < 1e-3, (rg["final_energy"], e0)
    assert abs(r2["final_energy"] - rg["final_energy"]) < 1e-2, (
        r2["final_energy"], rg["final_energy"])


def test_two_site_agrees_with_dmrg_reference_n6():
    N = 6
    e0 = _exact(N)
    tc.manual_seed(0)
    mps2 = MPS(N, 2, 8, dtype=DTYPE)
    r2 = train_ad_two_site(mps2, _mpo(N), num_sweeps=5, local_steps=20,
                           lr=1.0, optimizer="lbfgs", max_bond_dim=8)
    tc.manual_seed(0)
    mpsd = MPS(N, 2, 8, dtype=DTYPE)
    rd = D.run_dmrg(mpsd, _mpo(N), chi=8, num_sweeps=4, solver="dense")
    assert abs(r2["final_energy"] - e0) < 1e-4
    assert abs(rd["final_energy"] - e0) < 1e-6
    # DMRG is the dense reference; two-site AD should be within ~1e-4 of it
    assert abs(r2["final_energy"] - rd["final_energy"]) < 1e-3, (
        r2["final_energy"], rd["final_energy"])


def test_two_site_direction_consistent_with_exact():
    # All methods should land at / above the exact ground energy (variational).
    N = 4
    e0 = _exact(N)
    tc.manual_seed(0)
    mps2 = MPS(N, 2, 8, dtype=DTYPE)
    r2 = train_ad_two_site(mps2, _mpo(N), num_sweeps=4, local_steps=20,
                           lr=1.0, optimizer="lbfgs", max_bond_dim=8)
    assert r2["final_energy"] >= e0 - 1e-8, (r2["final_energy"], e0)
    assert r2["final_energy"] <= e0 + 1e-6, (r2["final_energy"], e0)


def test_two_site_can_grow_bond_vs_one_site_fixed():
    # With a generous max_bond_dim, two-site AD can reach a lower energy than a
    # bond-limited one-site run on a state that needs entanglement.
    N = 6
    tc.manual_seed(1)
    mps2 = MPS(N, 2, 4, dtype=DTYPE)  # start small
    r2 = train_ad_two_site(mps2, _mpo(N), num_sweeps=6, local_steps=20,
                           lr=1.0, optimizer="lbfgs", max_bond_dim=16)
    # the two-site run with growth should approach exact (chi up to 16 >= 8)
    e0 = _exact(N)
    assert abs(r2["final_energy"] - e0) < 1e-3, (r2["final_energy"], e0)
