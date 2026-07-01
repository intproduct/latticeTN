"""Stage 5A AD local-tensor optimization — vs global AD-MPS comparison.

AD-local (one center tensor at a time, QR center sweep) and global AD-MPS (all
tensors trained simultaneously) are two strategies on the SAME differentiable
Rayleigh-quotient loss; both must reach the same variational minimum and stay
at/above the exact ground energy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_local import train_ad_local  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _mpo(N):
    return MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)


def _exact(N):
    return exact_ground_energy(heisenberg_dense(N, dtype=DTYPE, device="cpu"))[0]


def test_ad_local_matches_global_ad_n4():
    N = 4
    E0 = _exact(N)
    mpo = _mpo(N)

    tc.manual_seed(0)
    mps_g = MPS(N, 2, 8, dtype=DTYPE)
    ad_g = ADVariationalMPS(mps_g, mpo)
    r_global = train_ad_mps(ad_g, num_steps=300, lr=1e-2, optimizer="adam")

    tc.manual_seed(0)
    mps_l = MPS(N, 2, 8, dtype=DTYPE)
    r_local = train_ad_local(mps_l, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                             optimizer="lbfgs", stabilization="qr")

    assert r_global["final_energy"] >= E0 - 1e-6
    assert r_local["final_energy"] >= E0 - 1e-6
    # both reach the same variational minimum (local is more accurate here)
    assert abs(r_local["final_energy"] - r_global["final_energy"]) < 1e-3, \
        (r_local["final_energy"], r_global["final_energy"], E0)


def test_ad_local_matches_global_ad_n6():
    N = 6
    E0 = _exact(N)
    mpo = _mpo(N)

    tc.manual_seed(0)
    mps_g = MPS(N, 2, 8, dtype=DTYPE)
    ad_g = ADVariationalMPS(mps_g, mpo)
    r_global = train_ad_mps(ad_g, num_steps=300, lr=1e-2, optimizer="adam")

    tc.manual_seed(0)
    mps_l = MPS(N, 2, 8, dtype=DTYPE)
    r_local = train_ad_local(mps_l, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                             optimizer="lbfgs", stabilization="qr")

    assert r_global["final_energy"] >= E0 - 1e-6
    assert r_local["final_energy"] >= E0 - 1e-6
    assert abs(r_local["final_energy"] - r_global["final_energy"]) < 1e-2, \
        (r_local["final_energy"], r_global["final_energy"], E0)
