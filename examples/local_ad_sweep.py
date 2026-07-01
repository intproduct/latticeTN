"""Example: AD local-tensor optimization on the 1D Heisenberg chain.

Trains the same system as heisenberg_ad_mps.py but with AD local-tensor
optimization (Stage 5A): one center tensor at a time, swept by QR, with
optional post-step QR stabilization. CPU-only, small N, fast.

Run:  python examples/local_ad_sweep.py
"""

from __future__ import annotations

import torch as tc

from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_local import train_ad_local
from latticetn.operators import heisenberg_dense, exact_ground_energy

DTYPE = tc.complex128


def main() -> None:
    N, chi = 6, 8
    tc.manual_seed(0)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)

    E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=DTYPE, device="cpu"))
    print(f"N={N}  exact E0 = {E0:.10f}")

    # The optimizer is loss.backward() + LBFGS on one center tensor at a time.
    # `stabilization="qr"` is OPTIONAL post-step gauge projection (NOT the solver).
    res = train_ad_local(mps, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                         optimizer="lbfgs", stabilization="qr")
    print(f"AD local-tensor final E = {res['final_energy']:.10f}")
    print(f"abs err vs exact        = {abs(res['final_energy'] - E0):.2e}")
    print(f"stabilization           = {res['stabilization']}")
    assert res["final_energy"] >= E0 - 1e-6, "energy below ground!"


if __name__ == "__main__":
    main()
