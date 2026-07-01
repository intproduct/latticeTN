"""Example: global AD-MPS on the 1D Heisenberg chain.

Builds an MPS + Heisenberg MPO, computes the differentiable native Rayleigh
energy, compares to exact diagonalization, then trains with the global AD-MPS
solver (Stage 4R). CPU-only, small N, fast.

Run:  python examples/heisenberg_ad_mps.py
"""

from __future__ import annotations

import torch as tc

from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.contractions import rayleigh_energy_native
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps
from latticetn.operators import heisenberg_dense, exact_ground_energy

DTYPE = tc.complex128


def main() -> None:
    N, chi = 6, 8
    tc.manual_seed(0)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)

    e_init = float(rayleigh_energy_native(mps, mpo).real)
    E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=DTYPE, device="cpu"))
    print(f"N={N}  exact E0 = {E0:.10f}")
    print(f"initial (random MPS) E = {e_init:.10f}")

    ad = ADVariationalMPS(mps, mpo)                       # all tensors trainable
    res = train_ad_mps(ad, num_steps=300, lr=1e-2, optimizer="adam",
                       projection="tensor_norm")
    print(f"global AD-MPS final E = {res['final_energy']:.10f}")
    print(f"abs err vs exact      = {abs(res['final_energy'] - E0):.2e}")
    assert res["final_energy"] >= E0 - 1e-6, "energy below ground!"


if __name__ == "__main__":
    main()
