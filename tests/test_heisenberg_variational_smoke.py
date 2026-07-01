"""Stage 7: short variational Heisenberg solve smoke test (CPU, complex128).

Imports the solver from scripts/run_heisenberg_small.py and asserts:
- the energy decreases from a random initial state,
- the final energy does not fall below the exact ground energy (variational
  principle), within a tiny tolerance (would indicate a convention bug),
- for small exact-representable systems the final energy is close to E0.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch as tc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_heisenberg_small as rhs  # noqa: E402

DTYPE = tc.complex128


def test_smoke_energy_decreases_and_stays_above_ground_N4():
    res = rhs.solve(N=4, chi=4, steps=150, lr=1e-2, seed=0, device="cpu")
    assert res["final_E"] < res["initial_E"]                # energy decreased
    assert not res["below_ground"], res                     # not below E0
    # chi=4 is exact-representable for N=4 -> should be very close to E0
    assert res["abs_err"] < 1e-3, res


def test_smoke_N6_chi8_approaches_ground():
    res = rhs.solve(N=6, chi=8, steps=300, lr=1e-2, seed=0, device="cpu")
    # chi=8 = 2^(N/2) is exact-representable for N=6 open chain.
    assert res["final_E"] < res["initial_E"]
    assert not res["below_ground"], res
    assert res["abs_err"] < 1e-2, res


def test_smoke_reproducible_with_seed():
    a = rhs.solve(N=4, chi=4, steps=50, lr=1e-2, seed=1, device="cpu")
    b = rhs.solve(N=4, chi=4, steps=50, lr=1e-2, seed=1, device="cpu")
    assert abs(a["final_E"] - b["final_E"]) < 1e-12


def test_smoke_variational_principle_holds_many_seeds():
    # For every seed, final energy must stay >= E0 (within tolerance).
    for seed in range(5):
        res = rhs.solve(N=4, chi=4, steps=100, lr=1e-2, seed=seed, device="cpu")
        assert res["final_E"] >= res["exact_E0"] - 1e-6, (seed, res)
