"""Exact-diagonalization reference models for the latticeTN validation tests.

This is a helper module (not a test file). It exposes dense reference
Hamiltonians and exact ground-state energies, backed by `latticetn.operators`.

Conventions (see docs/PHYSICS_SPEC.md, docs/CLAUDE_PROGRESS.md):
- Spin convention: S = sigma / 2.
- Heisenberg: H = J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1}),
  J=1.0, open boundary.
- TFI: H = -J Sz Sz - h sum_i Sx_i, open boundary (spin convention).
"""

from __future__ import annotations

import numpy as np
import torch as tc

from latticetn.operators import (
    heisenberg_dense,
    tfi_dense,
    exact_ground_energy,
    spin_operators,
)

__all__ = [
    "heisenberg_dense",
    "tfi_dense",
    "exact_ground_energy",
    "spin_operators",
    "heisenberg_ground_energy",
    "bethe_ground_energy_per_site",
]


def heisenberg_ground_energy(N: int, J: float = 1.0) -> float:
    """Exact Heisenberg ground energy for an open spin-1/2 chain of length N.

    Uses dense ED (good for N <= ~12). For reference cross-checks.
    """
    H = heisenberg_dense(N, J=J)
    E0, _ = exact_ground_energy(H)
    return E0


def bethe_ground_energy_per_site() -> float:
    """Bethe-ansatz E0/N = 1/4 - ln(2) for the infinite antiferromagnetic chain."""
    return 0.25 - float(np.log(2.0))
