"""latticetn core package: autograd tensor-network building blocks.

Conventions (see docs/PHYSICS_SPEC.md and docs/CLAUDE_PROGRESS.md):

- Spin convention: S = sigma / 2 (NOT Pauli).
- MPS tensor index order: (left_bond, phys, right_bond).
- MPO tensor index order: (left_bond, right_bond, phys_in, phys_out).
  phys_in contracts with the ket, phys_out with the bra.
- Open boundary conditions by default.
- dtype torch.complex128, device cpu by default.
"""

from .operators import (
    spin_operators,
    pauli_matrices,
    heisenberg_dense,
    tfi_dense,
    exact_ground_energy,
)
from .mpo import MPO
from .mps import MPS

__all__ = [
    "spin_operators",
    "pauli_matrices",
    "heisenberg_dense",
    "tfi_dense",
    "exact_ground_energy",
    "MPO",
    "MPS",
]
