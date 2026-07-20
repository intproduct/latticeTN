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
    spinless_fermion_dense,
    hubbard_dense,
    exact_ground_energy,
)
from .fermion_operators import fermion_operators, hubbard_local_operators
from . import model_builder, model_spec, model_registry, hamiltonian_builder, config_schema, runner
from . import charges, charge_sectors, initial_states, sector_observables, tdvp
from .mpo import MPO
from .mps import MPS

__all__ = [
    "spin_operators",
    "pauli_matrices",
    "heisenberg_dense",
    "tfi_dense",
    "spinless_fermion_dense",
    "hubbard_dense",
    "fermion_operators",
    "hubbard_local_operators",
    "exact_ground_energy",
    "model_builder",
    "model_spec",
    "model_registry",
    "hamiltonian_builder",
    "config_schema",
    "runner",
    "charges",
    "charge_sectors",
    "initial_states",
    "sector_observables",
    "tdvp",
    "MPO",
    "MPS",
]
