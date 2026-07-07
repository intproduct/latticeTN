"""Benchmark helpers for Stage 11 physics validation."""

from .exact_reference import (
    ExactResult,
    dense_model_hamiltonian,
    exact_ground_reference,
    hubbard_sector_indices,
    restrict_dense_hamiltonian,
    spinless_sector_indices,
)

__all__ = [
    "ExactResult",
    "dense_model_hamiltonian",
    "exact_ground_reference",
    "hubbard_sector_indices",
    "restrict_dense_hamiltonian",
    "spinless_sector_indices",
]
