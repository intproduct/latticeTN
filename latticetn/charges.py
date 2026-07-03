"""Charge metadata for fermionic local bases.

Stage 8 keeps dense MPS tensors, but records the local quantum numbers needed
for fixed-sector product states, diagnostics, and soft penalties.
"""

from __future__ import annotations

import torch as tc

from .fermion_operators import fermion_operators, hubbard_local_operators


def spinless_charge_metadata() -> dict[str, list[int]]:
    """Return charge labels for the spinless basis ``|0>, |1>``."""
    return {
        "n": [0, 1],
        "parity": [0, 1],
    }


def hubbard_charge_metadata() -> dict[str, list[int]]:
    """Return charge labels for ``|0>, |up>, |down>, |up,down>``."""
    n_up = [0, 1, 0, 1]
    n_down = [0, 0, 1, 1]
    n_tot = [u + d for u, d in zip(n_up, n_down)]
    return {
        "n_up": n_up,
        "n_down": n_down,
        "n_tot": n_tot,
        "sz2": [0, 1, -1, 0],
        "parity": [n % 2 for n in n_tot],
    }


def local_number_operator(model: str, dtype=tc.complex128, device="cpu") -> tc.Tensor:
    """Return the local total-number operator for a supported fermion model."""
    model = model.lower()
    if model in {"spinless", "spinless_tv", "fermion"}:
        return fermion_operators(dtype=dtype, device=device)["n"]
    if model == "hubbard":
        return hubbard_local_operators(dtype=dtype, device=device)["ntot"]
    raise ValueError(f"unknown fermion model {model!r}")


def local_nup_operator(dtype=tc.complex128, device="cpu") -> tc.Tensor:
    return hubbard_local_operators(dtype=dtype, device=device)["nup"]


def local_ndown_operator(dtype=tc.complex128, device="cpu") -> tc.Tensor:
    return hubbard_local_operators(dtype=dtype, device=device)["ndown"]


def local_ntot_operator(dtype=tc.complex128, device="cpu") -> tc.Tensor:
    return hubbard_local_operators(dtype=dtype, device=device)["ntot"]


def local_sz_operator(dtype=tc.complex128, device="cpu") -> tc.Tensor:
    return hubbard_local_operators(dtype=dtype, device=device)["sz"]


__all__ = [
    "spinless_charge_metadata",
    "hubbard_charge_metadata",
    "local_number_operator",
    "local_nup_operator",
    "local_ndown_operator",
    "local_ntot_operator",
    "local_sz_operator",
]
