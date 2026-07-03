"""Fixed-sector product-state initializers for dense MPS.

All helpers return open-boundary product MPS objects with bond dimension one.
They validate the requested sector instead of silently producing a mismatched
state.
"""

from __future__ import annotations

import torch as tc

from .mps import MPS
from .charges import hubbard_charge_metadata


def _validate_N(N: int) -> None:
    if int(N) != N or N <= 0:
        raise ValueError(f"N must be a positive integer, got {N!r}")


def _product_mps(indices: list[int], dim: int, dtype, device) -> MPS:
    tensors = []
    for phys in indices:
        if not (0 <= phys < dim):
            raise ValueError(f"physical index {phys} out of range for dim={dim}")
        A = tc.zeros((1, dim, 1), dtype=dtype, device=device)
        A[0, phys, 0] = 1.0
        tensors.append(A)
    return MPS.from_tensors(tensors, dtype=dtype, device=device, requires_grad=False)


def neel_spin_state(N: int, dtype=tc.complex128, device="cpu") -> MPS:
    """Spin-1/2 Neel product state in the package spin basis, ``0,1,0,1,...``."""
    _validate_N(N)
    return _product_mps([0 if i % 2 == 0 else 1 for i in range(N)], 2, dtype, device)


def spinless_half_filled_cdw_state(N: int, dtype=tc.complex128, device="cpu") -> MPS:
    """Half-filled spinless CDW product state ``1,0,1,0,...``."""
    if N % 2 != 0:
        raise ValueError("half-filled CDW requires even N")
    return spinless_fixed_number_state(N, N // 2, pattern="cdw", dtype=dtype, device=device)


def _choose_spinless_sites(N: int, n_particles: int, pattern: str) -> set[int]:
    if not (0 <= n_particles <= N):
        raise ValueError(f"n_particles must satisfy 0 <= n_particles <= N, got {n_particles}")
    if pattern == "left":
        order = list(range(N))
    elif pattern == "cdw":
        order = list(range(0, N, 2)) + list(range(1, N, 2))
    elif pattern == "centered":
        center = (N - 1) / 2
        order = sorted(range(N), key=lambda i: (abs(i - center), i))
    else:
        raise ValueError("pattern must be one of 'left', 'cdw', or 'centered'")
    return set(order[:n_particles])


def spinless_fixed_number_state(
    N: int,
    n_particles: int,
    pattern: str = "left",
    dtype=tc.complex128,
    device="cpu",
) -> MPS:
    """Spinless fixed-particle-number product state in basis ``|0>, |1>``."""
    _validate_N(N)
    occupied = _choose_spinless_sites(N, n_particles, pattern)
    return _product_mps([1 if i in occupied else 0 for i in range(N)], 2, dtype, device)


def hubbard_half_filled_neel_state(N: int, dtype=tc.complex128, device="cpu") -> MPS:
    """Hubbard half-filled Neel state ``|up>, |down>, |up>, |down>, ...``."""
    if N % 2 != 0:
        raise ValueError("half-filled Hubbard Neel state requires even N")
    return hubbard_fixed_sector_state(
        N, N // 2, N // 2, pattern="neel", dtype=dtype, device=device
    )


def _ordered_sites(N: int, preferred_parity: int) -> list[int]:
    return list(range(preferred_parity, N, 2)) + list(range(1 - preferred_parity, N, 2))


def _choose_hubbard_sets(N: int, n_up: int, n_down: int, pattern: str) -> tuple[set[int], set[int]]:
    if not (0 <= n_up <= N and 0 <= n_down <= N):
        raise ValueError(
            f"n_up and n_down must each satisfy 0 <= n <= N, got {n_up}, {n_down}"
        )
    if pattern == "left":
        up_sites = set(range(n_up))
        down_sites = set(range(n_down))
    elif pattern in {"neel", "balanced"}:
        up_order = _ordered_sites(N, 0)
        down_order = _ordered_sites(N, 1)
        if pattern == "balanced" and n_up + n_down > N:
            # Put the unavoidable doublons near the center instead of the edge.
            center_order = sorted(range(N), key=lambda i: (abs(i - (N - 1) / 2), i))
            shared = set(center_order[: n_up + n_down - N])
            up_rest = [i for i in up_order if i not in shared]
            down_rest = [i for i in down_order if i not in shared]
            up_sites = set(up_rest[: n_up - len(shared)]) | shared
            down_sites = set(down_rest[: n_down - len(shared)]) | shared
        else:
            up_sites = set(up_order[:n_up])
            down_sites = set(down_order[:n_down])
    else:
        raise ValueError("pattern must be one of 'neel', 'left', or 'balanced'")
    if len(up_sites) != n_up or len(down_sites) != n_down:
        raise ValueError("could not construct the requested Hubbard sector")
    return up_sites, down_sites


def hubbard_fixed_sector_state(
    N: int,
    n_up: int,
    n_down: int,
    pattern: str = "neel",
    dtype=tc.complex128,
    device="cpu",
) -> MPS:
    """Hubbard fixed-sector product state.

    Basis convention: ``0=|0>``, ``1=|up>``, ``2=|down>``, ``3=|up,down>``.
    """
    _validate_N(N)
    up_sites, down_sites = _choose_hubbard_sets(N, n_up, n_down, pattern)
    indices = []
    for i in range(N):
        has_up = i in up_sites
        has_down = i in down_sites
        if has_up and has_down:
            indices.append(3)
        elif has_up:
            indices.append(1)
        elif has_down:
            indices.append(2)
        else:
            indices.append(0)

    meta = hubbard_charge_metadata()
    got_up = sum(meta["n_up"][x] for x in indices)
    got_down = sum(meta["n_down"][x] for x in indices)
    if got_up != n_up or got_down != n_down:
        raise ValueError(
            f"constructed Hubbard state has sector ({got_up}, {got_down}), "
            f"expected ({n_up}, {n_down})"
        )
    return _product_mps(indices, 4, dtype, device)


__all__ = [
    "neel_spin_state",
    "spinless_half_filled_cdw_state",
    "spinless_fixed_number_state",
    "hubbard_half_filled_neel_state",
    "hubbard_fixed_sector_state",
]
