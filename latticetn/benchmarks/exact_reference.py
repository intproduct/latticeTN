"""Independent small-system exact references for Stage 11 benchmarks.

These helpers are intentionally dense and exponential. They are for small-N
physics validation only and must not be used in large-N benchmark runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch as tc

from latticetn.operators import (
    exact_ground_energy,
    heisenberg_dense,
    hubbard_dense,
    spinless_fermion_dense,
    tfi_dense,
)


@dataclass(frozen=True)
class ExactResult:
    model: str
    N: int
    parameters: dict[str, float]
    sector: dict[str, int] | None
    energy: float
    energy_per_site: float
    dim_full: int
    dim_sector: int


def dense_model_hamiltonian(
    model: str,
    N: int,
    parameters: dict[str, float] | None = None,
    *,
    dtype: tc.dtype = tc.complex128,
    device: str = "cpu",
) -> tc.Tensor:
    """Build a dense small-N Hamiltonian from independent model references."""

    params = dict(parameters or {})
    if model == "heisenberg":
        return heisenberg_dense(N, J=float(params.get("J", 1.0)), dtype=dtype, device=device)
    if model == "tfi":
        return tfi_dense(
            N,
            J=float(params.get("J", 1.0)),
            h=float(params.get("h", 1.0)),
            dtype=dtype,
            device=device,
        )
    if model == "spinless_tv":
        return spinless_fermion_dense(
            N,
            t=float(params.get("t", 1.0)),
            V=float(params.get("V", 0.0)),
            mu=float(params.get("mu", 0.0)),
            dtype=dtype,
            device=device,
        )
    if model == "hubbard":
        return hubbard_dense(
            N,
            t=float(params.get("t", 1.0)),
            U=float(params.get("U", 4.0)),
            mu=float(params.get("mu", 0.0)),
            h=float(params.get("h", 0.0)),
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"unknown exact-reference model {model!r}")


def spinless_sector_indices(N: int, n_particles: int) -> list[int]:
    """Return computational-basis indices with exactly ``n_particles`` ones."""

    if not 0 <= n_particles <= N:
        raise ValueError("spinless particle sector must satisfy 0 <= n <= N")
    return [idx for idx in range(2**N) if int(idx).bit_count() == n_particles]


def _hubbard_site_counts(site_state: int) -> tuple[int, int]:
    if site_state == 0:
        return 0, 0
    if site_state == 1:
        return 1, 0
    if site_state == 2:
        return 0, 1
    if site_state == 3:
        return 1, 1
    raise ValueError(f"invalid Hubbard local state {site_state}")


def hubbard_sector_indices(N: int, nup: int, ndown: int) -> list[int]:
    """Return base-4 site-major indices in a fixed ``(N_up, N_down)`` sector."""

    if not 0 <= nup <= N or not 0 <= ndown <= N:
        raise ValueError("Hubbard sectors must satisfy 0 <= N_up,N_down <= N")
    out: list[int] = []
    for idx in range(4**N):
        x = idx
        up = 0
        down = 0
        for _ in range(N):
            site = x % 4
            x //= 4
            du, dd = _hubbard_site_counts(site)
            up += du
            down += dd
        if up == nup and down == ndown:
            out.append(idx)
    return out


def restrict_dense_hamiltonian(H: tc.Tensor, indices: list[int]) -> tc.Tensor:
    """Extract a dense Hamiltonian block on a sorted basis-index list."""

    if not indices:
        raise ValueError("sector index list is empty")
    idx = tc.tensor(indices, dtype=tc.long, device=H.device)
    return H.index_select(0, idx).index_select(1, idx)


def _sector_indices(model: str, N: int, sector: dict[str, Any] | None) -> list[int] | None:
    if sector is None:
        return None
    if model == "spinless_tv":
        target = sector.get("target_n")
        if target is None:
            return None
        return spinless_sector_indices(N, int(target))
    if model == "hubbard":
        nup = sector.get("target_nup")
        ndown = sector.get("target_ndown")
        if nup is None or ndown is None:
            return None
        return hubbard_sector_indices(N, int(nup), int(ndown))
    return None


def exact_ground_reference(
    model: str,
    N: int,
    parameters: dict[str, float] | None = None,
    sector: dict[str, int] | None = None,
    *,
    dtype: tc.dtype = tc.complex128,
    device: str = "cpu",
) -> ExactResult:
    """Return the exact small-N ground energy, optionally in a fixed sector."""

    H = dense_model_hamiltonian(model, N, parameters, dtype=dtype, device=device)
    H_eval = H
    indices = _sector_indices(model, N, sector)
    if indices is not None:
        H_eval = restrict_dense_hamiltonian(H, indices)
    energy, _ = exact_ground_energy(H_eval)
    return ExactResult(
        model=model,
        N=N,
        parameters={k: float(v) for k, v in dict(parameters or {}).items()},
        sector=dict(sector) if sector is not None else None,
        energy=energy,
        energy_per_site=energy / N,
        dim_full=int(H.shape[0]),
        dim_sector=int(H_eval.shape[0]),
    )
