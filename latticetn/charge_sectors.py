"""Charge-aware dense MPS helpers for hard fixed-sector constraints.

Stage 9 keeps ordinary dense PyTorch tensors, but attaches U(1) charge metadata
and boolean masks. Forbidden entries are zeroed after initialization, after
backward, and after optimizer steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch as tc

from .charges import spinless_charge_metadata, hubbard_charge_metadata
from .initial_states import _choose_spinless_sites, _choose_hubbard_sets
from .mps import MPS


SpinlessCharge = int
HubbardCharge = tuple[int, int]


@dataclass
class BondChargeSectors:
    """Allowed virtual charges on one MPS bond."""

    charges: list[SpinlessCharge] | list[HubbardCharge]
    dims: list[int] | None = None


@dataclass
class ChargeAwareMPS:
    """Lightweight metadata wrapper around the existing dense MPS class."""

    mps: MPS
    masks: list[tc.Tensor]
    bond_charges: list[BondChargeSectors]
    model: str
    target_sector: dict
    split_strategy: str = "dense_global_ad_masked"

    @property
    def tensors(self):
        return self.mps.tensors


def _check_N_target(N: int, target: int, name: str) -> None:
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if not (0 <= target <= N):
        raise ValueError(f"{name} must satisfy 0 <= {name} <= N, got {target}")


def _trim_charges(charges: list, chi: int | None, center) -> list:
    if chi is None or len(charges) <= chi:
        return charges
    if chi <= 0:
        raise ValueError(f"chi must be positive when provided, got {chi}")
    ranked = sorted(charges, key=lambda q: (center(q), q))
    return sorted(ranked[:chi])


def build_spinless_bond_sectors(
    N: int,
    target_n: int,
    chi: int | None = None,
    min_reachable: bool = True,
) -> list[BondChargeSectors]:
    """Build allowed cumulative particle numbers on all ``N+1`` MPS bonds."""

    _check_N_target(N, target_n, "target_n")
    sectors: list[BondChargeSectors] = []
    for i in range(N + 1):
        if min_reachable:
            lo = max(0, target_n - (N - i))
            hi = min(i, target_n)
        else:
            lo, hi = 0, target_n
        charges = list(range(lo, hi + 1))
        expected = target_n * i / N
        charges = _trim_charges(charges, chi, lambda q: abs(q - expected))
        sectors.append(BondChargeSectors(charges=charges))
    if sectors[0].charges != [0]:
        raise ValueError("left boundary spinless charge sector must be [0]")
    if sectors[-1].charges != [target_n]:
        raise ValueError("right boundary spinless charge sector must be [target_n]")
    return sectors


def build_hubbard_bond_sectors(
    N: int,
    target_nup: int,
    target_ndown: int,
    chi: int | None = None,
    min_reachable: bool = True,
) -> list[BondChargeSectors]:
    """Build allowed cumulative ``(N_up, N_down)`` charges on all bonds."""

    _check_N_target(N, target_nup, "target_nup")
    _check_N_target(N, target_ndown, "target_ndown")
    sectors: list[BondChargeSectors] = []
    for i in range(N + 1):
        if min_reachable:
            up_lo = max(0, target_nup - (N - i))
            up_hi = min(i, target_nup)
            dn_lo = max(0, target_ndown - (N - i))
            dn_hi = min(i, target_ndown)
        else:
            up_lo, up_hi = 0, target_nup
            dn_lo, dn_hi = 0, target_ndown
        charges = [(u, d) for u in range(up_lo, up_hi + 1) for d in range(dn_lo, dn_hi + 1)]
        exp_u = target_nup * i / N
        exp_d = target_ndown * i / N
        charges = _trim_charges(
            charges,
            chi,
            lambda q: abs(q[0] - exp_u) + abs(q[1] - exp_d),
        )
        sectors.append(BondChargeSectors(charges=charges))
    if sectors[0].charges != [(0, 0)]:
        raise ValueError("left boundary Hubbard charge sector must be [(0, 0)]")
    if sectors[-1].charges != [(target_nup, target_ndown)]:
        raise ValueError("right boundary Hubbard charge sector must equal target sector")
    return sectors


def spinless_tensor_charge_mask(
    left_charges: Sequence[int],
    right_charges: Sequence[int],
    local_charges: Sequence[int] | None = None,
    device=None,
    dtype=tc.bool,
) -> tc.Tensor:
    """Mask for ``Q_right = Q_left + n[s]``."""

    if local_charges is None:
        local_charges = spinless_charge_metadata()["n"]
    mask = tc.zeros((len(left_charges), len(local_charges), len(right_charges)), dtype=tc.bool, device=device)
    right_lookup = {q: b for b, q in enumerate(right_charges)}
    for a, ql in enumerate(left_charges):
        for s, qs in enumerate(local_charges):
            b = right_lookup.get(ql + qs)
            if b is not None:
                mask[a, s, b] = True
    return mask.to(dtype=dtype)


def hubbard_tensor_charge_mask(
    left_charges: Sequence[tuple[int, int]],
    right_charges: Sequence[tuple[int, int]],
    local_charges: Sequence[tuple[int, int]] | None = None,
    device=None,
    dtype=tc.bool,
) -> tc.Tensor:
    """Mask for ``Q_right = Q_left + (n_up[s], n_down[s])``."""

    if local_charges is None:
        meta = hubbard_charge_metadata()
        local_charges = list(zip(meta["n_up"], meta["n_down"]))
    mask = tc.zeros((len(left_charges), len(local_charges), len(right_charges)), dtype=tc.bool, device=device)
    right_lookup = {q: b for b, q in enumerate(right_charges)}
    for a, (ul, dl) in enumerate(left_charges):
        for s, (us, ds) in enumerate(local_charges):
            b = right_lookup.get((ul + us, dl + ds))
            if b is not None:
                mask[a, s, b] = True
    return mask.to(dtype=dtype)


def build_spinless_masks(
    bond_charges: list[BondChargeSectors],
    device=None,
) -> list[tc.Tensor]:
    local = spinless_charge_metadata()["n"]
    return [
        spinless_tensor_charge_mask(bond_charges[i].charges, bond_charges[i + 1].charges, local, device=device)
        for i in range(len(bond_charges) - 1)
    ]


def build_hubbard_masks(
    bond_charges: list[BondChargeSectors],
    device=None,
) -> list[tc.Tensor]:
    meta = hubbard_charge_metadata()
    local = list(zip(meta["n_up"], meta["n_down"]))
    return [
        hubbard_tensor_charge_mask(bond_charges[i].charges, bond_charges[i + 1].charges, local, device=device)
        for i in range(len(bond_charges) - 1)
    ]


def apply_charge_masks_(mps_or_tensors, masks: Sequence[tc.Tensor]) -> None:
    """In-place zeroing of forbidden tensor entries."""

    tensors = getattr(mps_or_tensors, "tensors", mps_or_tensors)
    with tc.no_grad():
        for tensor, mask in zip(tensors, masks):
            tensor.mul_(mask.to(dtype=tensor.dtype, device=tensor.device))


def zero_forbidden_gradients_(parameters_or_tensors, masks: Sequence[tc.Tensor]) -> float:
    """Zero forbidden gradient entries and return their pre-zero max magnitude."""

    tensors = list(parameters_or_tensors)
    max_abs = 0.0
    for tensor, mask in zip(tensors, masks):
        if tensor.grad is None:
            continue
        forbidden = ~mask.to(device=tensor.grad.device)
        if forbidden.any():
            vals = tensor.grad.detach()[forbidden]
            if vals.numel() > 0:
                max_abs = max(max_abs, float(vals.abs().max().detach().cpu()))
            tensor.grad.data.mul_(mask.to(dtype=tensor.grad.dtype, device=tensor.grad.device))
    return max_abs


def max_forbidden_abs(mps_or_tensors, masks: Sequence[tc.Tensor]) -> float:
    """Return the largest forbidden entry magnitude."""

    tensors = getattr(mps_or_tensors, "tensors", mps_or_tensors)
    max_abs = 0.0
    for tensor, mask in zip(tensors, masks):
        forbidden = ~mask.to(device=tensor.device)
        if forbidden.any():
            vals = tensor.detach()[forbidden]
            if vals.numel() > 0:
                max_abs = max(max_abs, float(vals.abs().max().cpu()))
    return max_abs


def _charge_index(charges: Sequence, charge) -> int:
    try:
        return list(charges).index(charge)
    except ValueError as exc:
        raise ValueError(f"charge {charge!r} is not present in bond sectors {charges!r}") from exc


def _make_masked_random_tensors(
    masks: Sequence[tc.Tensor],
    dtype,
    device,
    scale: float = 1e-3,
) -> list[tc.Tensor]:
    tensors = []
    for mask in masks:
        if dtype.is_complex:
            real = tc.randn(mask.shape, dtype=tc.float64 if dtype == tc.complex128 else tc.float32, device=device)
            imag = tc.randn(mask.shape, dtype=real.dtype, device=device)
            t = (real + 1j * imag).to(dtype) * scale
        else:
            t = tc.randn(mask.shape, dtype=dtype, device=device) * scale
        tensors.append(t * mask.to(dtype=dtype, device=device))
    return tensors


def spinless_hard_sector_product_mps(
    N: int,
    target_n: int,
    chi: int | None,
    pattern: str = "cdw",
    device=None,
    dtype=None,
) -> ChargeAwareMPS:
    """Create a masked spinless MPS in the exact target particle sector."""

    device = "cpu" if device is None else device
    dtype = tc.complex128 if dtype is None else dtype
    bonds = build_spinless_bond_sectors(N, target_n, chi=chi)
    masks = build_spinless_masks(bonds, device=device)
    occupied = _choose_spinless_sites(N, target_n, pattern)
    tensors = _make_masked_random_tensors(masks, dtype=dtype, device=device)
    q = 0
    for i, tensor in enumerate(tensors):
        s = 1 if i in occupied else 0
        a = _charge_index(bonds[i].charges, q)
        q_next = q + s
        b = _charge_index(bonds[i + 1].charges, q_next)
        tensor[a, s, b] = tensor[a, s, b] + 1.0
        q = q_next
    mps = MPS.from_tensors(tensors, dtype=dtype, device=device, requires_grad=False)
    apply_charge_masks_(mps, masks)
    return ChargeAwareMPS(
        mps=mps,
        masks=masks,
        bond_charges=bonds,
        model="spinless_tv",
        target_sector={"n": target_n},
    )


def hubbard_hard_sector_product_mps(
    N: int,
    target_nup: int,
    target_ndown: int,
    chi: int | None,
    pattern: str = "neel",
    device=None,
    dtype=None,
) -> ChargeAwareMPS:
    """Create a masked Hubbard MPS in the exact ``(N_up, N_down)`` sector."""

    device = "cpu" if device is None else device
    dtype = tc.complex128 if dtype is None else dtype
    bonds = build_hubbard_bond_sectors(N, target_nup, target_ndown, chi=chi)
    masks = build_hubbard_masks(bonds, device=device)
    up_sites, down_sites = _choose_hubbard_sets(N, target_nup, target_ndown, pattern)
    tensors = _make_masked_random_tensors(masks, dtype=dtype, device=device)
    q = (0, 0)
    for i, tensor in enumerate(tensors):
        has_up = i in up_sites
        has_down = i in down_sites
        if has_up and has_down:
            s, dq = 3, (1, 1)
        elif has_up:
            s, dq = 1, (1, 0)
        elif has_down:
            s, dq = 2, (0, 1)
        else:
            s, dq = 0, (0, 0)
        a = _charge_index(bonds[i].charges, q)
        q_next = (q[0] + dq[0], q[1] + dq[1])
        b = _charge_index(bonds[i + 1].charges, q_next)
        tensor[a, s, b] = tensor[a, s, b] + 1.0
        q = q_next
    mps = MPS.from_tensors(tensors, dtype=dtype, device=device, requires_grad=False)
    apply_charge_masks_(mps, masks)
    return ChargeAwareMPS(
        mps=mps,
        masks=masks,
        bond_charges=bonds,
        model="hubbard",
        target_sector={"n_up": target_nup, "n_down": target_ndown},
    )


__all__ = [
    "BondChargeSectors",
    "ChargeAwareMPS",
    "build_spinless_bond_sectors",
    "build_hubbard_bond_sectors",
    "spinless_tensor_charge_mask",
    "hubbard_tensor_charge_mask",
    "build_spinless_masks",
    "build_hubbard_masks",
    "apply_charge_masks_",
    "zero_forbidden_gradients_",
    "max_forbidden_abs",
    "spinless_hard_sector_product_mps",
    "hubbard_hard_sector_product_mps",
]
