"""Charge-aware dense MPS helpers for hard fixed-sector constraints.

Stage 9 keeps ordinary dense PyTorch tensors, but attaches U(1) charge metadata
and boolean masks. Forbidden entries are zeroed after initialization, after
backward, and after optimizer steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Sequence

import torch as tc

from .charges import spinless_charge_metadata, hubbard_charge_metadata
from .initial_states import _choose_spinless_sites, _choose_hubbard_sets
from .mps import MPS
from .canonical import canonical_residual


SpinlessCharge = int
HubbardCharge = tuple[int, int]


@dataclass
class BondChargeSectors:
    """Allowed virtual charges and degeneracies on one MPS bond.

    ``charges[k]`` labels a symmetry sector and ``dims[k]`` is the number of
    independent Schmidt/degeneracy channels in that sector. The dense tensor
    index uses :attr:`expanded_charges`.
    """

    charges: list[SpinlessCharge] | list[HubbardCharge]
    dims: list[int] | None = None

    def __post_init__(self) -> None:
        if self.dims is None:
            self.dims = [1] * len(self.charges)
        if len(self.dims) != len(self.charges):
            raise ValueError("BondChargeSectors.dims must align with charges")
        if any(int(dim) <= 0 for dim in self.dims):
            raise ValueError("all charge-sector degeneracies must be positive")

    @property
    def expanded_charges(self) -> list:
        return [
            charge
            for charge, dim in zip(self.charges, self.dims)
            for _ in range(int(dim))
        ]

    @property
    def bond_dim(self) -> int:
        return sum(int(dim) for dim in self.dims)


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


def _allocate_channels(
    charges: list,
    capacities: dict,
    chi: int | None,
    center,
    required,
) -> BondChargeSectors:
    """Allocate a nested chi-prefix of ``(charge, degeneracy)`` channels."""

    if chi is not None and chi <= 0:
        raise ValueError(f"chi must be positive when provided, got {chi}")
    if required not in capacities:
        raise ValueError(f"required charge {required!r} is not reachable")
    if chi is None:
        return BondChargeSectors(charges=list(charges), dims=[1] * len(charges))

    ordered = sorted(charges, key=lambda q: (center(q), q))
    tokens = [required]
    for alpha in range(1, chi + 1):
        for charge in ordered:
            if alpha == 1 and charge == required:
                continue
            if capacities[charge] >= alpha:
                tokens.append(charge)
                if len(tokens) == chi:
                    break
        if len(tokens) == chi:
            break
    counts = {charge: tokens.count(charge) for charge in set(tokens)}
    kept = sorted(counts)
    return BondChargeSectors(
        charges=kept,
        dims=[counts[charge] for charge in kept],
    )


def _clip_to_bidirectional_graph(
    sectors: list[BondChargeSectors],
    local_charges: Sequence,
) -> list[BondChargeSectors]:
    """Clip channel multiplicities until every channel is reachable both ways."""

    dims = [
        {charge: int(dim) for charge, dim in zip(bond.charges, bond.dims)}
        for bond in sectors
    ]
    changed = True
    while changed:
        changed = False
        for i in range(1, len(dims)):
            left = dims[i - 1]
            for charge in list(dims[i]):
                capacity = sum(
                    multiplicity
                    for q_left, multiplicity in left.items()
                    for q_local in local_charges
                    if _charge_add(q_left, q_local) == charge
                )
                new_dim = min(dims[i][charge], capacity)
                if new_dim != dims[i][charge]:
                    dims[i][charge] = new_dim
                    changed = True
        for i in range(len(dims) - 2, -1, -1):
            right = dims[i + 1]
            for charge in list(dims[i]):
                capacity = sum(
                    multiplicity
                    for q_right, multiplicity in right.items()
                    for q_local in local_charges
                    if _charge_add(charge, q_local) == q_right
                )
                new_dim = min(dims[i][charge], capacity)
                if new_dim != dims[i][charge]:
                    dims[i][charge] = new_dim
                    changed = True
    clipped = []
    for bond_dims in dims:
        charges = sorted(q for q, dim in bond_dims.items() if dim > 0)
        clipped.append(BondChargeSectors(
            charges=charges,
            dims=[bond_dims[q] for q in charges],
        ))
    if any(not bond.charges for bond in clipped):
        raise ValueError("chi allocation disconnected the fixed-sector charge graph")
    return clipped


def build_spinless_bond_sectors(
    N: int,
    target_n: int,
    chi: int | None = None,
    min_reachable: bool = True,
    _required_path: Sequence[int] | None = None,
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
        capacities = {
            q: min(comb(i, q), comb(N - i, target_n - q))
            for q in charges
        }
        required = (
            _required_path[i]
            if _required_path is not None
            else min(charges, key=lambda q: (abs(q - expected), q))
        )
        sectors.append(_allocate_channels(
            charges, capacities, chi,
            lambda q: abs(q - expected),
            required,
        ))
    sectors = _clip_to_bidirectional_graph(sectors, [0, 1])
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
    _required_path: Sequence[tuple[int, int]] | None = None,
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
        capacities = {
            (u, d): min(
                comb(i, u) * comb(i, d),
                comb(N - i, target_nup - u) * comb(N - i, target_ndown - d),
            )
            for u, d in charges
        }
        distance = lambda q: abs(q[0] - exp_u) + abs(q[1] - exp_d)
        required = (
            _required_path[i]
            if _required_path is not None
            else min(charges, key=lambda q: (distance(q), q))
        )
        sectors.append(_allocate_channels(
            charges, capacities, chi, distance, required,
        )
        )
    meta = hubbard_charge_metadata()
    local = list(zip(meta["n_up"], meta["n_down"]))
    sectors = _clip_to_bidirectional_graph(sectors, local)
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
    right_lookup = {}
    for b, q in enumerate(right_charges):
        right_lookup.setdefault(q, []).append(b)
    for a, ql in enumerate(left_charges):
        for s, qs in enumerate(local_charges):
            for b in right_lookup.get(ql + qs, []):
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
    right_lookup = {}
    for b, q in enumerate(right_charges):
        right_lookup.setdefault(q, []).append(b)
    for a, (ul, dl) in enumerate(left_charges):
        for s, (us, ds) in enumerate(local_charges):
            for b in right_lookup.get((ul + us, dl + ds), []):
                mask[a, s, b] = True
    return mask.to(dtype=dtype)


def build_spinless_masks(
    bond_charges: list[BondChargeSectors],
    device=None,
) -> list[tc.Tensor]:
    local = spinless_charge_metadata()["n"]
    return [
        spinless_tensor_charge_mask(
            bond_charges[i].expanded_charges,
            bond_charges[i + 1].expanded_charges,
            local,
            device=device,
        )
        for i in range(len(bond_charges) - 1)
    ]


def build_hubbard_masks(
    bond_charges: list[BondChargeSectors],
    device=None,
) -> list[tc.Tensor]:
    meta = hubbard_charge_metadata()
    local = list(zip(meta["n_up"], meta["n_down"]))
    return [
        hubbard_tensor_charge_mask(
            bond_charges[i].expanded_charges,
            bond_charges[i + 1].expanded_charges,
            local,
            device=device,
        )
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


def _local_charges_for(camps: ChargeAwareMPS):
    if camps.model == "spinless_tv":
        return list(spinless_charge_metadata()["n"])
    if camps.model == "hubbard":
        meta = hubbard_charge_metadata()
        return list(zip(meta["n_up"], meta["n_down"]))
    raise ValueError(f"unsupported charge-aware model {camps.model!r}")


def _charge_add(left, local):
    if isinstance(left, tuple):
        return tuple(a + b for a, b in zip(left, local))
    return left + local


def _copy_charge_aware(camps: ChargeAwareMPS, tensors: Sequence[tc.Tensor]) -> ChargeAwareMPS:
    mps = MPS.from_tensors(tensors, dtype=camps.mps.dtype, device=camps.mps.device,
                           requires_grad=False)
    out = ChargeAwareMPS(
        mps=mps,
        masks=[mask.clone() for mask in camps.masks],
        bond_charges=camps.bond_charges,
        model=camps.model,
        target_sector=dict(camps.target_sector),
        split_strategy=camps.split_strategy,
    )
    apply_charge_masks_(out.mps, out.masks)
    return out


def sector_left_canonicalize(camps: ChargeAwareMPS) -> ChargeAwareMPS:
    """Exact left QR sweep that never mixes distinct virtual charges."""
    tensors = [t.detach().clone() for t in camps.tensors]
    local_charges = _local_charges_for(camps)
    with tc.no_grad():
        for i in range(len(tensors) - 1):
            l, d, r = tensors[i].shape
            mat = tensors[i].reshape(l * d, r)
            qleft = camps.bond_charges[i].expanded_charges
            qright = camps.bond_charges[i + 1].expanded_charges
            qmat = tc.zeros_like(mat)
            residual = tc.zeros((r, r), dtype=mat.dtype, device=mat.device)
            for charge in dict.fromkeys(qright):
                cols = [b for b, q in enumerate(qright) if q == charge]
                rows = [a * d + s for a, ql in enumerate(qleft)
                        for s, qs in enumerate(local_charges)
                        if _charge_add(ql, qs) == charge]
                if not cols:
                    continue
                block = mat[rows][:, cols]
                q, rr = tc.linalg.qr(block, mode="reduced")
                if q.shape[1] != len(cols):
                    raise ValueError(
                        f"charge block {charge!r} is rank-shape deficient: "
                        f"{len(rows)} rows for {len(cols)} virtual channels")
                qmat[tc.tensor(rows, device=mat.device)[:, None],
                     tc.tensor(cols, device=mat.device)[None, :]] = q
                residual[tc.tensor(cols, device=mat.device)[:, None],
                         tc.tensor(cols, device=mat.device)[None, :]] = rr
            tensors[i] = qmat.reshape(l, d, r)
            tensors[i + 1] = tc.einsum("ab,bsd->asd", residual, tensors[i + 1])
    return _copy_charge_aware(camps, tensors)


def sector_mixed_canonicalize(camps: ChargeAwareMPS, center: int) -> ChargeAwareMPS:
    """Exact charge-block mixed-canonical QR retraction."""
    if not (0 <= center < camps.mps.N):
        raise ValueError(f"center {center} out of range for N={camps.mps.N}")
    tensors = [t.detach().clone() for t in camps.tensors]
    local_charges = _local_charges_for(camps)
    # Reuse the left block algorithm on the requested prefix.
    prefix = _copy_charge_aware(camps, tensors)
    if center:
        # Inline a bounded left sweep so sites to the right of center remain available.
        with tc.no_grad():
            for i in range(center):
                l, d, r = tensors[i].shape
                mat = tensors[i].reshape(l * d, r)
                qleft = camps.bond_charges[i].expanded_charges
                qright = camps.bond_charges[i + 1].expanded_charges
                qmat = tc.zeros_like(mat)
                residual = tc.zeros((r, r), dtype=mat.dtype, device=mat.device)
                for charge in dict.fromkeys(qright):
                    cols = [b for b, q in enumerate(qright) if q == charge]
                    rows = [a * d + s for a, ql in enumerate(qleft)
                            for s, qs in enumerate(local_charges)
                            if _charge_add(ql, qs) == charge]
                    block = mat[rows][:, cols]
                    q, rr = tc.linalg.qr(block, mode="reduced")
                    if q.shape[1] != len(cols):
                        raise ValueError(f"charge block {charge!r} is rank-shape deficient")
                    qmat[tc.tensor(rows, device=mat.device)[:, None], tc.tensor(cols, device=mat.device)[None, :]] = q
                    residual[tc.tensor(cols, device=mat.device)[:, None], tc.tensor(cols, device=mat.device)[None, :]] = rr
                tensors[i] = qmat.reshape(l, d, r)
                tensors[i + 1] = tc.einsum("ab,bsd->asd", residual, tensors[i + 1])
    with tc.no_grad():
        for i in range(len(tensors) - 1, center, -1):
            l, d, r = tensors[i].shape
            mat = tensors[i].reshape(l, d * r)
            qleft = camps.bond_charges[i].expanded_charges
            qright = camps.bond_charges[i + 1].expanded_charges
            bmat = tc.zeros_like(mat)
            residual = tc.zeros((l, l), dtype=mat.dtype, device=mat.device)
            for charge in dict.fromkeys(qleft):
                rows = [a for a, q in enumerate(qleft) if q == charge]
                cols = [s * r + b for s, qs in enumerate(local_charges)
                        for b, qr in enumerate(qright)
                        if _charge_add(charge, qs) == qr]
                block = mat[rows][:, cols]
                qt, rr = tc.linalg.qr(block.t(), mode="reduced")
                if qt.shape[1] != len(rows):
                    raise ValueError(f"charge block {charge!r} is rank-shape deficient")
                b = qt.t()
                c = rr.t()
                bmat[tc.tensor(rows, device=mat.device)[:, None], tc.tensor(cols, device=mat.device)[None, :]] = b
                residual[tc.tensor(rows, device=mat.device)[:, None], tc.tensor(rows, device=mat.device)[None, :]] = c
            tensors[i] = bmat.reshape(l, d, r)
            tensors[i - 1] = tc.einsum("asd,db->asb", tensors[i - 1], residual)
    return _copy_charge_aware(prefix, tensors)


def sector_normalize_center(camps: ChargeAwareMPS, center: int | None = None) -> ChargeAwareMPS:
    """Charge-preserving mixed canonicalization followed by center normalization."""
    if center is None:
        center = camps.mps.N - 1
    out = sector_mixed_canonicalize(camps, center)
    with tc.no_grad():
        norm = out.tensors[center].norm()
        if not bool(tc.isfinite(norm)) or float(norm) == 0.0:
            raise ValueError(f"cannot normalize charge-sector center with norm {float(norm)!r}")
        out.tensors[center].div_(norm)
    return out


def sector_canonical_residual(camps: ChargeAwareMPS, center: int | None = None) -> float:
    """Mixed-canonical residual for a charge-aware MPS."""
    return canonical_residual(camps.mps, center=center)


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
    occupied = _choose_spinless_sites(N, target_n, pattern)
    required = [0]
    q_req = 0
    for i in range(N):
        q_req += 1 if i in occupied else 0
        required.append(q_req)
    bonds = build_spinless_bond_sectors(
        N, target_n, chi=chi, _required_path=required
    )
    masks = build_spinless_masks(bonds, device=device)
    tensors = _make_masked_random_tensors(masks, dtype=dtype, device=device)
    q = 0
    for i, tensor in enumerate(tensors):
        s = 1 if i in occupied else 0
        a = _charge_index(bonds[i].expanded_charges, q)
        q_next = q + s
        b = _charge_index(bonds[i + 1].expanded_charges, q_next)
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
    up_sites, down_sites = _choose_hubbard_sets(N, target_nup, target_ndown, pattern)
    required = [(0, 0)]
    q_req = (0, 0)
    for i in range(N):
        dq = (1 if i in up_sites else 0, 1 if i in down_sites else 0)
        q_req = (q_req[0] + dq[0], q_req[1] + dq[1])
        required.append(q_req)
    bonds = build_hubbard_bond_sectors(
        N,
        target_nup,
        target_ndown,
        chi=chi,
        _required_path=required,
    )
    masks = build_hubbard_masks(bonds, device=device)
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
        a = _charge_index(bonds[i].expanded_charges, q)
        q_next = (q[0] + dq[0], q[1] + dq[1])
        b = _charge_index(bonds[i + 1].expanded_charges, q_next)
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
    "sector_left_canonicalize",
    "sector_mixed_canonicalize",
    "sector_normalize_center",
    "sector_canonical_residual",
    "spinless_hard_sector_product_mps",
    "hubbard_hard_sector_product_mps",
]
