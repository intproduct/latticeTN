"""General 1D open-boundary model builder (Stage 7B).

This module provides a **unified 1D model-construction layer** that abstracts
the existing Heisenberg and spinless-fermion t-V Hamiltonians behind a single
``ModelSpec`` interface. It is a **model/MPO construction layer, NOT a new
solver** — the AD mainline (differentiable Rayleigh quotient + autograd +
torch optimizer) is unchanged; SVD/QR/canonicalization remain auxiliary
stabilization; exact/DMRG/Lanczos remain reference baselines.

Why a unified spec?
-------------------
Stage 7A added the spinless fermion t-V chain as a parallel Hamiltonian/MPO
path to Heisenberg. Rather than have every future model (Hubbard, XXZ, ...)
re-implement its own dense + MPO + observable wiring, Stage 7B introduces one
spec type that describes a 1D open-boundary chain as a list of terms, and two
builders (``build_dense`` / ``build_mpo``) that construct the Hamiltonian from
the spec. Stage 7B ships two presets — Heisenberg and spinless fermion t-V —
and the builders dispatch to the existing, validated generators
(``operators.heisenberg_dense`` / ``operators.spinless_fermion_dense`` /
``MPO.generate_heisenberg`` / ``MPO.generate_spinless_fermion``) so the
physics is byte-identical to Stage 1–7A. Future stages can add presets and
native spec-driven MPO construction without touching the AD loss path.

Term taxonomy (with explicit boson/fermion distinction)
-------------------------------------------------------
A ``ModelSpec`` carries a ``statistics`` field: ``"boson"`` (spin / hard-core
boson, NO Jordan-Wigner string) or ``"fermion"`` (Jordan-Wigner, with the
parity string ``F = (-1)^n``). The term types are:

- ``OnsiteTerm(op, coeff)``          — sum_i coeff * op_i (diagonal or not;
                                       no JW string needed even for fermions,
                                       because a single-site operator commutes
                                       with the parity string).
- ``TwoSiteTerm(op_i, op_j, coeff)`` — sum_i coeff * op_i op_{i+1}, bosonic
                                       (spin / hard-core boson) two-site term.
- ``FermionHopTerm(coeff)``          — -coeff * sum_i (c^d_i c_{i+1} + h.c.),
                                       JW fermionic; the global operators carry
                                       the parity string ``F^i`` on sites
                                       0..i-1 (NOT hard-core-boson).
- ``DensityDensityTerm(coeff, op)``  — sum_i coeff * op_i op_{i+1}, diagonal;
                                       default op = ``n - 1/2``. No JW string
                                       (diagonal operators commute with F).

The fermion terms MUST keep the JW parity string; they do not degrade to
hard-core-boson terms. This is enforced by dispatching fermion specs to
``spinless_fermion_dense`` / ``MPO.generate_spinless_fermion`` (which build
the string explicitly) and by the alignment tests
(``test_model_builder_fermion`` / ``test_model_builder_mpo_dense``).

Conventions: open boundary, default ``torch.complex128``. Spin preset uses
``S = sigma/2`` (NOT Pauli); fermion preset uses basis ``|0>`/`|1>``, d=2.
The two statistics never mix in one spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch as tc

from .operators import heisenberg_dense, spinless_fermion_dense, hubbard_dense
from .mpo import MPO


Statistics = Literal["boson", "fermion"]


@dataclass
class OnsiteTerm:
    """On-site term ``coeff * sum_i op_i`` (no JW string even for fermions)."""
    op: tc.Tensor
    coeff: float


@dataclass
class TwoSiteTerm:
    """Bosonic two-site term ``coeff * sum_i op_i op_{i+1}`` (NO JW string)."""
    op_i: tc.Tensor
    op_j: tc.Tensor
    coeff: float


@dataclass
class FermionHopTerm:
    """Fermionic NN hopping ``-coeff * sum_i (c^d_i c_{i+1} + h.c.)`` (JW).

    Carries the Jordan-Wigner parity string ``F^i`` on sites 0..i-1; does NOT
    degrade to a hard-core-boson hop.
    """
    coeff: float


@dataclass
class DensityDensityTerm:
    """Diagonal density-density ``coeff * sum_i op_i op_{i+1}`` (no JW string)."""
    coeff: float
    op: tc.Tensor  # typically n - 1/2


@dataclass
class ModelSpec:
    """A 1D open-boundary chain Hamiltonian described as a list of terms.

    Attributes
    ----------
    N : int
        Chain length (number of sites).
    dim : int
        Local Hilbert-space dimension (2 for spin-1/2 / spinless fermion).
    statistics : "boson" | "fermion"
        Whether the operators are bosonic/spin (no JW string) or fermionic
        (JW parity string). The two never mix in one spec.
    terms : list
        Term objects (OnsiteTerm / TwoSiteTerm / FermionHopTerm /
        DensityDensityTerm).
    name : str
        Human-readable model name (for reports).
    dtype : torch.dtype
        Default ``torch.complex128``.
    device : str
        Default ``"cpu"``.
    """
    N: int
    dim: int
    statistics: Statistics
    terms: list = field(default_factory=list)
    name: str = ""
    dtype: tc.dtype = tc.complex128
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def heisenberg_model(N: int, J: float = 1.0, *,
                      dtype: tc.dtype = tc.complex128,
                      device: str = "cpu") -> ModelSpec:
    """Heisenberg open-chain preset: ``H = J * sum_i S_i . S_{i+1}``.

    Spin convention ``S = sigma/2`` (NOT Pauli). Dispatches dense/MPO building
    to ``operators.heisenberg_dense`` / ``MPO.generate_heisenberg``.
    """
    from .operators import spin_operators
    ops = spin_operators(dtype=dtype, device=device)
    spec = ModelSpec(
        N=N, dim=2, statistics="boson",
        name="heisenberg",
        dtype=dtype, device=device,
        terms=[
            TwoSiteTerm(op_i=ops["Sz"], op_j=ops["Sz"], coeff=J),
            TwoSiteTerm(op_i=ops["S+"], op_j=ops["S-"], coeff=J * 0.5),
            TwoSiteTerm(op_i=ops["S-"], op_j=ops["S+"], coeff=J * 0.5),
        ],
    )
    # stash J for the MPO dispatcher
    spec._J = J  # type: ignore[attr-defined]
    return spec


def spinless_fermion_tv_model(N: int, t: float = 1.0, V: float = 0.0,
                              mu: float = 0.0, *,
                              dtype: tc.dtype = tc.complex128,
                              device: str = "cpu") -> ModelSpec:
    """Spinless fermion t-V open-chain preset.

        H = -t sum_i (c^d_i c_{i+1} + h.c.)
            + V sum_i (n_i - 1/2)(n_{i+1} - 1/2)
            - mu sum_i (n_i - 1/2)

    Jordan-Wigner fermions (parity string F = (-1)^n); NOT hard-core boson.
    Dispatches dense/MPO building to ``operators.spinless_fermion_dense`` /
    ``MPO.generate_spinless_fermion``.
    """
    from .fermion_operators import fermion_operators
    ops = fermion_operators(dtype=dtype, device=device)
    spec = ModelSpec(
        N=N, dim=2, statistics="fermion",
        name="spinless_fermion_tv",
        dtype=dtype, device=device,
        terms=[
            FermionHopTerm(coeff=t),
            DensityDensityTerm(coeff=V, op=ops["n_minus_half"]),
            OnsiteTerm(op=ops["n_minus_half"], coeff=-mu),
        ],
    )
    spec._t = t  # type: ignore[attr-defined]
    spec._V = V  # type: ignore[attr-defined]
    spec._mu = mu  # type: ignore[attr-defined]
    return spec


def hubbard_model(N: int, t: float = 1.0, U: float = 4.0, mu: float = 0.0,
                  h: float = 0.0, *,
                  dtype: tc.dtype = tc.complex128,
                  device: str = "cpu") -> ModelSpec:
    """Spinful Hubbard open-chain preset.

        H = -t  sum_{i,sigma} (c^d_{i sigma} c_{i+1,sigma} + h.c.)
            + U  sum_i (n_{i up} - 1/2)(n_{i down} - 1/2)
            - mu sum_i (n_{i up} + n_{i down} - 1)
            - h  sum_i (n_{i up} - n_{i down})

    Local basis ``|0>, |up>, |down>, |up,down>`` (d=4); global mode ordering
    site-major ``(0_up,0_down,1_up,1_down,...)``. Jordan-Wigner fermions
    (per-site parity ``P = F_up x F_down`` on the left-factor site of each
    spin-resolved hop); NOT a spin model and NOT a hard-core-boson model.
    Dispatches dense/MPO building to ``operators.hubbard_dense`` /
    ``MPO.generate_hubbard`` (Stage 7C).
    """
    from .fermion_operators import hubbard_local_operators
    hop = hubbard_local_operators(dtype=dtype, device=device)
    I4 = hop["I"]
    nup = hop["nup"]
    ndown = hop["ndown"]
    nmh_up = nup - 0.5 * I4
    nmh_down = ndown - 0.5 * I4
    spec = ModelSpec(
        N=N, dim=4, statistics="fermion",
        name="hubbard",
        dtype=dtype, device=device,
        terms=[
            # Spin-resolved NN hopping (Jordan-Wigner). Two FermionHopTerm-like
            # entries per spin would be ideal, but the existing FermionHopTerm
            # is spinless-only (d=2); the Hubbard dispatch uses the validated
            # generators directly, so the terms list here is descriptive.
            FermionHopTerm(coeff=t),               # up-spin hop (descriptive)
            FermionHopTerm(coeff=t),               # down-spin hop (descriptive)
            DensityDensityTerm(coeff=U, op=nmh_up),  # (n_up-1/2)(n_down-1/2)
            OnsiteTerm(op=nmh_up + nmh_down, coeff=-mu),  # -mu(n_tot-1)
            OnsiteTerm(op=nup - ndown, coeff=-h),          # -h(n_up-n_down)
        ],
    )
    spec._t = t  # type: ignore[attr-defined]
    spec._U = U  # type: ignore[attr-defined]
    spec._mu = mu  # type: ignore[attr-defined]
    spec._h = h  # type: ignore[attr-defined]
    return spec


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_dense(spec: ModelSpec) -> tc.Tensor:
    """Build the dense Hamiltonian (2**N, 2**N) from a ModelSpec.

    Stage 7B dispatches to the existing validated dense references so the
    physics is byte-identical to Stage 1–7A:
      - Heisenberg preset  -> ``operators.heisenberg_dense``.
      - spinless fermion t-V preset -> ``operators.spinless_fermion_dense``.

    A future stage may add a generic term-by-term dense assembler here; for
    now the dispatch guarantees the JW parity string is preserved exactly for
    fermions (no hard-core-boson degradation).
    """
    if spec.name == "heisenberg":
        J = getattr(spec, "_J", 1.0)
        return heisenberg_dense(spec.N, J=J, dtype=spec.dtype,
                                device=spec.device)
    if spec.name == "spinless_fermion_tv":
        t = getattr(spec, "_t", 1.0)
        V = getattr(spec, "_V", 0.0)
        mu = getattr(spec, "_mu", 0.0)
        return spinless_fermion_dense(spec.N, t=t, V=V, mu=mu,
                                      dtype=spec.dtype, device=spec.device)
    if spec.name == "hubbard":
        t = getattr(spec, "_t", 1.0)
        U = getattr(spec, "_U", 4.0)
        mu = getattr(spec, "_mu", 0.0)
        h = getattr(spec, "_h", 0.0)
        return hubbard_dense(spec.N, t=t, U=U, mu=mu, h=h,
                             dtype=spec.dtype, device=spec.device)
    raise NotImplementedError(
        f"build_dense: preset {spec.name!r} not registered. Stage 7B ships "
        "'heisenberg' and 'spinless_fermion_tv'; Stage 7C adds 'hubbard'.")


def build_mpo(spec: ModelSpec) -> MPO:
    """Build the MPO from a ModelSpec (dispatches to existing generators).

    Stage 7B dispatches to the existing validated MPO generators:
      - Heisenberg preset  -> ``MPO.generate_heisenberg``.
      - spinless fermion t-V preset -> ``MPO.generate_spinless_fermion``.

    The MPO ``to_dense`` matches ``build_dense`` (see
    ``test_model_builder_mpo_dense``). For fermions the JW parity string is
    carried by the MPO's parity-carrying virtual state.
    """
    mpo = MPO.from_bonds(spec.N, spec.dim, dtype=spec.dtype, device=spec.device)
    if spec.name == "heisenberg":
        J = getattr(spec, "_J", 1.0)
        return mpo.generate_heisenberg(J=J)
    if spec.name == "spinless_fermion_tv":
        t = getattr(spec, "_t", 1.0)
        V = getattr(spec, "_V", 0.0)
        mu = getattr(spec, "_mu", 0.0)
        return mpo.generate_spinless_fermion(t=t, V=V, mu=mu)
    if spec.name == "hubbard":
        t = getattr(spec, "_t", 1.0)
        U = getattr(spec, "_U", 4.0)
        mu = getattr(spec, "_mu", 0.0)
        h = getattr(spec, "_h", 0.0)
        return mpo.generate_hubbard(t=t, U=U, mu=mu, h=h)
    raise NotImplementedError(
        f"build_mpo: preset {spec.name!r} not registered. Stage 7B ships "
        "'heisenberg' and 'spinless_fermion_tv'; Stage 7C adds 'hubbard'.")


__all__ = [
    "OnsiteTerm", "TwoSiteTerm", "FermionHopTerm", "DensityDensityTerm",
    "ModelSpec", "heisenberg_model", "spinless_fermion_tv_model",
    "hubbard_model", "build_dense", "build_mpo",
]
