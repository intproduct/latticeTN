"""Particle-number and fixed-sector diagnostics for dense MPS."""

from __future__ import annotations

import math

import torch as tc

from .mps import MPS
from .mpo import MPO
from .charges import (
    local_number_operator,
    local_nup_operator,
    local_ndown_operator,
    local_ntot_operator,
    local_sz_operator,
)
from .numerics import nonnegative_if_roundoff, positive, real_if_hermitian


def _additive_mpo(mps: MPS, local_op: tc.Tensor) -> MPO:
    d = mps.dim
    I = tc.eye(d, dtype=mps.dtype, device=mps.device)
    q = local_op.to(dtype=mps.dtype, device=mps.device)
    tensors = []
    for i in range(mps.N):
        W = tc.zeros((2, 2, d, d), dtype=mps.dtype, device=mps.device)
        W[0, 0] = I
        W[0, 1] = q
        W[1, 1] = I
        if i == 0:
            W = W[0:1]
        if i == mps.N - 1:
            W = W[:, 1:2]
        tensors.append(W)
    return MPO(tensors, length=mps.N, dim=d, dtype=mps.dtype, device=mps.device)


def _additive_square_mpo(mps: MPS, local_op: tc.Tensor) -> MPO:
    d = mps.dim
    I = tc.eye(d, dtype=mps.dtype, device=mps.device)
    q = local_op.to(dtype=mps.dtype, device=mps.device)
    q2 = q @ q
    tensors = []
    for i in range(mps.N):
        W = tc.zeros((3, 3, d, d), dtype=mps.dtype, device=mps.device)
        W[0, 0] = I
        W[0, 1] = q
        W[0, 2] = q2
        W[1, 1] = I
        W[1, 2] = 2.0 * q
        W[2, 2] = I
        if i == 0:
            W = W[0:1]
        if i == mps.N - 1:
            W = W[:, 2:3]
        tensors.append(W)
    return MPO(tensors, length=mps.N, dim=d, dtype=mps.dtype, device=mps.device)


def additive_expectation(mps: MPS, local_op: tc.Tensor) -> tc.Tensor:
    """Return ``<sum_i op_i>`` as a differentiable scalar tensor."""
    mpo = _additive_mpo(mps, local_op)
    den = positive(mps.overlap(mps), name="additive observable norm")
    return real_if_hermitian(
        mps._expect_MPO(mpo) / den,
        name="additive observable expectation",
    )


def additive_variance(mps: MPS, local_op: tc.Tensor) -> tc.Tensor:
    """Return ``Var(sum_i op_i)`` as a differentiable scalar tensor."""
    q = additive_expectation(mps, local_op)
    q2_mpo = _additive_square_mpo(mps, local_op)
    den = positive(mps.overlap(mps), name="additive-square observable norm")
    q2 = real_if_hermitian(
        mps._expect_MPO(q2_mpo) / den,
        name="additive-square observable expectation",
    )
    return nonnegative_if_roundoff(
        q2 - q * q,
        name="additive observable variance",
        scale=tc.maximum(q2.abs(), (q * q).abs()),
    )


def _as_float(x: tc.Tensor) -> float:
    return float(x.detach().real.cpu())


def _finite_report(report: dict) -> dict:
    for key, value in report.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"non-finite sector report field {key}={value}")
    return report


def total_particle_number(mps: MPS, model: str = "spinless") -> tc.Tensor:
    return additive_expectation(mps, local_number_operator(model, mps.dtype, mps.device))


def particle_number_variance(mps: MPS, model: str = "spinless") -> tc.Tensor:
    return additive_variance(mps, local_number_operator(model, mps.dtype, mps.device))


def sector_leakage_report(mps: MPS, target_n: int | float) -> dict:
    n = _as_float(total_particle_number(mps, model="spinless"))
    var = _as_float(particle_number_variance(mps, model="spinless"))
    return _finite_report({
        "n": n,
        "n_target": target_n,
        "abs_error": abs(n - float(target_n)),
        "variance": var,
    })


def total_nup(mps: MPS) -> tc.Tensor:
    return additive_expectation(mps, local_nup_operator(mps.dtype, mps.device))


def total_ndown(mps: MPS) -> tc.Tensor:
    return additive_expectation(mps, local_ndown_operator(mps.dtype, mps.device))


def total_ntot(mps: MPS) -> tc.Tensor:
    return additive_expectation(mps, local_ntot_operator(mps.dtype, mps.device))


def total_sz(mps: MPS) -> tc.Tensor:
    return additive_expectation(mps, local_sz_operator(mps.dtype, mps.device))


def variance_nup(mps: MPS) -> tc.Tensor:
    return additive_variance(mps, local_nup_operator(mps.dtype, mps.device))


def variance_ndown(mps: MPS) -> tc.Tensor:
    return additive_variance(mps, local_ndown_operator(mps.dtype, mps.device))


def variance_ntot(mps: MPS) -> tc.Tensor:
    return additive_variance(mps, local_ntot_operator(mps.dtype, mps.device))


def hubbard_sector_leakage_report(
    mps: MPS,
    target_nup: int | float,
    target_ndown: int | float,
) -> dict:
    n_up = _as_float(total_nup(mps))
    n_down = _as_float(total_ndown(mps))
    n_tot = _as_float(total_ntot(mps))
    sz = _as_float(total_sz(mps))
    return _finite_report({
        "n_up": n_up,
        "n_up_target": target_nup,
        "n_up_abs_error": abs(n_up - float(target_nup)),
        "n_down": n_down,
        "n_down_target": target_ndown,
        "n_down_abs_error": abs(n_down - float(target_ndown)),
        "n_tot": n_tot,
        "sz": sz,
        "variance_n_up": _as_float(variance_nup(mps)),
        "variance_n_down": _as_float(variance_ndown(mps)),
        "variance_n_tot": _as_float(variance_ntot(mps)),
    })


__all__ = [
    "additive_expectation",
    "additive_variance",
    "total_particle_number",
    "particle_number_variance",
    "sector_leakage_report",
    "total_nup",
    "total_ndown",
    "total_ntot",
    "total_sz",
    "variance_nup",
    "variance_ndown",
    "variance_ntot",
    "hubbard_sector_leakage_report",
]
