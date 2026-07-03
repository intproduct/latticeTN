"""Unified Hamiltonian-to-MPO entry point."""

from __future__ import annotations

import torch as tc

from .model_spec import ModelSpec, validate_model_spec
from .mpo import MPO


def build_mpo(spec: ModelSpec | dict, dtype=None, device=None) -> MPO:
    """Build an MPO from a Stage 10 ``ModelSpec``.

    Preset models dispatch to the already validated hand-written MPO
    generators. Unsupported custom terms fail clearly instead of silently
    falling back to a different Hamiltonian.
    """

    if isinstance(spec, dict):
        spec = ModelSpec.from_dict(spec)
    validate_model_spec(spec)
    dtype = tc.complex128 if dtype is None else dtype
    device = "cpu" if device is None else device
    name = spec.name
    params = dict(spec.parameters)

    if name == "spinless_fermion_tv":
        name = "spinless_tv"

    if name == "heisenberg":
        return MPO.from_bonds(spec.N, 2, dtype=dtype, device=device).generate_heisenberg(
            J=float(params.get("J", 1.0))
        )
    if name == "tfi":
        return MPO.from_bonds(spec.N, 2, dtype=dtype, device=device).generate_tfi(
            J=float(params.get("J", 1.0)),
            h=float(params.get("h", 1.0)),
        )
    if name == "spinless_tv":
        return MPO.from_bonds(spec.N, 2, dtype=dtype, device=device).generate_spinless_fermion(
            t=float(params.get("t", 1.0)),
            V=float(params.get("V", 0.0)),
            mu=float(params.get("mu", 0.0)),
        )
    if name == "hubbard":
        return MPO.from_bonds(spec.N, 4, dtype=dtype, device=device).generate_hubbard(
            t=float(params.get("t", 1.0)),
            U=float(params.get("U", 4.0)),
            mu=float(params.get("mu", 0.0)),
            h=float(params.get("h", 0.0)),
        )
    if name == "xxz":
        raise NotImplementedError("xxz is registered as experimental but no MPO builder is implemented yet")
    raise NotImplementedError(
        f"custom ModelSpec {name!r} is not supported by the Stage 10 MPO builder yet"
    )


__all__ = ["build_mpo"]
