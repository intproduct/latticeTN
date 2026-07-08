"""Preset model registry for Stage 10 backend/API contracts."""

from __future__ import annotations

from copy import deepcopy

from .model_spec import ModelSpec, TermSpec, OperatorRef


_SCHEMAS: dict[str, dict] = {
    "heisenberg": {
        "id": "heisenberg",
        "label": "1D Heisenberg spin-1/2 chain",
        "local_basis": "spin_half",
        "parameters": [{"name": "J", "type": "float", "default": 1.0}],
        "supported_methods": ["dmrg", "ad_global", "ad_two_site"],
        "supported_sector_modes": ["none"],
    },
    "tfi": {
        "id": "tfi",
        "label": "1D transverse-field Ising chain",
        "local_basis": "spin_half",
        "parameters": [
            {"name": "J", "type": "float", "default": 1.0},
            {"name": "h", "type": "float", "default": 1.0},
        ],
        "supported_methods": ["ad_global", "ad_two_site"],
        "supported_sector_modes": ["none"],
    },
    "spinless_tv": {
        "id": "spinless_tv",
        "label": "1D spinless fermion t-V model",
        "local_basis": "spinless",
        "parameters": [
            {"name": "t", "type": "float", "default": 1.0},
            {"name": "V", "type": "float", "default": 0.0},
            {"name": "mu", "type": "float", "default": 0.0},
        ],
        "supported_methods": ["ad_global", "ad_two_site"],
        "supported_sector_modes": ["none", "soft", "hard"],
    },
    "hubbard": {
        "id": "hubbard",
        "label": "1D Hubbard model",
        "local_basis": "hubbard",
        "parameters": [
            {"name": "t", "type": "float", "default": 1.0},
            {"name": "U", "type": "float", "default": 4.0},
            {"name": "mu", "type": "float", "default": 0.0},
            {"name": "h", "type": "float", "default": 0.0},
        ],
        "supported_methods": ["ad_global", "ad_two_site"],
        "supported_sector_modes": ["none", "soft", "hard"],
    },
    "xxz": {
        "id": "xxz",
        "label": "1D XXZ spin-1/2 chain",
        "local_basis": "spin_half",
        "parameters": [
            {"name": "Jxy", "type": "float", "default": 1.0},
            {"name": "Jz", "type": "float", "default": 1.0},
        ],
        "supported_methods": [],
        "supported_sector_modes": ["none"],
        "status": "experimental_not_implemented",
    },
}


def list_model_ids() -> list[str]:
    return sorted(_SCHEMAS)


def get_model_schema(model_id: str) -> dict:
    try:
        return deepcopy(_SCHEMAS[model_id])
    except KeyError as exc:
        raise ValueError(f"unknown model id {model_id!r}") from exc


def _defaults(model_id: str) -> dict[str, float]:
    return {p["name"]: float(p["default"]) for p in _SCHEMAS[model_id]["parameters"]}


def build_model_spec(
    model_id: str,
    N: int,
    parameters: dict | None = None,
    boundary: str = "obc",
    sector: dict | None = None,
) -> ModelSpec:
    schema = get_model_schema(model_id)
    if schema.get("status") == "experimental_not_implemented":
        raise NotImplementedError(f"model {model_id!r} is registered but not implemented")
    params = _defaults(model_id)
    params.update({k: float(v) for k, v in (parameters or {}).items()})
    terms = _preset_terms(model_id, params)
    return ModelSpec(
        name=model_id,
        N=N,
        local_basis=schema["local_basis"],
        boundary=boundary,
        parameters=params,
        terms=terms,
        sector=sector,
        metadata={"preset": True, "label": schema["label"]},
    )


def _preset_terms(model_id: str, params: dict[str, float]) -> list[TermSpec]:
    if model_id == "heisenberg":
        return [
            TermSpec(params["J"], [OperatorRef("Sx", "i"), OperatorRef("Sx", "i+1")], "nearest_neighbor"),
            TermSpec(params["J"], [OperatorRef("Sy", "i"), OperatorRef("Sy", "i+1")], "nearest_neighbor"),
            TermSpec(params["J"], [OperatorRef("Sz", "i"), OperatorRef("Sz", "i+1")], "nearest_neighbor"),
        ]
    if model_id == "tfi":
        return [
            TermSpec(-params["J"], [OperatorRef("Sz", "i"), OperatorRef("Sz", "i+1")], "nearest_neighbor"),
            TermSpec(-params["h"], [OperatorRef("Sx", "i")], "onsite"),
        ]
    if model_id == "spinless_tv":
        return [
            TermSpec(params["t"], [OperatorRef("c†", "i"), OperatorRef("c", "i+1")], "nearest_neighbor_hopping", plus_hc=True),
            TermSpec(params["V"], [OperatorRef("n-1/2", "i"), OperatorRef("n-1/2", "i+1")], "nearest_neighbor"),
            TermSpec(-params["mu"], [OperatorRef("n-1/2", "i")], "onsite"),
        ]
    if model_id == "hubbard":
        return [
            TermSpec(params["t"], [OperatorRef("c†_sigma", "i"), OperatorRef("c_sigma", "i+1")], "nearest_neighbor_hopping", plus_hc=True),
            TermSpec(params["U"], [OperatorRef("n_up-1/2", "i"), OperatorRef("n_down-1/2", "i")], "onsite"),
            TermSpec(-params["mu"], [OperatorRef("n_tot-1", "i")], "onsite"),
            TermSpec(-params["h"], [OperatorRef("n_up-n_down", "i")], "onsite"),
        ]
    raise ValueError(f"no preset terms for {model_id!r}")


def build_mpo_from_model_spec(model_spec: ModelSpec, dtype=None, device=None):
    from .hamiltonian_builder import build_mpo

    return build_mpo(model_spec, dtype=dtype, device=device)


__all__ = [
    "list_model_ids",
    "get_model_schema",
    "build_model_spec",
    "build_mpo_from_model_spec",
]
