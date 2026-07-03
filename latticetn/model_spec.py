"""Structured 1D Hamiltonian model specification.

Stage 10 intentionally uses explicit JSON/dict/dataclass specs. It does not
parse free-form Hamiltonian strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_TERM_PATTERNS = {
    "onsite",
    "nearest_neighbor",
    "nearest_neighbor_hopping",
    "two_site",
}


@dataclass
class OperatorRef:
    op: str
    site: str | int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OperatorRef":
        return cls(op=str(data["op"]), site=data["site"])

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "site": self.site}


@dataclass
class TermSpec:
    coefficient: str | float
    operators: list[OperatorRef]
    pattern: str
    plus_hc: bool = False
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TermSpec":
        return cls(
            coefficient=data.get("coefficient", 1.0),
            operators=[OperatorRef.from_dict(x) for x in data.get("operators", [])],
            pattern=str(data["pattern"]),
            plus_hc=bool(data.get("plus_hc", False)),
            description=data.get("description"),
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "coefficient": self.coefficient,
            "operators": [op.to_dict() for op in self.operators],
            "pattern": self.pattern,
            "plus_hc": self.plus_hc,
        }
        if self.description is not None:
            out["description"] = self.description
        return out


@dataclass
class ModelSpec:
    name: str
    N: int
    local_basis: str
    boundary: str = "obc"
    parameters: dict[str, float] = field(default_factory=dict)
    terms: list[TermSpec] = field(default_factory=list)
    sector: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        spec = cls(
            name=str(data["name"]),
            N=int(data["N"]),
            local_basis=str(data.get("local_basis", _default_basis(str(data["name"])))),
            boundary=str(data.get("boundary", "obc")),
            parameters={k: float(v) for k, v in data.get("parameters", {}).items()},
            terms=[TermSpec.from_dict(x) for x in data.get("terms", [])],
            sector=data.get("sector"),
            metadata=dict(data.get("metadata", {})),
        )
        validate_model_spec(spec)
        return spec

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "N": self.N,
            "local_basis": self.local_basis,
            "boundary": self.boundary,
            "parameters": dict(self.parameters),
            "terms": [term.to_dict() for term in self.terms],
            "metadata": dict(self.metadata),
        }
        if self.sector is not None:
            out["sector"] = dict(self.sector)
        return out


def _default_basis(name: str) -> str:
    return {
        "heisenberg": "spin_half",
        "tfi": "spin_half",
        "xxz": "spin_half",
        "spinless_tv": "spinless",
        "spinless_fermion_tv": "spinless",
        "hubbard": "hubbard",
    }.get(name, "custom")


def validate_model_spec(spec: ModelSpec) -> None:
    if spec.N <= 0:
        raise ValueError(f"ModelSpec.N must be positive, got {spec.N}")
    if spec.boundary not in {"obc", "open"}:
        raise ValueError(f"only open/OBC boundary is supported, got {spec.boundary!r}")
    for term in spec.terms:
        if term.pattern not in SUPPORTED_TERM_PATTERNS:
            raise ValueError(
                f"unsupported term pattern {term.pattern!r}; "
                f"supported patterns are {sorted(SUPPORTED_TERM_PATTERNS)}"
            )


__all__ = [
    "OperatorRef",
    "TermSpec",
    "ModelSpec",
    "validate_model_spec",
    "SUPPORTED_TERM_PATTERNS",
]
