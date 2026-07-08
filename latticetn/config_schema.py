"""Method/runtime/observable/result schemas for the Stage 10 runner API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MethodConfig:
    name: str
    chi: int
    sweeps: int
    optimizer: str | None = None
    local_steps: int | None = None
    lbfgs_iters: int | None = None
    lbfgs_tolerance_grad: float | None = None
    lbfgs_tolerance_change: float | None = None
    lr: float | None = None
    sector_mode: str = "none"
    initialization: str = "auto"
    projection: str = "tensor_norm"
    two_site_precondition: str = "theta_norm"
    post_step_stabilization: str = "none"
    grad_clip: float | None = None
    global_steps: int | None = None
    optimizer_reset: str = "never"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MethodConfig":
        return cls(
            name=str(data["name"]),
            chi=int(data["chi"]),
            sweeps=int(data.get("sweeps", 1)),
            optimizer=data.get("optimizer"),
            local_steps=data.get("local_steps"),
            lbfgs_iters=data.get("lbfgs_iters"),
            lbfgs_tolerance_grad=data.get("lbfgs_tolerance_grad"),
            lbfgs_tolerance_change=data.get("lbfgs_tolerance_change"),
            lr=data.get("lr"),
            sector_mode=str(data.get("sector_mode", "none")),
            initialization=str(data.get("initialization", data.get("init", "auto"))),
            projection=str(data.get("projection", data.get("stabilization", "tensor_norm"))),
            two_site_precondition=str(data.get("two_site_precondition", "theta_norm")),
            post_step_stabilization=str(data.get("post_step_stabilization", data.get("stabilization", "none"))),
            grad_clip=data.get("grad_clip"),
            global_steps=data.get("global_steps"),
            optimizer_reset=str(data.get("optimizer_reset", "never")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chi": self.chi,
            "sweeps": self.sweeps,
            "optimizer": self.optimizer,
            "local_steps": self.local_steps,
            "lbfgs_iters": self.lbfgs_iters,
            "lbfgs_tolerance_grad": self.lbfgs_tolerance_grad,
            "lbfgs_tolerance_change": self.lbfgs_tolerance_change,
            "lr": self.lr,
            "sector_mode": self.sector_mode,
            "initialization": self.initialization,
            "projection": self.projection,
            "two_site_precondition": self.two_site_precondition,
            "post_step_stabilization": self.post_step_stabilization,
            "grad_clip": self.grad_clip,
            "global_steps": self.global_steps,
            "optimizer_reset": self.optimizer_reset,
        }


@dataclass
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "complex64"
    seed: int | None = None
    no_ed: bool = True
    output: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        return cls(
            device=str(data.get("device", "auto")),
            dtype=str(data.get("dtype", "complex64")),
            seed=data.get("seed"),
            no_ed=bool(data.get("no_ed", True)),
            output=data.get("output"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "no_ed": self.no_ed,
            "output": self.output,
        }


@dataclass
class ObservableSpec:
    names: list[str] = field(default_factory=lambda: ["energy"])

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ObservableSpec":
        if data is None:
            return cls()
        return cls(names=[str(x) for x in data.get("names", ["energy"])])

    def to_dict(self) -> dict[str, Any]:
        return {"names": list(self.names)}


@dataclass
class ResultSchema:
    model: dict
    method: dict
    runtime: dict
    summary: dict
    sweep_history: list[dict]
    observables: dict
    diagnostics: dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResultSchema":
        return cls(
            model=dict(data["model"]),
            method=dict(data["method"]),
            runtime=dict(data["runtime"]),
            summary=dict(data["summary"]),
            sweep_history=list(data.get("sweep_history", [])),
            observables=dict(data.get("observables", {})),
            diagnostics=dict(data.get("diagnostics", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "method": self.method,
            "runtime": self.runtime,
            "summary": self.summary,
            "sweep_history": self.sweep_history,
            "observables": self.observables,
            "diagnostics": self.diagnostics,
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def write_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = [
    "MethodConfig",
    "RuntimeConfig",
    "ObservableSpec",
    "ResultSchema",
    "write_json",
]
