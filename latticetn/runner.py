"""Unified Stage 10 job runner API."""

from __future__ import annotations

import time
from argparse import Namespace
from typing import Any

import torch as tc

from .model_spec import ModelSpec
from .config_schema import MethodConfig, RuntimeConfig, ObservableSpec, ResultSchema
from .hamiltonian_builder import build_mpo
from .mps import MPS
from .initial_states import neel_spin_state, spinless_half_filled_cdw_state, hubbard_half_filled_neel_state
from .charge_sectors import (
    ChargeAwareMPS,
    spinless_hard_sector_product_mps,
    hubbard_hard_sector_product_mps,
    apply_charge_masks_,
    zero_forbidden_gradients_,
    max_forbidden_abs,
)
from .sector_observables import (
    total_particle_number,
    sector_leakage_report,
    total_nup,
    total_ndown,
    hubbard_sector_leakage_report,
)
from . import contractions as K


def parse_dtype(name: str) -> tc.dtype:
    table = {"complex64": tc.complex64, "complex128": tc.complex128}
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if tc.cuda.is_available() else "cpu"
    if name == "cuda" and not tc.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return name


def _model_dim(name: str) -> int:
    return 4 if name == "hubbard" else 2


def _sector_mode(model: ModelSpec, method: MethodConfig) -> str:
    if method.sector_mode != "none":
        return method.sector_mode
    if model.sector and "mode" in model.sector:
        return str(model.sector["mode"])
    return "none"


def _target_n(model: ModelSpec) -> int | None:
    if not model.sector:
        return None
    return model.sector.get("target_n")


def _target_nup_ndown(model: ModelSpec) -> tuple[int | None, int | None]:
    if not model.sector:
        return None, None
    return model.sector.get("target_nup"), model.sector.get("target_ndown")


def _make_initial_mps(model: ModelSpec, method: MethodConfig, dtype, device) -> tuple[MPS, ChargeAwareMPS | None]:
    name = model.name
    mode = _sector_mode(model, method)
    if mode == "hard":
        if name == "spinless_tv":
            target = _target_n(model)
            if target is None:
                if model.N % 2 != 0:
                    raise ValueError("hard spinless sector requires target_n for odd N")
                target = model.N // 2
            camps = spinless_hard_sector_product_mps(model.N, target, method.chi, dtype=dtype, device=device)
            return camps.mps, camps
        if name == "hubbard":
            nup, ndown = _target_nup_ndown(model)
            if nup is None or ndown is None:
                if model.N % 2 != 0:
                    raise ValueError("hard Hubbard sector requires target_nup/target_ndown for odd N")
                nup = ndown = model.N // 2
            camps = hubbard_hard_sector_product_mps(
                model.N, nup, ndown, method.chi, dtype=dtype, device=device
            )
            return camps.mps, camps
        raise ValueError("hard sector mode is supported only for spinless_tv and hubbard")
    if name in {"heisenberg", "tfi"}:
        return neel_spin_state(model.N, dtype=dtype, device=device), None
    if name == "spinless_tv":
        return spinless_half_filled_cdw_state(model.N, dtype=dtype, device=device), None
    if name == "hubbard":
        return hubbard_half_filled_neel_state(model.N, dtype=dtype, device=device), None
    return MPS(model.N, _model_dim(name), method.chi, dtype=dtype, device=device), None


def _energy(mps: MPS, mpo) -> tc.Tensor:
    return K.rayleigh_energy_native(mps, mpo).real


def _float(x: tc.Tensor) -> float:
    return float(x.detach().real.cpu())


def _bond_dims(mps: MPS) -> list[int]:
    return [int(mps.tensors[i].shape[2]) for i in range(mps.N - 1)]


def _max_bond(mps: MPS) -> int:
    bd = _bond_dims(mps)
    return max(bd) if bd else 1


def _sector_report(model: ModelSpec, mps: MPS) -> dict | None:
    name = model.name
    if name == "spinless_tv":
        target = _target_n(model)
        if target is not None:
            return sector_leakage_report(mps, target_n=target)
    if name == "hubbard":
        nup, ndown = _target_nup_ndown(model)
        if nup is not None and ndown is not None:
            return hubbard_sector_leakage_report(mps, target_nup=nup, target_ndown=ndown)
    return None


def _sector_penalty(model: ModelSpec, method: MethodConfig, mps: MPS) -> tc.Tensor:
    zero = tc.zeros((), dtype=tc.float64, device=mps.device)
    if _sector_mode(model, method) != "soft":
        return zero
    sector = model.sector or {}
    if model.name == "spinless_tv":
        target = sector.get("target_n")
        lam = float(sector.get("lambda_n", 0.0))
        if target is None or lam == 0.0:
            return zero
        return lam * (total_particle_number(mps, model="spinless") - float(target)) ** 2
    if model.name == "hubbard":
        loss = zero
        nup = sector.get("target_nup")
        ndown = sector.get("target_ndown")
        lam_up = float(sector.get("lambda_nup", 0.0))
        lam_down = float(sector.get("lambda_ndown", 0.0))
        if nup is not None and lam_up != 0.0:
            loss = loss + lam_up * (total_nup(mps) - float(nup)) ** 2
        if ndown is not None and lam_down != 0.0:
            loss = loss + lam_down * (total_ndown(mps) - float(ndown)) ** 2
        return loss
    return zero


def _grad_norm(params, device: str) -> float:
    sq = tc.zeros((), dtype=tc.float64, device=device)
    seen = False
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq = sq + (g.conj() * g).real.sum().to(tc.float64)
        seen = True
    return float(sq.sqrt().cpu()) if seen else 0.0


def _run_ad(model: ModelSpec, method: MethodConfig, runtime: RuntimeConfig, dtype, device: str) -> dict:
    mpo = build_mpo(model, dtype=dtype, device=device)
    mps, camps = _make_initial_mps(model, method, dtype, device)
    if runtime.seed is not None:
        tc.manual_seed(int(runtime.seed))
    for p in mps.tensors:
        p.requires_grad_(True)
    params = list(mps.parameters())
    mode = _sector_mode(model, method)
    history = []
    max_grad = 0.0
    max_forbidden_grad = 0.0
    t0 = time.perf_counter()
    for sweep in range(method.sweeps):
        opt_name = method.optimizer or "adam"
        lr = float(method.lr if method.lr is not None else 0.01)
        if opt_name == "adam":
            opt = tc.optim.Adam(params, lr=lr)
        elif opt_name == "lbfgs":
            opt = tc.optim.LBFGS(
                params,
                lr=lr,
                max_iter=int(method.lbfgs_iters or 5),
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError(f"unsupported optimizer {opt_name!r}")

        def closure():
            opt.zero_grad(set_to_none=True)
            if camps is not None:
                apply_charge_masks_(mps, camps.masks)
            loss = _energy(mps, mpo) + _sector_penalty(model, method, mps).real
            loss.backward()
            nonlocal max_forbidden_grad
            if camps is not None:
                max_forbidden_grad = max(
                    max_forbidden_grad,
                    zero_forbidden_gradients_(params, camps.masks),
                )
            return loss

        steps = int(method.local_steps or 1)
        for _ in range(max(1, steps)):
            if opt_name == "adam":
                closure()
                max_grad = max(max_grad, _grad_norm(params, mps.device))
                opt.step()
            else:
                opt.step(closure)
                max_grad = max(max_grad, _grad_norm(params, mps.device))
            if camps is not None:
                apply_charge_masks_(mps, camps.masks)
        energy = _float(_energy(mps, mpo))
        rec = {
            "sweep": sweep,
            "energy": energy,
            "energy_per_site": energy / model.N,
            "sector_report": _sector_report(model, mps),
            "bond_dims": _bond_dims(mps),
            "max_bond": _max_bond(mps),
            "gradient_norm": max_grad,
        }
        if mode == "hard":
            rec["max_forbidden_abs"] = max_forbidden_abs(mps, camps.masks)
            rec["max_forbidden_grad_abs"] = max_forbidden_grad
        history.append(rec)
    runtime_s = time.perf_counter() - t0
    final_e = history[-1]["energy"] if history else _float(_energy(mps, mpo))
    return {
        "history": history,
        "summary": {
            "final_energy": final_e,
            "final_energy_per_site": final_e / model.N,
            "final_max_bond": _max_bond(mps),
            "final_gradient_norm": max_grad,
            "runtime": runtime_s,
        },
        "diagnostics": {
            "ed_used": False,
            "classical_dmrg_used": False,
            "lanczos_used": False,
            "ad_used": True,
            "dense_hamiltonian_built": False,
            "sector_mode": mode,
            "max_forbidden_abs": max_forbidden_abs(mps, camps.masks) if camps is not None else None,
            "max_forbidden_grad_abs": max_forbidden_grad if camps is not None else None,
        },
        "mps": mps,
    }


def _run_classical_dmrg(model: ModelSpec, method: MethodConfig, runtime: RuntimeConfig, dtype, device: str) -> dict:
    if model.name != "heisenberg":
        raise NotImplementedError("classical dmrg Stage 10 entry currently supports heisenberg only")
    from .dmrg import two_site_sweep

    mpo = build_mpo(model, dtype=dtype, device=device)
    mps = MPS(model.N, 2, method.chi, dtype=dtype, device=device)
    tensors = [x.detach().clone() for x in mps.tensors]
    history = []
    t0 = time.perf_counter()
    solver = "lanczos"
    for sweep in range(method.sweeps):
        direction = "right" if sweep % 2 == 0 else "left"
        tensors, local_e, trunc = two_site_sweep(
            tensors,
            mpo,
            method.chi,
            direction,
            solver=solver,
            lanczos_kwargs={"max_iter": 8},
        )
        mps_cur = MPS.from_tensors(tensors, dtype=dtype, device=device, requires_grad=False)
        e = _float(_energy(mps_cur, mpo))
        history.append({
            "sweep": sweep,
            "direction": direction,
            "energy": e,
            "energy_per_site": e / model.N,
            "local_last_energy": local_e,
            "truncation": max(trunc) if trunc else 0.0,
            "bond_dims": _bond_dims(mps_cur),
            "max_bond": _max_bond(mps_cur),
        })
    runtime_s = time.perf_counter() - t0
    final_e = history[-1]["energy"] if history else float("nan")
    return {
        "history": history,
        "summary": {
            "final_energy": final_e,
            "final_energy_per_site": final_e / model.N,
            "final_max_bond": history[-1]["max_bond"] if history else 1,
            "runtime": runtime_s,
        },
        "diagnostics": {
            "ed_used": False,
            "classical_dmrg_used": True,
            "lanczos_used": True,
            "ad_used": False,
            "dense_hamiltonian_built": False,
            "sector_mode": "none",
        },
        "mps": MPS.from_tensors(tensors, dtype=dtype, device=device, requires_grad=False),
    }


def _observables(names: list[str], model: ModelSpec, result: dict) -> dict:
    obs = {}
    summary = result["summary"]
    history = result["history"]
    mps = result["mps"]
    for name in names:
        if name == "energy":
            obs["energy"] = summary["final_energy"]
        elif name == "energy_per_site":
            obs["energy_per_site"] = summary["final_energy_per_site"]
        elif name == "sector":
            obs["sector"] = _sector_report(model, mps)
        elif name == "bond_dims":
            obs["bond_dims"] = _bond_dims(mps)
        elif name == "truncation":
            obs["truncation"] = [x.get("truncation", x.get("max_trunc")) for x in history]
        elif name == "gradient_norm":
            obs["gradient_norm"] = summary.get("final_gradient_norm")
        elif name == "runtime":
            obs["runtime"] = summary.get("runtime")
        else:
            raise NotImplementedError(f"observable {name!r} is not implemented")
    return obs


def run_latticetn_job(
    model_spec: ModelSpec | dict,
    method_config: MethodConfig | dict,
    runtime_config: RuntimeConfig | dict,
    observables: ObservableSpec | dict | None = None,
) -> dict:
    """Run a structured latticeTN job and return a JSON-serializable dict."""

    model = model_spec if isinstance(model_spec, ModelSpec) else ModelSpec.from_dict(model_spec)
    method = method_config if isinstance(method_config, MethodConfig) else MethodConfig.from_dict(method_config)
    runtime = runtime_config if isinstance(runtime_config, RuntimeConfig) else RuntimeConfig.from_dict(runtime_config)
    obs_spec = observables if isinstance(observables, ObservableSpec) else ObservableSpec.from_dict(observables)

    if not runtime.no_ed and model.N > 10:
        raise ValueError("ED was requested for a large Hilbert space; refuse unless N <= 10")
    dtype = parse_dtype(runtime.dtype)
    device = resolve_device(runtime.device)
    if runtime.seed is not None:
        tc.manual_seed(int(runtime.seed))

    if method.name == "ad_dmrg":
        raw = _run_ad(model, method, runtime, dtype, device)
    elif method.name == "dmrg":
        raw = _run_classical_dmrg(model, method, runtime, dtype, device)
    else:
        raise ValueError(f"unsupported method {method.name!r}")

    model_dict = model.to_dict()
    runtime_dict = runtime.to_dict()
    runtime_dict["resolved_device"] = device
    result = ResultSchema(
        model=model_dict,
        method=method.to_dict(),
        runtime=runtime_dict,
        summary=raw["summary"],
        sweep_history=raw["history"],
        observables=_observables(obs_spec.names, model, raw),
        diagnostics=raw["diagnostics"],
    )
    out = result.to_dict()
    if runtime.output:
        result.write_json(runtime.output)
    return out


def namespace_from_legacy_ad_args(args: Namespace) -> tuple[ModelSpec, MethodConfig, RuntimeConfig, ObservableSpec]:
    """Translate the existing AD benchmark CLI args into Stage 10 schemas."""

    params = {"J": args.J, "h": args.h, "t": args.t, "V": args.V, "U": args.U, "mu": args.mu}
    relevant = {
        "heisenberg": {"J": params["J"]},
        "tfi": {"J": params["J"], "h": params["h"]},
        "spinless_tv": {"t": params["t"], "V": params["V"], "mu": params["mu"]},
        "hubbard": {"t": params["t"], "U": params["U"], "mu": params["mu"], "h": params["h"]},
    }[args.model]
    sector = {"mode": args.sector_mode}
    if args.target_n is not None:
        sector["target_n"] = args.target_n
    if args.target_nup is not None:
        sector["target_nup"] = args.target_nup
    if args.target_ndown is not None:
        sector["target_ndown"] = args.target_ndown
    if args.lambda_n != 0.0:
        sector["lambda_n"] = args.lambda_n
    if args.lambda_nup != 0.0:
        sector["lambda_nup"] = args.lambda_nup
    if args.lambda_ndown != 0.0:
        sector["lambda_ndown"] = args.lambda_ndown
    model = ModelSpec.from_dict({
        "name": args.model,
        "N": args.N,
        "boundary": "obc",
        "parameters": relevant,
        "sector": sector,
    })
    method = MethodConfig(
        name="ad_dmrg",
        chi=args.chi,
        sweeps=args.sweeps,
        optimizer=args.optimizer,
        local_steps=args.local_steps,
        lbfgs_iters=args.lbfgs_iters,
        lr=args.lr,
        sector_mode=args.sector_mode,
    )
    runtime = RuntimeConfig(
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        no_ed=args.no_ed,
        output=str(args.output) if args.output is not None else None,
    )
    return model, method, runtime, ObservableSpec(["energy", "energy_per_site", "sector", "bond_dims", "gradient_norm", "runtime"])


__all__ = ["run_latticetn_job", "namespace_from_legacy_ad_args", "parse_dtype", "resolve_device"]
