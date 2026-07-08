"""Unified Stage 10 job runner API."""

from __future__ import annotations

import time
import warnings
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
from .charges import local_number_operator, local_ntot_operator, local_sz_operator
from .fermion_operators import hubbard_local_operators
from . import contractions as K
from .ad_two_site import train_ad_two_site
from .ad_variational import _project


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


def _resolved_initialization(model: ModelSpec, method: MethodConfig) -> str:
    init = method.initialization or "auto"
    if init != "auto":
        return init
    if model.name in {"heisenberg", "tfi"}:
        return "neel"
    if model.name == "spinless_tv":
        return "spinless_cdw"
    if model.name == "hubbard":
        return "hubbard_neel"
    return "random"


def _make_initial_mps(model: ModelSpec, method: MethodConfig, dtype, device) -> tuple[MPS, ChargeAwareMPS | None, str]:
    name = model.name
    mode = _sector_mode(model, method)
    init = _resolved_initialization(model, method)
    if init == "random" and mode != "hard":
        return MPS(model.N, _model_dim(name), method.chi, dtype=dtype, device=device), None, init
    if mode == "hard":
        if name == "spinless_tv":
            target = _target_n(model)
            if target is None:
                if model.N % 2 != 0:
                    raise ValueError("hard spinless sector requires target_n for odd N")
                target = model.N // 2
            pattern = "cdw" if init in {"spinless_cdw", "cdw"} else "left"
            camps = spinless_hard_sector_product_mps(
                model.N, target, method.chi, pattern=pattern, dtype=dtype, device=device
            )
            return camps.mps, camps, init
        if name == "hubbard":
            nup, ndown = _target_nup_ndown(model)
            if nup is None or ndown is None:
                if model.N % 2 != 0:
                    raise ValueError("hard Hubbard sector requires target_nup/target_ndown for odd N")
                nup = ndown = model.N // 2
            camps = hubbard_hard_sector_product_mps(
                model.N, nup, ndown, method.chi,
                pattern="neel" if init in {"hubbard_neel", "neel"} else "balanced",
                dtype=dtype, device=device
            )
            return camps.mps, camps, init
        raise ValueError("hard sector mode is supported only for spinless_tv and hubbard")
    if init == "neel":
        if name not in {"heisenberg", "tfi"}:
            raise ValueError(f"initialization {init!r} is valid only for spin models")
        return neel_spin_state(model.N, dtype=dtype, device=device), None, init
    if init in {"spinless_cdw", "cdw"}:
        if name != "spinless_tv":
            raise ValueError(f"initialization {init!r} is valid only for spinless_tv")
        return spinless_half_filled_cdw_state(model.N, dtype=dtype, device=device), None, init
    if init in {"hubbard_neel", "neel_hubbard"}:
        if name != "hubbard":
            raise ValueError(f"initialization {init!r} is valid only for hubbard")
        return hubbard_half_filled_neel_state(model.N, dtype=dtype, device=device), None, init
    raise ValueError(f"unsupported initialization {init!r}")


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


def _local_expectation(mps: MPS, local_op: tc.Tensor, site: int) -> float:
    if not 0 <= site < mps.N:
        raise ValueError(f"site index {site} outside chain length {mps.N}")
    q = local_op.to(dtype=mps.dtype, device=mps.device)
    identity = tc.eye(mps.dim, dtype=mps.dtype, device=mps.device)
    numer = tc.ones((1, 1), dtype=mps.dtype, device=mps.device)
    denom = tc.ones((1, 1), dtype=mps.dtype, device=mps.device)
    for idx, A in enumerate(mps.tensors):
        op = q if idx == site else identity
        numer = tc.einsum("ab,asi,st,btj->ij", numer, A.conj(), op, A)
        denom = tc.einsum("ab,asi,bsj->ij", denom, A.conj(), A)
    value = numer.reshape(()) / denom.reshape(())
    return _float(value)


def _mid_site(model: ModelSpec) -> int:
    return model.N // 2


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
    if runtime.seed is not None:
        tc.manual_seed(int(runtime.seed))
    mps, camps, init = _make_initial_mps(model, method, dtype, device)
    for p in mps.tensors:
        p.requires_grad_(True)
    params = list(mps.parameters())
    mode = _sector_mode(model, method)
    history = []
    max_grad = 0.0
    max_forbidden_grad = 0.0
    optimizer_steps = 0
    closure_evals = 0
    t0 = time.perf_counter()
    opt_name = method.optimizer or "adam"
    lr = float(method.lr if method.lr is not None else 0.01)
    if opt_name == "adam":
        opt = tc.optim.Adam(params, lr=lr)
    elif opt_name == "lbfgs":
        opt = tc.optim.LBFGS(
            params,
            lr=lr,
            max_iter=int(method.lbfgs_iters or 5),
            tolerance_grad=float(method.lbfgs_tolerance_grad or 1e-7),
            tolerance_change=float(method.lbfgs_tolerance_change or 1e-9),
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"unsupported optimizer {opt_name!r}")

    initial_energy = _float(_energy(mps, mpo))
    initial_bond_dims = _bond_dims(mps)
    best_energy = initial_energy
    best_step = 0
    global_steps = int(method.global_steps or max(1, method.sweeps) * max(1, int(method.local_steps or 1)))

    def closure():
        nonlocal max_forbidden_grad, closure_evals
        opt.zero_grad(set_to_none=True)
        if camps is not None:
            apply_charge_masks_(mps, camps.masks)
        loss = _energy(mps, mpo) + _sector_penalty(model, method, mps).real
        closure_evals += 1
        loss.backward()
        if camps is not None:
            max_forbidden_grad = max(
                max_forbidden_grad,
                zero_forbidden_gradients_(params, camps.masks),
            )
        if method.grad_clip is not None and method.grad_clip > 0:
            tc.nn.utils.clip_grad_norm_(params, float(method.grad_clip))
        return loss

    for step in range(1, global_steps + 1):
        if opt_name == "adam":
            closure()
            max_grad = max(max_grad, _grad_norm(params, mps.device))
            opt.step()
        else:
            opt.step(closure)
            max_grad = max(max_grad, _grad_norm(params, mps.device))
        optimizer_steps += 1
        if camps is not None:
            apply_charge_masks_(mps, camps.masks)
        _project(mps, method.projection)
        if camps is not None:
            apply_charge_masks_(mps, camps.masks)
        energy = _float(_energy(mps, mpo))
        if energy < best_energy:
            best_energy = energy
            best_step = step
        rec = {
            "step": step,
            "sweep": step - 1,
            "energy": energy,
            "energy_per_site": energy / model.N,
            "sector_report": _sector_report(model, mps),
            "bond_dims": _bond_dims(mps),
            "max_bond": _max_bond(mps),
            "gradient_norm": max_grad,
            "state_norm": _float(K.native_norm(mps)),
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
            "initial_energy": initial_energy,
            "initial_bond_dims": initial_bond_dims,
            "final_bond_dims": _bond_dims(mps),
            "chi_requested": method.chi,
            "initial_max_bond": max(initial_bond_dims) if initial_bond_dims else 1,
            "final_gradient_norm": max_grad,
            "best_energy": best_energy,
            "best_step": best_step,
            "global_steps": global_steps,
            "optimizer_steps": optimizer_steps,
            "closure_evals": closure_evals,
            "runtime": runtime_s,
        },
        "diagnostics": {
            "algorithm_id": "ad_global",
            "optimizer_path": "global_ad_hard_charge_mask" if mode == "hard" else (
                "global_ad_with_sector_penalty" if mode == "soft" else "global_ad"
            ),
            "ed_used": False,
            "classical_dmrg_used": False,
            "lanczos_used": False,
            "ad_used": True,
            "dense_hamiltonian_built": False,
            "sector_mode": mode,
            "initialization": init,
            "projection": method.projection,
            "optimizer_reset": "never",
            "max_forbidden_abs": max_forbidden_abs(mps, camps.masks) if camps is not None else None,
            "max_forbidden_grad_abs": max_forbidden_grad if camps is not None else None,
        },
        "mps": mps,
    }


def _run_ad_two_site(model: ModelSpec, method: MethodConfig, runtime: RuntimeConfig, dtype, device: str) -> dict:
    mode = _sector_mode(model, method)
    if mode != "none":
        raise ValueError(
            f"ad_two_site does not support sector_mode={mode!r}; use ad_global "
            "for soft penalties or hard charge masks"
        )
    if runtime.seed is not None:
        tc.manual_seed(int(runtime.seed))
    mpo = build_mpo(model, dtype=dtype, device=device)
    mps, _camps, init = _make_initial_mps(model, method, dtype, device)
    initial_bond_dims = _bond_dims(mps)
    t0 = time.perf_counter()
    raw = train_ad_two_site(
        mps,
        mpo,
        num_sweeps=method.sweeps,
        local_steps=int(method.local_steps or 1),
        lr=float(method.lr if method.lr is not None else 1.0),
        optimizer=method.optimizer or "lbfgs",
        lbfgs_iters=int(method.lbfgs_iters or 5),
        lbfgs_tolerance_grad=float(method.lbfgs_tolerance_grad or 1e-12),
        lbfgs_tolerance_change=float(method.lbfgs_tolerance_change or 1e-15),
        max_bond_dim=method.chi,
        precondition=method.two_site_precondition,
        stabilization=method.post_step_stabilization,
    )
    runtime_s = time.perf_counter() - t0
    history = []
    for rec in raw["sweeps"]:
        energy = float(rec["energy_after"])
        history.append({
            "sweep": rec["sweep"],
            "direction": rec["direction"],
            "energy": energy,
            "energy_per_site": energy / model.N,
            "sector_report": _sector_report(model, mps),
            "bond_dims": raw["final_bond_dims"],
            "max_bond": rec["max_bond"],
            "max_trunc": rec["max_trunc"],
        })
    final_e = float(raw["final_energy"])
    return {
        "history": history,
        "summary": {
            "initial_energy": float(raw["initial_energy"]),
            "final_energy": final_e,
            "final_energy_per_site": final_e / model.N,
            "final_max_bond": raw["max_bond"],
            "initial_bond_dims": initial_bond_dims,
            "final_bond_dims": raw["final_bond_dims"],
            "chi_requested": method.chi,
            "initial_max_bond": max(initial_bond_dims) if initial_bond_dims else 1,
            "best_energy": min(raw["energy_history"]),
            "best_step": raw["energy_history"].index(min(raw["energy_history"])),
            "directional_sweeps": method.sweeps,
            "local_steps_per_bond": int(method.local_steps or 1),
            "optimizer_steps": raw["optimizer_steps"],
            "closure_evals": raw["closure_evals"],
            "runtime": runtime_s,
        },
        "diagnostics": {
            "algorithm_id": "ad_two_site",
            "optimizer_path": "two_site_ad_local_theta",
            "ed_used": False,
            "classical_dmrg_used": False,
            "lanczos_used": False,
            "ad_used": True,
            "dense_hamiltonian_built": False,
            "sector_mode": mode,
            "initialization": init,
            "projection": None,
            "two_site_precondition": method.two_site_precondition,
            "post_step_stabilization": method.post_step_stabilization,
            "max_forbidden_abs": None,
            "max_forbidden_grad_abs": None,
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
        elif name == "local_density_mid":
            if model.name == "spinless_tv":
                op = local_number_operator("spinless_tv", mps.dtype, mps.device)
            elif model.name == "hubbard":
                op = local_ntot_operator(mps.dtype, mps.device)
            else:
                raise NotImplementedError(f"observable {name!r} is not implemented for {model.name!r}")
            obs["local_density_mid"] = _local_expectation(mps, op, _mid_site(model))
        elif name == "double_occupancy_mid":
            if model.name != "hubbard":
                raise NotImplementedError(f"observable {name!r} is implemented only for Hubbard")
            op = hubbard_local_operators(dtype=mps.dtype, device=mps.device)["double_occ"]
            obs["double_occupancy_mid"] = _local_expectation(mps, op, _mid_site(model))
        elif name == "local_sz_mid":
            if model.name != "hubbard":
                raise NotImplementedError(f"observable {name!r} is implemented only for Hubbard")
            obs["local_sz_mid"] = _local_expectation(mps, local_sz_operator(mps.dtype, mps.device), _mid_site(model))
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

    alias_resolution = None
    if method.name == "ad_dmrg":
        alias_resolution = {"requested": "ad_dmrg", "resolved": "ad_global"}
        warnings.warn(
            "Method 'ad_dmrg' is deprecated; use 'ad_global' or 'ad_two_site'. "
            "The compatibility alias resolves to 'ad_global'.",
            DeprecationWarning,
            stacklevel=2,
        )
        data = method.to_dict()
        data["name"] = "ad_global"
        method = MethodConfig.from_dict(data)

    if method.name == "ad_global":
        raw = _run_ad(model, method, runtime, dtype, device)
    elif method.name == "ad_two_site":
        raw = _run_ad_two_site(model, method, runtime, dtype, device)
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
    if alias_resolution is not None:
        out["diagnostics"]["deprecated_alias_resolution"] = alias_resolution
        out["method"]["deprecated_alias_resolution"] = alias_resolution
    if runtime.output:
        from .config_schema import write_json
        write_json(out, runtime.output)
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
    cli_method = getattr(args, "method", "auto")
    if cli_method == "auto":
        method_name = "ad_two_site" if args.sector_mode == "none" else "ad_global"
    else:
        method_name = cli_method
    method = MethodConfig(
        name=method_name,
        chi=args.chi,
        sweeps=args.sweeps,
        optimizer=args.optimizer,
        local_steps=args.local_steps,
        lbfgs_iters=args.lbfgs_iters,
        lbfgs_tolerance_grad=getattr(args, "lbfgs_tolerance_grad", None),
        lbfgs_tolerance_change=getattr(args, "lbfgs_tolerance_change", None),
        lr=args.lr,
        sector_mode=args.sector_mode,
        initialization=args.init,
        projection=args.stabilization,
        two_site_precondition=getattr(args, "precondition", "theta_norm"),
        post_step_stabilization=args.stabilization,
        grad_clip=args.grad_clip,
        global_steps=(args.sweeps * args.local_steps if method_name == "ad_global" else None),
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
