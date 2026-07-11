#!/usr/bin/env python3
"""AD-mainline benchmark runner for open-boundary 1D MPO models.

This runner deliberately avoids exact diagonalization, dense Hamiltonian
construction, classical DMRG, and Lanczos. It is intended for large-system AD
benchmarks and small CPU/GPU smoke tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_two_site import ADTwoSiteOptimizer  # noqa: E402
from latticetn import contractions as K  # noqa: E402
from latticetn.initial_states import (  # noqa: E402
    neel_spin_state,
    spinless_half_filled_cdw_state,
    hubbard_half_filled_neel_state,
)
from latticetn.charge_sectors import (  # noqa: E402
    ChargeAwareMPS,
    apply_charge_masks_,
    zero_forbidden_gradients_,
    max_forbidden_abs,
    spinless_hard_sector_product_mps,
    hubbard_hard_sector_product_mps,
)
from latticetn.sector_observables import (  # noqa: E402
    total_particle_number,
    sector_leakage_report,
    total_nup,
    total_ndown,
    hubbard_sector_leakage_report,
)
from latticetn.model_spec import ModelSpec  # noqa: E402
from latticetn.runner import run_latticetn_job, namespace_from_legacy_ad_args  # noqa: E402


BETHE_HEISENBERG_E_INF = 0.25 - math.log(2.0)


def parse_dtype(name: str) -> tc.dtype:
    table = {"complex64": tc.complex64, "complex128": tc.complex128}
    return table[name]


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if tc.cuda.is_available() else "cpu"
    if name == "cuda" and not tc.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return name


def local_dim(model: str) -> int:
    return 4 if model == "hubbard" else 2


def default_init(model: str) -> str:
    if model in {"heisenberg", "tfi"}:
        return "neel"
    if model == "spinless_tv":
        return "spinless_cdw"
    if model == "hubbard":
        return "hubbard_neel"
    raise ValueError(f"unknown model {model!r}")


def build_mpo(args, dtype: tc.dtype, device: str) -> MPO:
    mpo = MPO.from_bonds(args.N, local_dim(args.model), dtype=dtype, device=device)
    if args.model == "heisenberg":
        return mpo.generate_heisenberg(J=args.J)
    if args.model == "tfi":
        return mpo.generate_tfi(J=args.J, h=args.h)
    if args.model == "spinless_tv":
        return mpo.generate_spinless_fermion(t=args.t, V=args.V, mu=args.mu)
    if args.model == "hubbard":
        return mpo.generate_hubbard(t=args.t, U=args.U, mu=args.mu, h=args.h)
    raise ValueError(f"unknown model {args.model!r}")


def make_random_mps(model: str, N: int, chi: int, dtype: tc.dtype, device: str, seed: int) -> MPS:
    tc.manual_seed(seed)
    mps = MPS(N, local_dim(model), chi, dtype=dtype, device=device)
    with tc.no_grad():
        for p in mps.tensors:
            n = p.norm()
            if n > 0:
                p.copy_(p / n)
    return mps


def make_mps(args, dtype: tc.dtype, device: str) -> tuple[MPS, str]:
    init = default_init(args.model) if args.init == "auto" else args.init
    if init == "random":
        return make_random_mps(args.model, args.N, args.chi, dtype, device, args.seed), init
    if init == "neel":
        return neel_spin_state(args.N, dtype=dtype, device=device), init
    if init == "spinless_cdw":
        if args.model != "spinless_tv":
            raise ValueError("spinless_cdw init is only valid for --model spinless_tv")
        return spinless_half_filled_cdw_state(args.N, dtype=dtype, device=device), init
    if init == "hubbard_neel":
        if args.model != "hubbard":
            raise ValueError("hubbard_neel init is only valid for --model hubbard")
        return hubbard_half_filled_neel_state(args.N, dtype=dtype, device=device), init
    raise ValueError(f"unsupported init {init!r}")


def _infer_spinless_target(args) -> int:
    if args.target_n is not None:
        return args.target_n
    if args.init == "auto" and args.N % 2 == 0:
        return args.N // 2
    raise ValueError("--sector-mode hard for spinless_tv requires --target-n")


def _infer_hubbard_targets(args) -> tuple[int, int]:
    if args.target_nup is not None and args.target_ndown is not None:
        return args.target_nup, args.target_ndown
    if args.init == "auto" and args.N % 2 == 0:
        return args.N // 2, args.N // 2
    raise ValueError("--sector-mode hard for hubbard requires --target-nup and --target-ndown")


def make_hard_charge_mps(args, dtype: tc.dtype, device: str) -> tuple[ChargeAwareMPS, str]:
    init = default_init(args.model) if args.init == "auto" else args.init
    if args.model == "spinless_tv":
        target_n = _infer_spinless_target(args)
        pattern = "cdw" if init in {"spinless_cdw", "auto"} else "left"
        return (
            spinless_hard_sector_product_mps(
                args.N, target_n=target_n, chi=args.chi, pattern=pattern,
                dtype=dtype, device=device,
            ),
            init,
        )
    if args.model == "hubbard":
        target_nup, target_ndown = _infer_hubbard_targets(args)
        pattern = "neel" if init in {"hubbard_neel", "auto"} else "balanced"
        return (
            hubbard_hard_sector_product_mps(
                args.N, target_nup=target_nup, target_ndown=target_ndown,
                chi=args.chi, pattern=pattern, dtype=dtype, device=device,
            ),
            init,
        )
    raise ValueError("--sector-mode hard is supported only for spinless_tv and hubbard")


def bond_dims(mps: MPS) -> list[int]:
    return [int(mps.tensors[i].shape[2]) for i in range(mps.N - 1)]


def max_bond(mps: MPS) -> int:
    dims = bond_dims(mps)
    return max(dims) if dims else 1


def tensor_to_float(x: tc.Tensor) -> float:
    return float(x.detach().real.cpu())


def current_energy(mps: MPS, mpo: MPO) -> tc.Tensor:
    e = K.rayleigh_energy_native(mps, mpo)
    if not tc.isfinite(e.detach()).all():
        raise FloatingPointError("non-finite Rayleigh energy")
    return e.real


def sector_report(args, mps: MPS) -> dict | None:
    if args.model == "spinless_tv" and args.target_n is not None:
        return sector_leakage_report(mps, target_n=args.target_n)
    if args.model == "hubbard" and args.target_nup is not None and args.target_ndown is not None:
        return hubbard_sector_leakage_report(
            mps, target_nup=args.target_nup, target_ndown=args.target_ndown
        )
    return None


def sector_penalty(args, mps: MPS) -> tc.Tensor:
    zero = tc.zeros((), dtype=tc.float64, device=mps.device)
    if args.model == "spinless_tv" and args.lambda_n != 0.0:
        if args.target_n is None:
            raise ValueError("--lambda-n requires --target-n")
        n = total_particle_number(mps, model="spinless")
        return args.lambda_n * (n - float(args.target_n)) ** 2
    if args.model == "hubbard":
        loss = zero
        if args.lambda_nup != 0.0:
            if args.target_nup is None:
                raise ValueError("--lambda-nup requires --target-nup")
            loss = loss + args.lambda_nup * (total_nup(mps) - float(args.target_nup)) ** 2
        if args.lambda_ndown != 0.0:
            if args.target_ndown is None:
                raise ValueError("--lambda-ndown requires --target-ndown")
            loss = loss + args.lambda_ndown * (total_ndown(mps) - float(args.target_ndown)) ** 2
        return loss
    return zero


def grad_norm(params, device: str) -> float:
    sq = tc.zeros((), dtype=tc.float64, device=device)
    seen = False
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq = sq + (g.conj() * g).real.sum().to(tc.float64)
        seen = True
    return float(sq.sqrt().cpu()) if seen else 0.0


def run_global_penalty_ad(args, mps: MPS, mpo: MPO) -> tuple[list[dict], float]:
    """Global AD fallback used when a sector penalty is enabled."""
    for p in mps.tensors:
        p.requires_grad_(True)
    params = list(mps.parameters())
    history = []
    max_grad = 0.0
    for sweep in range(args.sweeps):
        opt = tc.optim.Adam(params, lr=args.lr) if args.optimizer == "adam" else tc.optim.LBFGS(
            params, lr=args.lr, max_iter=args.lbfgs_iters, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = current_energy(mps, mpo) + sector_penalty(args, mps).real
            loss.backward()
            return loss

        steps = args.local_steps if args.optimizer == "adam" else max(1, args.local_steps)
        for _ in range(steps):
            if args.optimizer == "adam":
                loss = closure()
                if args.grad_clip is not None and args.grad_clip > 0:
                    tc.nn.utils.clip_grad_norm_(params, args.grad_clip)
                max_grad = max(max_grad, grad_norm(params, mps.device))
                opt.step()
            else:
                opt.step(closure)
                max_grad = max(max_grad, grad_norm(params, mps.device))
        e = tensor_to_float(current_energy(mps, mpo))
        history.append({
            "sweep": sweep,
            "energy": e,
            "energy_per_site": e / args.N,
            "sector_report": sector_report(args, mps),
            "max_bond": max_bond(mps),
            "max_grad_norm": max_grad,
        })
    return history, max_grad


def run_hard_sector_ad(args, camps: ChargeAwareMPS, mpo: MPO) -> tuple[list[dict], float, float]:
    """Global AD with hard charge masks applied to values and gradients."""

    mps = camps.mps
    apply_charge_masks_(mps, camps.masks)
    for p in mps.tensors:
        p.requires_grad_(True)
    params = list(mps.parameters())
    history = []
    max_grad = 0.0
    max_forbidden_grad = 0.0
    for sweep in range(args.sweeps):
        opt = tc.optim.Adam(params, lr=args.lr) if args.optimizer == "adam" else tc.optim.LBFGS(
            params, lr=args.lr, max_iter=args.lbfgs_iters, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            apply_charge_masks_(mps, camps.masks)
            loss = current_energy(mps, mpo)
            loss.backward()
            nonlocal max_forbidden_grad
            max_forbidden_grad = max(
                max_forbidden_grad,
                zero_forbidden_gradients_(params, camps.masks),
            )
            return loss

        steps = args.local_steps if args.optimizer == "adam" else max(1, args.local_steps)
        for _ in range(steps):
            if args.optimizer == "adam":
                closure()
                max_grad = max(max_grad, grad_norm(params, mps.device))
                if args.grad_clip is not None and args.grad_clip > 0:
                    tc.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                apply_charge_masks_(mps, camps.masks)
            else:
                opt.step(closure)
                max_grad = max(max_grad, grad_norm(params, mps.device))
                apply_charge_masks_(mps, camps.masks)
        e = tensor_to_float(current_energy(mps, mpo))
        history.append({
            "sweep": sweep,
            "energy": e,
            "energy_per_site": e / args.N,
            "sector_report": sector_report(args, mps),
            "max_bond": max_bond(mps),
            "max_grad_norm": max_grad,
            "max_forbidden_abs": max_forbidden_abs(mps, camps.masks),
            "max_forbidden_grad_abs": max_forbidden_grad,
            "split_strategy": camps.split_strategy,
        })
    return history, max_grad, max_forbidden_grad


def run_two_site_ad(args, mps: MPS, mpo: MPO) -> tuple[list[dict], float]:
    ad = ADTwoSiteOptimizer(mps, mpo, bond=0)
    history = []
    max_grad_seen = 0.0
    for sweep in range(args.sweeps):
        direction = "right" if sweep % 2 == 0 else "left"
        bonds = range(args.N - 1) if direction == "right" else range(args.N - 2, -1, -1)
        sweep_grad = 0.0
        sweep_trunc = 0.0
        for b in bonds:
            ad.reset_bond(b)
            params = list(ad.parameters())
            if args.optimizer == "adam":
                opt = tc.optim.Adam(params, lr=args.lr)
                for _ in range(args.local_steps):
                    opt.zero_grad(set_to_none=True)
                    loss = ad.energy()
                    loss.backward()
                    sweep_grad = max(sweep_grad, grad_norm(params, ad.device))
                    if args.grad_clip is not None and args.grad_clip > 0:
                        tc.nn.utils.clip_grad_norm_(params, args.grad_clip)
                    opt.step()
            else:
                opt = tc.optim.LBFGS(
                    params, lr=args.lr, max_iter=args.lbfgs_iters, line_search_fn="strong_wolfe"
                )

                def closure():
                    opt.zero_grad(set_to_none=True)
                    loss = ad.energy()
                    loss.backward()
                    return loss

                for _ in range(max(1, args.local_steps)):
                    opt.step(closure)
                    sweep_grad = max(sweep_grad, grad_norm(params, ad.device))
            trunc, _ = ad.split(max_bond_dim=args.chi, direction=direction)
            sweep_trunc = max(sweep_trunc, float(trunc))
        max_grad_seen = max(max_grad_seen, sweep_grad)
        e = tensor_to_float(current_energy(ad.mps, mpo))
        history.append({
            "sweep": sweep,
            "direction": direction,
            "energy": e,
            "energy_per_site": e / args.N,
            "sector_report": sector_report(args, ad.mps),
            "max_bond": max_bond(ad.mps),
            "max_trunc": sweep_trunc,
            "max_grad_norm": sweep_grad,
        })
    return history, max_grad_seen


def cuda_memory(device: str) -> dict:
    if not device.startswith("cuda") or not tc.cuda.is_available():
        return {}
    dev = tc.device(device)
    return {
        "allocated": tc.cuda.memory_allocated(dev),
        "reserved": tc.cuda.memory_reserved(dev),
        "peak_allocated": tc.cuda.max_memory_allocated(dev),
    }


def run_ad_model_benchmark(args: argparse.Namespace) -> dict:
    if not args.no_ed:
        raise ValueError("AD benchmark runner requires --no-ed")
    if args.model_spec_json is not None:
        model_data = json.loads(args.model_spec_json.read_text(encoding="utf-8"))
        model, method, runtime, obs = namespace_from_legacy_ad_args(args)
        model = ModelSpec.from_dict(model_data)
        unified = run_latticetn_job(model, method, runtime, obs)
        legacy = _legacy_result_from_unified(unified)
        print("latticeTN AD model benchmark")
        print(f"model={legacy['model']} N={legacy['N']} chi={legacy['chi']} sweeps={legacy['sweeps']}")
        print("ED status = skipped by design")
        print("classical DMRG/Lanczos = not used")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
        return legacy

    model_schema, method_schema, runtime_schema, obs_schema = namespace_from_legacy_ad_args(args)
    unified = run_latticetn_job(model_schema, method_schema, runtime_schema, obs_schema)
    legacy = _legacy_result_from_unified(unified)

    print("latticeTN AD model benchmark")
    print(f"model={legacy['model']} N={legacy['N']} chi={legacy['chi']} sweeps={legacy['sweeps']}")
    print(
        f"device={legacy['device']} dtype={legacy['dtype']} "
        f"optimizer={legacy['optimizer']} init={legacy['init']}"
    )
    print(f"algorithm = {legacy['algorithm_id']} ({legacy['optimizer_path']})")
    print(f"sector mode = {legacy['sector_mode']}")
    print("ED status = skipped by design")
    print("classical DMRG/Lanczos = not used")
    for rec in legacy["history"]:
        print(
            f"sweep={rec['sweep']} E={rec['energy']:.12f} "
            f"E/site={rec['energy_per_site']:.12f} max_bond={rec.get('max_bond')} "
            f"max_grad={rec.get('gradient_norm', rec.get('max_grad_norm', 0.0)):.3e}"
        )
        if rec.get("sector_report") is not None:
            print(f"  sector report = {rec['sector_report']}")
        if legacy["sector_mode"] == "hard":
            print(
                f"  hard sector: max_forbidden_abs={rec.get('max_forbidden_abs', 0.0):.3e} "
                f"max_forbidden_grad_abs={rec.get('max_forbidden_grad_abs', 0.0):.3e}"
            )
    print(f"final energy = {legacy['final_energy']:.12f}")
    print(f"final E/site = {legacy['final_energy_per_site']:.12f}")
    print(f"final max bond = {legacy['final_max_bond']}")
    print(f"final max grad norm = {legacy['final_max_grad_norm']:.6e}")
    print(f"runtime = {legacy['runtime']:.3f} s")
    print("ED status = skipped by design")
    print("classical DMRG/Lanczos = not used")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
    return legacy


def _legacy_result_from_unified(unified: dict) -> dict:
    summary = unified["summary"]
    diagnostics = unified["diagnostics"]
    model = unified["model"]
    method = unified["method"]
    runtime = unified["runtime"]
    history = unified["sweep_history"]
    final_sector = history[-1].get("sector_report") if history else None
    optimizer_path = diagnostics.get("optimizer_path", "unknown")
    algorithm_id = diagnostics.get("algorithm_id", method["name"])
    return {
        "model": model["name"],
        "N": model["N"],
        "chi": method["chi"],
        "chi_requested": method["chi"],
        "initial_bond_dims": summary.get("initial_bond_dims"),
        "initial_max_bond": summary.get("initial_max_bond"),
        "final_bond_dims": summary.get("final_bond_dims"),
        "sweeps": method["sweeps"],
        "device": runtime["resolved_device"],
        "dtype": runtime["dtype"],
        "optimizer": method["optimizer"],
        "algorithm_id": algorithm_id,
        "optimizer_path": optimizer_path,
        "deprecated_alias_resolution": diagnostics.get("deprecated_alias_resolution"),
        "sector_mode": method["sector_mode"],
        "init": method.get("initialization", diagnostics.get("initialization", "auto")),
        "projection": method.get("projection"),
        "canonical_interval": method.get("canonical_interval"),
        "canonicalization_method": method.get("canonicalization_method"),
        "raw_norm_before_projection": summary.get("raw_norm_before_projection"),
        "physical_norm_after_projection": summary.get("physical_norm_after_projection"),
        "canonical_residual": summary.get("canonical_residual"),
        "optimizer_reset_events": summary.get("optimizer_reset_events"),
        "stabilization": method.get("post_step_stabilization"),
        "two_site_precondition": method.get("two_site_precondition"),
        "grad_clip": method.get("grad_clip"),
        "lr": method.get("lr"),
        "lbfgs_tolerance_grad": method.get("lbfgs_tolerance_grad"),
        "lbfgs_tolerance_change": method.get("lbfgs_tolerance_change"),
        "global_steps": summary.get("global_steps"),
        "directional_sweeps": summary.get("directional_sweeps"),
        "local_steps_per_bond": summary.get("local_steps_per_bond"),
        "optimizer_steps": summary.get("optimizer_steps"),
        "closure_evals": summary.get("closure_evals"),
        "best_energy": summary.get("best_energy"),
        "best_step": summary.get("best_step"),
        "initial_energy": summary.get("initial_energy", history[0]["energy"] if history else summary["final_energy"]),
        "initial_energy_per_site": (
            summary.get("initial_energy", history[0]["energy"] if history else summary["final_energy"]) / model["N"]
        ),
        "initial_sector_report": history[0].get("sector_report") if history else None,
        "history": history,
        "final_energy": summary["final_energy"],
        "final_energy_per_site": summary["final_energy_per_site"],
        "final_sector_report": final_sector,
        "final_max_bond": summary["final_max_bond"],
        "final_max_grad_norm": summary.get("final_gradient_norm", 0.0),
        "final_max_forbidden_abs": diagnostics.get("max_forbidden_abs"),
        "final_max_forbidden_grad_abs": diagnostics.get("max_forbidden_grad_abs"),
        "hard_sector_split_strategy": "dense_global_ad_masked" if method["sector_mode"] == "hard" else None,
        "runtime": summary["runtime"],
        "gpu_memory": {},
        "ed_status": "skipped by design",
        "classical_dmrg_lanczos": "not used",
        "dense_hamiltonian_built": diagnostics["dense_hamiltonian_built"],
        "dmrg_lanczos_used": False,
        "heisenberg_bethe_e_inf": BETHE_HEISENBERG_E_INF if model["name"] == "heisenberg" else None,
        "stage10_result": unified,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["heisenberg", "tfi", "spinless_tv", "hubbard"], required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--chi", type=int, required=True)
    p.add_argument("--sweeps", type=int, default=1)
    p.add_argument("--method", choices=["auto", "ad_two_site", "ad_global"], default="auto")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex128")
    p.add_argument("--init", choices=["auto", "neel", "random", "spinless_cdw", "hubbard_neel"], default="auto")
    p.add_argument("--optimizer", choices=["lbfgs", "adam"], default="adam")
    p.add_argument("--local-steps", type=int, default=1)
    p.add_argument("--lbfgs-iters", type=int, default=5)
    p.add_argument("--lbfgs-tolerance-grad", type=float, default=None)
    p.add_argument("--lbfgs-tolerance-change", type=float, default=None)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--precondition", choices=["theta_norm", "none"], default="theta_norm")
    p.add_argument(
        "--stabilization",
        choices=["tensor_norm", "none", "canonical", "sector_canonical"],
        default="none",
    )
    p.add_argument("--canonical-interval", type=int, default=1)
    p.add_argument("--canonicalization-method", choices=["qr", "svd"], default="qr")
    p.add_argument(
        "--reset-optimizer-on-canonicalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--normalize-final-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--sector-mode", choices=["none", "soft", "hard"], default="none")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--model-spec-json", type=Path, default=None)
    p.add_argument("--no-ed", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=1.0)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--V", type=float, default=0.0)
    p.add_argument("--U", type=float, default=4.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--target-n", type=int, default=None)
    p.add_argument("--lambda-n", type=float, default=0.0)
    p.add_argument("--target-nup", type=int, default=None)
    p.add_argument("--target-ndown", type=int, default=None)
    p.add_argument("--lambda-nup", type=float, default=0.0)
    p.add_argument("--lambda-ndown", type=float, default=0.0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    run_ad_model_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
