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
from latticetn.sector_observables import (  # noqa: E402
    total_particle_number,
    sector_leakage_report,
    total_nup,
    total_ndown,
    hubbard_sector_leakage_report,
)


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
        raise ValueError("Stage 8 benchmark runner requires --no-ed")
    dtype = parse_dtype(args.dtype)
    device = resolve_device(args.device)
    if device.startswith("cuda"):
        tc.cuda.reset_peak_memory_stats(tc.device(device))
    tc.manual_seed(args.seed)

    mpo = build_mpo(args, dtype, device)
    mps, init = make_mps(args, dtype, device)
    initial_energy = tensor_to_float(current_energy(mps, mpo))
    initial_sector = sector_report(args, mps)
    use_penalty_path = (
        (args.model == "spinless_tv" and args.lambda_n != 0.0)
        or (args.model == "hubbard" and (args.lambda_nup != 0.0 or args.lambda_ndown != 0.0))
    )

    print("latticeTN AD model benchmark")
    print(f"model={args.model} N={args.N} chi={args.chi} sweeps={args.sweeps}")
    print(f"device={device} dtype={args.dtype} optimizer={args.optimizer} init={init}")
    if device.startswith("cuda"):
        print(f"GPU: {tc.cuda.get_device_name(tc.device(device))}")
    print("ED status = skipped by design")
    print("classical DMRG/Lanczos = not used")
    print(f"initial energy = {initial_energy:.12f}")
    if initial_sector is not None:
        print(f"initial sector report = {initial_sector}")

    t0 = time.perf_counter()
    if use_penalty_path:
        history, max_grad = run_global_penalty_ad(args, mps, mpo)
        optimizer_path = "global_ad_with_sector_penalty"
    else:
        history, max_grad = run_two_site_ad(args, mps, mpo)
        optimizer_path = "two_site_ad"
    runtime = time.perf_counter() - t0

    final_energy = history[-1]["energy"] if history else initial_energy
    final_sector = sector_report(args, mps)
    for rec in history:
        print(
            f"sweep={rec['sweep']} E={rec['energy']:.12f} "
            f"E/site={rec['energy_per_site']:.12f} max_bond={rec['max_bond']} "
            f"max_grad={rec['max_grad_norm']:.3e}"
        )
        if rec.get("sector_report") is not None:
            print(f"  sector report = {rec['sector_report']}")
    print(f"final energy = {final_energy:.12f}")
    print(f"final E/site = {final_energy / args.N:.12f}")
    print(f"final max bond = {max_bond(mps)}")
    print(f"final max grad norm = {max_grad:.6e}")
    print(f"runtime = {runtime:.3f} s")
    print("ED status = skipped by design")
    print("classical DMRG/Lanczos = not used")

    result = {
        "model": args.model,
        "N": args.N,
        "chi": args.chi,
        "sweeps": args.sweeps,
        "device": device,
        "dtype": args.dtype,
        "optimizer": args.optimizer,
        "optimizer_path": optimizer_path,
        "init": init,
        "initial_energy": initial_energy,
        "initial_energy_per_site": initial_energy / args.N,
        "initial_sector_report": initial_sector,
        "history": history,
        "final_energy": final_energy,
        "final_energy_per_site": final_energy / args.N,
        "final_sector_report": final_sector,
        "final_max_bond": max_bond(mps),
        "final_max_grad_norm": max_grad,
        "runtime": runtime,
        "gpu_memory": cuda_memory(device),
        "ed_status": "skipped by design",
        "classical_dmrg_lanczos": "not used",
        "dense_hamiltonian_built": False,
        "dmrg_lanczos_used": False,
        "heisenberg_bethe_e_inf": BETHE_HEISENBERG_E_INF if args.model == "heisenberg" else None,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["heisenberg", "tfi", "spinless_tv", "hubbard"], required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--chi", type=int, required=True)
    p.add_argument("--sweeps", type=int, default=1)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex128")
    p.add_argument("--init", choices=["auto", "neel", "random", "spinless_cdw", "hubbard_neel"], default="auto")
    p.add_argument("--optimizer", choices=["lbfgs", "adam"], default="adam")
    p.add_argument("--local-steps", type=int, default=1)
    p.add_argument("--lbfgs-iters", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--stabilization", choices=["tensor_norm", "none"], default="none")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--output", type=Path, default=None)
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
