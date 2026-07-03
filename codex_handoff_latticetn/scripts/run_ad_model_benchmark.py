#!/usr/bin/env python3
"""Generic AD-mainline benchmark runner for latticeTN 1D MPO models.

This script is intentionally AD-only:
  - uses latticetn.ad_two_site.ADTwoSiteOptimizer;
  - optimizes the differentiable two-site Rayleigh quotient with loss.backward()
    and a torch optimizer;
  - uses SVD only as the post-step split/compression;
  - does NOT import dmrg.py or lanczos.py;
  - does NOT build a dense Hamiltonian / exact diagonalization reference.

Supported models:
  heisenberg, tfi, spinless_tv, hubbard.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Literal

# Keep CPU bookkeeping from over-threading. This is especially useful on Windows
# when many small tensor contractions are launched repeatedly.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "latticetn").is_dir():
    sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_two_site import ADTwoSiteOptimizer  # noqa: E402
from latticetn import contractions as K  # noqa: E402

BETHE_HEISENBERG_E_INF = 0.25 - math.log(2.0)


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def parse_dtype(name: str) -> tc.dtype:
    table = {
        "complex64": tc.complex64,
        "complex128": tc.complex128,
        "float32": tc.float32,
        "float64": tc.float64,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {name!r}; choose one of {sorted(table)}") from exc


def infer_local_dim(model: str) -> int:
    return 4 if model == "hubbard" else 2


def default_init_for_model(model: str) -> str:
    if model in {"heisenberg", "tfi"}:
        return "neel"
    if model == "spinless_tv":
        return "cdw"
    if model == "hubbard":
        return "hubbard_neel"
    raise ValueError(f"unknown model {model!r}")


# ---------------------------------------------------------------------------
# Model and initialization builders
# ---------------------------------------------------------------------------

def build_mpo(
    model: str,
    N: int,
    dtype: tc.dtype,
    device: str,
    J: float,
    h: float,
    t: float,
    V: float,
    U: float,
    mu: float,
    field: float,
) -> MPO:
    """Build a supported open-boundary MPO without constructing a dense H."""
    dim = infer_local_dim(model)
    mpo = MPO.from_bonds(N, dim, dtype=dtype, device=device)
    if model == "heisenberg":
        return mpo.generate_heisenberg(J=J)
    if model == "tfi":
        return mpo.generate_tfi(J=J, h=h)
    if model == "spinless_tv":
        return mpo.generate_spinless_fermion(t=t, V=V, mu=mu)
    if model == "hubbard":
        return mpo.generate_hubbard(t=t, U=U, mu=mu, h=field)
    raise ValueError(f"unknown model {model!r}")


def product_phys_index(model: str, init: str, site: int) -> int:
    """Return the physical index for a product-state MPS tensor."""
    if init == "auto":
        init = default_init_for_model(model)

    if model == "hubbard":
        # Hubbard basis: 0=|0>, 1=|up>, 2=|down>, 3=|up,down>.
        if init in {"hubbard_neel", "neel"}:
            return 1 if site % 2 == 0 else 2
        if init == "empty":
            return 0
        if init == "all_up":
            return 1
        if init == "all_down":
            return 2
        if init == "doublon":
            return 3
        raise ValueError(
            f"init={init!r} not supported for hubbard; choose auto, hubbard_neel, "
            "neel, empty, all_up, all_down, doublon, or random"
        )

    # Spin / spinless local dimension 2. For spin models, 0=up and 1=down.
    # For spinless fermions, 0=empty and 1=occupied, so cdw is 1010... by index.
    if init == "neel":
        return 0 if site % 2 == 0 else 1
    if init == "all_up":
        return 0
    if init == "all_down":
        return 1
    if init == "empty":
        return 0
    if init == "cdw":
        return 1 if site % 2 == 0 else 0
    raise ValueError(
        f"init={init!r} not supported for model={model}; choose auto, neel, "
        "all_up, all_down, empty, cdw, or random"
    )


def make_product_mps(
    model: str,
    N: int,
    init: str,
    dtype: tc.dtype,
    device: str,
) -> MPS:
    dim = infer_local_dim(model)
    tensors = []
    for i in range(N):
        A = tc.zeros((1, dim, 1), dtype=dtype, device=device)
        phys = product_phys_index(model, init, i)
        A[0, phys, 0] = 1.0
        tensors.append(A)
    return MPS.from_tensors(tensors, dtype=dtype, device=device, requires_grad=False)


def make_random_mps(model: str, N: int, chi: int, dtype: tc.dtype, device: str, seed: int) -> MPS:
    """Random MPS fallback. Product states are usually more stable for large N."""
    tc.manual_seed(seed)
    dim = infer_local_dim(model)
    mps = MPS(N, dim, chi, dtype=dtype, device=device)
    with tc.no_grad():
        for i in range(N):
            n = mps.tensors[i].norm()
            if n > 0:
                mps.tensors[i].data = (mps.tensors[i] / n).to(dtype).contiguous().data
    return mps


def make_mps(model: str, N: int, chi: int, init: str, dtype: tc.dtype, device: str, seed: int) -> MPS:
    if init == "random":
        return make_random_mps(model, N, chi=chi, dtype=dtype, device=device, seed=seed)
    return make_product_mps(model, N, init=init, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Diagnostics and AD local optimization
# ---------------------------------------------------------------------------

def bond_dims(mps: MPS) -> list[int]:
    return [int(mps.tensors[i].shape[2]) for i in range(mps.N - 1)]


def max_bond(mps: MPS) -> int:
    bd = bond_dims(mps)
    return max(bd) if bd else 1


def finite_or_raise(x: tc.Tensor, where: str) -> None:
    if not tc.isfinite(x.detach()).all().item():
        raise FloatingPointError(f"non-finite tensor encountered at {where}")


def grad_norm(params, device) -> float:
    sq = tc.zeros((), dtype=tc.float64, device=device)
    seen = False
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.is_complex():
            sq = sq + (g.conj() * g).real.sum().to(dtype=tc.float64)
        else:
            sq = sq + (g * g).sum().to(dtype=tc.float64)
        seen = True
    if not seen:
        return 0.0
    return float(sq.sqrt().detach().cpu())


def stabilize_theta_unit_norm(adtso: ADTwoSiteOptimizer) -> None:
    """Scale-invariant post-step stabilization outside the loss graph."""
    with tc.no_grad():
        n = adtso.theta.norm()
        if n > 0:
            adtso.theta.data = (adtso.theta / n).to(adtso.theta.dtype).contiguous().data


def local_optimize(
    adtso: ADTwoSiteOptimizer,
    bond: int,
    direction: Literal["right", "left"],
    optimizer_name: Literal["adam", "lbfgs"],
    local_steps: int,
    lr: float,
    lbfgs_iters: int,
    max_bond_dim: int | None,
    cutoff: float | None,
    stabilization: Literal["none", "tensor_norm"],
    grad_clip: float | None,
) -> dict:
    """Optimize one two-site center tensor by AD, then SVD-split it."""
    adtso.reset_bond(bond)
    params = list(adtso.parameters())
    last_loss = None
    last_grad = 0.0

    if optimizer_name == "adam":
        opt = tc.optim.Adam(params, lr=lr)
        for _ in range(local_steps):
            opt.zero_grad(set_to_none=True)
            loss = adtso.energy()
            finite_or_raise(loss, f"bond {bond} Adam loss")
            loss.backward()
            last_grad = grad_norm(params, adtso.theta.device)
            if grad_clip is not None and grad_clip > 0:
                tc.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            if stabilization == "tensor_norm":
                stabilize_theta_unit_norm(adtso)
            last_loss = float(loss.detach().cpu())
    elif optimizer_name == "lbfgs":
        opt = tc.optim.LBFGS(
            params,
            lr=lr,
            max_iter=lbfgs_iters,
            line_search_fn="strong_wolfe",
        )
        # For LBFGS, local_steps means the number of opt.step(closure) calls;
        # each call internally evaluates the closure up to lbfgs_iters times.
        for _ in range(max(1, local_steps)):
            def closure():
                opt.zero_grad(set_to_none=True)
                loss = adtso.energy()
                finite_or_raise(loss, f"bond {bond} LBFGS closure loss")
                loss.backward()
                return loss

            loss_out = opt.step(closure)
            last_grad = grad_norm(params, adtso.theta.device)
            try:
                last_loss = float(tc.as_tensor(loss_out).detach().cpu())
            except Exception:
                last_loss = float("nan")
            if stabilization == "tensor_norm":
                stabilize_theta_unit_norm(adtso)
    else:
        raise ValueError(f"unknown optimizer {optimizer_name!r}")

    trunc, kept = adtso.split(
        max_bond_dim=max_bond_dim,
        cutoff=cutoff,
        direction=direction,
    )
    return {
        "bond": bond,
        "direction": direction,
        "local_loss": last_loss,
        "grad_norm": last_grad,
        "truncation": float(trunc),
        "kept_bond": int(kept),
    }


def current_energy(mps: MPS, mpo: MPO) -> float:
    e = K.rayleigh_energy_native(mps, mpo)
    finite_or_raise(e, "global Rayleigh energy")
    return float(e.detach().real.cpu())


def cuda_memory_summary(device: str) -> dict:
    if not str(device).startswith("cuda") or not tc.cuda.is_available():
        return {}
    dev = tc.device(device)
    return {
        "allocated_gb": tc.cuda.memory_allocated(dev) / 1024**3,
        "reserved_gb": tc.cuda.memory_reserved(dev) / 1024**3,
        "peak_allocated_gb": tc.cuda.max_memory_allocated(dev) / 1024**3,
    }


def format_cuda_memory(device: str) -> str:
    m = cuda_memory_summary(device)
    if not m:
        return ""
    return (
        f"allocated={m['allocated_gb']:.3f} GB "
        f"reserved={m['reserved_gb']:.3f} GB "
        f"peak={m['peak_allocated_gb']:.3f} GB"
    )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run_ad_model_benchmark(args: argparse.Namespace) -> dict:
    dtype = parse_dtype(args.dtype)
    if args.device.startswith("cuda") and not tc.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")

    tc.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        tc.cuda.reset_peak_memory_stats(tc.device(args.device))

    init = default_init_for_model(args.model) if args.init == "auto" else args.init
    mpo = build_mpo(
        model=args.model,
        N=args.N,
        dtype=dtype,
        device=args.device,
        J=args.J,
        h=args.h,
        t=args.t,
        V=args.V,
        U=args.U,
        mu=args.mu,
        field=args.field,
    )
    mps = make_mps(
        model=args.model,
        N=args.N,
        chi=args.chi,
        init=init,
        dtype=dtype,
        device=args.device,
        seed=args.seed,
    )
    adtso = ADTwoSiteOptimizer(mps, mpo, bond=0)

    print("=" * 96)
    print("latticeTN AD-mainline model benchmark, NO ED / NO classical DMRG / NO Lanczos")
    print(f"model={args.model} N={args.N} chi={args.chi} sweeps={args.sweeps} init={init}")
    print(
        f"params: J={args.J} h={args.h} t={args.t} V={args.V} "
        f"U={args.U} mu={args.mu} field={args.field}"
    )
    print(
        f"optimizer={args.optimizer} local_steps={args.local_steps} "
        f"lr={args.lr} lbfgs_iters={args.lbfgs_iters}"
    )
    print(
        f"max_bond_dim={args.chi} cutoff={args.cutoff} "
        f"stabilization={args.stabilization} grad_clip={args.grad_clip}"
    )
    print(f"device={args.device} dtype={dtype} torch={tc.__version__}")
    if args.device.startswith("cuda"):
        print(f"GPU: {tc.cuda.get_device_name(tc.device(args.device))}")
    if args.model == "heisenberg":
        print(f"Heisenberg Bethe e_inf = 1/4 - ln(2) = {BETHE_HEISENBERG_E_INF:.16f}")
        print("Finite OBC E/N is not expected to equal e_inf exactly.")
    print("ED status: skipped by design; this script never builds a dense Hamiltonian.")
    print("AD path: loss.backward() on two-site Rayleigh quotient; SVD is post-step split only.")
    print("classical DMRG/Lanczos: not imported, not used.")
    print("=" * 96)

    t_all = time.perf_counter()
    e0 = current_energy(adtso.mps, mpo)
    initial = {
        "sweep": -1,
        "direction": "init",
        "energy": e0,
        "energy_per_site": e0 / args.N,
        "delta_energy": 0.0,
        "delta_to_heisenberg_bethe_per_site": (
            e0 / args.N - BETHE_HEISENBERG_E_INF if args.model == "heisenberg" else None
        ),
        "max_bond": max_bond(adtso.mps),
        "max_trunc": 0.0,
        "max_grad_norm": 0.0,
        "runtime_s": 0.0,
        "bond_dims": bond_dims(adtso.mps),
    }
    history = [initial]
    print(
        f"init      E={e0:.12f} E/N={e0/args.N:.12f} "
        f"max_bond={initial['max_bond']}"
    )

    for s in range(args.sweeps):
        direction = "right" if s % 2 == 0 else "left"
        bonds = range(args.N - 1) if direction == "right" else range(args.N - 2, -1, -1)
        t0 = time.perf_counter()
        bond_reports = []
        for count, b in enumerate(bonds, start=1):
            br = local_optimize(
                adtso=adtso,
                bond=b,
                direction=direction,  # type: ignore[arg-type]
                optimizer_name=args.optimizer,  # type: ignore[arg-type]
                local_steps=args.local_steps,
                lr=args.lr,
                lbfgs_iters=args.lbfgs_iters,
                max_bond_dim=args.chi,
                cutoff=args.cutoff,
                stabilization=args.stabilization,  # type: ignore[arg-type]
                grad_clip=args.grad_clip,
            )
            bond_reports.append(br)
            if args.print_bonds:
                print(
                    f"  bond={b:03d} {direction:5s} "
                    f"local_loss={br['local_loss']:.12f} grad={br['grad_norm']:.3e} "
                    f"kept={br['kept_bond']:3d} trunc={br['truncation']:.2e}"
                )
            elif count % args.progress_every == 0 or count == args.N - 1:
                print(
                    f"  sweep {s:02d} {direction:5s}: "
                    f"finished {count:3d}/{args.N - 1} bonds",
                    flush=True,
                )

        e = current_energy(adtso.mps, mpo)
        dt = time.perf_counter() - t0
        bd = bond_dims(adtso.mps)
        max_trunc = max((x["truncation"] for x in bond_reports), default=0.0)
        max_grad = max((x["grad_norm"] for x in bond_reports), default=0.0)
        rec = {
            "sweep": s,
            "direction": direction,
            "energy": e,
            "energy_per_site": e / args.N,
            "delta_energy": e - history[-1]["energy"],
            "delta_to_heisenberg_bethe_per_site": (
                e / args.N - BETHE_HEISENBERG_E_INF if args.model == "heisenberg" else None
            ),
            "max_bond": max(bd) if bd else 1,
            "bond_dims": bd,
            "max_trunc": max_trunc,
            "max_grad_norm": max_grad,
            "runtime_s": dt,
            "bond_reports": bond_reports if args.store_bond_reports else None,
        }
        history.append(rec)
        bethe_str = ""
        if args.model == "heisenberg":
            bethe_str = f" dE/N_to_Bethe={rec['delta_to_heisenberg_bethe_per_site']:+.3e}"
        print(
            f"sweep={s:02d} dir={direction:5s} "
            f"E={e:.12f} E/N={e/args.N:.12f} dE={rec['delta_energy']:+.3e}"
            f"{bethe_str} max_trunc={max_trunc:.2e} max_grad={max_grad:.2e} "
            f"max_bond={rec['max_bond']} time={dt:.2f}s"
        )
        if args.print_bond_dims:
            print(f"  bond_dims = {bd}")
        mem = format_cuda_memory(args.device)
        if mem:
            print(f"  GPU memory: {mem}")

    total = time.perf_counter() - t_all
    final = history[-1]
    print("=" * 96)
    print(f"FINAL E                 = {final['energy']:.16f}")
    print(f"FINAL E / site          = {final['energy_per_site']:.16f}")
    if args.N > 1:
        print(f"FINAL E / bond          = {final['energy'] / (args.N - 1):.16f}")
    if args.model == "heisenberg":
        print(
            "E/site - Bethe e_inf    = "
            f"{final['delta_to_heisenberg_bethe_per_site']:.16f}"
        )
    print(f"final max trunc         = {final['max_trunc']:.3e}")
    print(f"final max grad norm     = {final['max_grad_norm']:.3e}")
    print(f"final max bond          = {final['max_bond']}")
    print(f"total runtime           = {total:.3f} s")
    mem = format_cuda_memory(args.device)
    if mem:
        print(f"final GPU memory: {mem}")
    print("ED status               = skipped by design")
    print("classical DMRG/Lanczos  = not used")
    print("=" * 96)

    result = {
        "model": args.model,
        "N": args.N,
        "chi": args.chi,
        "sweeps": args.sweeps,
        "device": args.device,
        "dtype": str(dtype),
        "init": init,
        "params": {
            "J": args.J,
            "h": args.h,
            "t": args.t,
            "V": args.V,
            "U": args.U,
            "mu": args.mu,
            "field": args.field,
        },
        "optimizer": args.optimizer,
        "local_steps": args.local_steps,
        "lr": args.lr,
        "lbfgs_iters": args.lbfgs_iters,
        "cutoff": args.cutoff,
        "stabilization": args.stabilization,
        "grad_clip": args.grad_clip,
        "history": history,
        "final_energy": final["energy"],
        "final_energy_per_site": final["energy_per_site"],
        "final_bond_dims": bond_dims(adtso.mps),
        "total_runtime_s": total,
        "cuda_memory": cuda_memory_summary(args.device),
        "references": {
            "heisenberg_bethe_e_inf": BETHE_HEISENBERG_E_INF if args.model == "heisenberg" else None,
            "finite_obc_not_equal_thermodynamic_limit": True,
        },
        "ed_skipped": True,
        "dense_hamiltonian_built": False,
        "dmrg_lanczos_used": False,
        "ad_mainline": True,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"JSON written to: {args.output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="heisenberg",
                   choices=["heisenberg", "tfi", "spinless_tv", "hubbard"])
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--chi", type=int, default=64)
    p.add_argument("--sweeps", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="complex64",
                   choices=["complex64", "complex128", "float32", "float64"])
    p.add_argument("--init", type=str, default="auto",
                   choices=["auto", "neel", "all_up", "all_down", "cdw", "hubbard_neel",
                            "empty", "doublon", "random"])
    p.add_argument("--optimizer", type=str, default="lbfgs", choices=["adam", "lbfgs"])
    p.add_argument("--local-steps", type=int, default=1,
                   help="Adam: gradient steps per bond. LBFGS: optimizer.step calls per bond.")
    p.add_argument("--lbfgs-iters", type=int, default=10,
                   help="Internal LBFGS iterations per optimizer.step call.")
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--cutoff", type=float, default=None)
    p.add_argument("--stabilization", type=str, default="tensor_norm",
                   choices=["none", "tensor_norm"])
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--print-bonds", action="store_true")
    p.add_argument("--print-bond-dims", action="store_true")
    p.add_argument("--store-bond-reports", action="store_true")
    p.add_argument("--progress-every", type=int, default=20)

    # Model parameters. Unused parameters are stored in JSON but ignored by the
    # corresponding MPO generator.
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=1.0,
                   help="TFI transverse field h. For Hubbard Zeeman field use --field.")
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--V", type=float, default=0.0)
    p.add_argument("--U", type=float, default=4.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--field", type=float, default=0.0,
                   help="Hubbard spin field parameter passed as h to generate_hubbard.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run_ad_model_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
