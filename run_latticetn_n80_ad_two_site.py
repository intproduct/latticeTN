#!/usr/bin/env python3
"""Large-N two-site AD local-tensor optimization runner for latticeTN.

This script is intentionally AD-only:
  - uses latticetn.ad_two_site.ADTwoSiteOptimizer;
  - optimizes the differentiable two-site Rayleigh quotient with loss.backward()
    and a torch optimizer;
  - uses SVD only as the post-step split/compression;
  - does NOT import dmrg.py or lanczos.py;
  - does NOT build a dense Hamiltonian / ED reference.

Default test case: N=80 open-boundary spin-1/2 Heisenberg chain, Neel product
state initialization, two-site AD sweeps with LBFGS.
"""

from __future__ import annotations

import argparse
import json
import math
import os

# Keep CPU bookkeeping from over-threading; must be set before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import time
from pathlib import Path
from typing import Literal

import torch as tc

# Allow running from the latticeTN project root without installing.
ROOT = Path(__file__).resolve().parent
if (ROOT / "latticetn").is_dir():
    sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_two_site import ADTwoSiteOptimizer  # noqa: E402
from latticetn import contractions as K  # noqa: E402

BETHE_E_INF = 0.25 - math.log(2.0)


def parse_dtype(name: str) -> tc.dtype:
    table = {
        "complex64": tc.complex64,
        "complex128": tc.complex128,
        "float32": tc.float32,
        "float64": tc.float64,
    }
    try:
        return table[name.lower()]
    except KeyError as e:
        raise ValueError(f"unknown dtype {name!r}; choose one of {sorted(table)}") from e


def make_product_mps(
    N: int,
    init: Literal["neel", "all_up", "all_down"],
    dtype: tc.dtype,
    device: str,
) -> MPS:
    """Create an open-boundary product-state MPS with bond dimension 1."""
    tensors = []
    for i in range(N):
        A = tc.zeros((1, 2, 1), dtype=dtype, device=device)
        if init == "neel":
            # site basis convention for spin operators: 0=up, 1=down.
            phys = 0 if (i % 2 == 0) else 1
        elif init == "all_up":
            phys = 0
        elif init == "all_down":
            phys = 1
        else:
            raise ValueError(f"unknown init {init!r}")
        A[0, phys, 0] = 1.0
        tensors.append(A)
    return MPS.from_tensors(tensors, dtype=dtype, device=device, requires_grad=False)


def make_random_mps(N: int, chi: int, dtype: tc.dtype, device: str, seed: int) -> MPS:
    """Random MPS fallback. Usually less stable than product init for large N."""
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=dtype, device=device)
    # Keep tensor scales bounded. Rayleigh quotient is scale invariant, but
    # long random MPS chains are easy to overflow/underflow without this.
    with tc.no_grad():
        for i in range(N):
            n = mps.tensors[i].norm()
            if n > 0:
                mps.tensors[i].data = (mps.tensors[i] / n).to(dtype).contiguous().data
    return mps


def bond_dims(mps: MPS) -> list[int]:
    return [int(mps.tensors[i].shape[2]) for i in range(mps.N - 1)]


def max_bond(mps: MPS) -> int:
    bd = bond_dims(mps)
    return max(bd) if bd else 1


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


def finite_or_raise(x: tc.Tensor, where: str) -> None:
    if not tc.isfinite(x.detach()).all().item():
        raise FloatingPointError(f"non-finite tensor encountered at {where}")


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
            finite_or_raise(loss, f"bond {bond} loss before backward")
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
        # For LBFGS, local_steps means number of optimizer.step(closure) calls;
        # each call internally evaluates the closure up to lbfgs_iters times.
        for _ in range(max(1, local_steps)):
            def closure():
                opt.zero_grad(set_to_none=True)
                loss = adtso.energy()
                finite_or_raise(loss, f"bond {bond} LBFGS closure loss")
                loss.backward()
                return loss

            loss_out = opt.step(closure)
            # Avoid an extra forward/backward diagnostic pass on large-N runs.
            # PyTorch leaves gradients from the last closure evaluation on theta;
            # use them only as a rough diagnostic. The authoritative energy is
            # the global Rayleigh quotient computed once per sweep.
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


def cuda_memory_summary(device: str) -> str:
    if not str(device).startswith("cuda") or not tc.cuda.is_available():
        return ""
    dev = tc.device(device)
    allocated = tc.cuda.memory_allocated(dev) / 1024**3
    reserved = tc.cuda.memory_reserved(dev) / 1024**3
    peak = tc.cuda.max_memory_allocated(dev) / 1024**3
    return f"allocated={allocated:.3f} GB reserved={reserved:.3f} GB peak={peak:.3f} GB"


def run_ad_two_site_large(
    N: int,
    chi: int,
    sweeps: int,
    device: str,
    dtype: tc.dtype,
    init: str,
    optimizer_name: str,
    local_steps: int,
    lr: float,
    lbfgs_iters: int,
    cutoff: float | None,
    stabilization: str,
    grad_clip: float | None,
    seed: int,
    print_bonds: bool,
    json_output: Path | None,
) -> dict:
    if device.startswith("cuda") and not tc.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")

    tc.manual_seed(seed)
    if device.startswith("cuda"):
        tc.cuda.reset_peak_memory_stats(tc.device(device))

    mpo = MPO.from_bonds(N, 2, dtype=dtype, device=device).generate_heisenberg(J=1.0)
    if init == "random":
        mps = make_random_mps(N, chi=chi, dtype=dtype, device=device, seed=seed)
    else:
        mps = make_product_mps(N, init=init, dtype=dtype, device=device)

    adtso = ADTwoSiteOptimizer(mps, mpo, bond=0)

    print("=" * 88)
    print("latticeTN N=80-style two-site AD local optimization, NO ED / NO DMRG / NO Lanczos")
    print("Model: spin-1/2 Heisenberg chain, open boundary, H = sum_i S_i . S_{i+1}, J=1")
    print(f"N={N} chi={chi} sweeps={sweeps} init={init}")
    print(f"optimizer={optimizer_name} local_steps={local_steps} lr={lr} lbfgs_iters={lbfgs_iters}")
    print(f"max_bond_dim={chi} cutoff={cutoff} stabilization={stabilization} grad_clip={grad_clip}")
    print(f"device={device} dtype={dtype} torch={tc.__version__}")
    if device.startswith("cuda"):
        print(f"GPU: {tc.cuda.get_device_name(tc.device(device))}")
    print(f"Bethe thermodynamic-limit e_inf = {BETHE_E_INF:.16f}")
    print("ED status: skipped by design; this script never builds a dense Hamiltonian.")
    print("AD path: loss.backward() on two-site Rayleigh quotient; SVD is post-step split only.")
    print("=" * 88)

    t_all = time.perf_counter()
    e0 = current_energy(adtso.mps, mpo)
    history = [{
        "sweep": -1,
        "direction": "init",
        "energy": e0,
        "energy_per_site": e0 / N,
        "delta_to_bethe_per_site": e0 / N - BETHE_E_INF,
        "max_bond": max_bond(adtso.mps),
        "max_trunc": 0.0,
        "runtime_s": 0.0,
    }]
    print(
        f"init      E={e0:.12f}  E/N={e0/N:.12f}  "
        f"dE/N_to_Bethe={e0/N - BETHE_E_INF:+.3e}  max_bond={max_bond(adtso.mps)}"
    )

    for s in range(sweeps):
        direction = "right" if s % 2 == 0 else "left"
        bonds = range(N - 1) if direction == "right" else range(N - 2, -1, -1)
        t0 = time.perf_counter()
        bond_reports = []
        for count, b in enumerate(bonds, start=1):
            br = local_optimize(
                adtso=adtso,
                bond=b,
                direction=direction,
                optimizer_name=optimizer_name,  # type: ignore[arg-type]
                local_steps=local_steps,
                lr=lr,
                lbfgs_iters=lbfgs_iters,
                max_bond_dim=chi,
                cutoff=cutoff,
                stabilization=stabilization,  # type: ignore[arg-type]
                grad_clip=grad_clip,
            )
            bond_reports.append(br)
            if print_bonds:
                print(
                    f"  bond={b:03d} {direction:5s} local_loss={br['local_loss']:.12f} "
                    f"grad={br['grad_norm']:.3e} kept={br['kept_bond']:3d} trunc={br['truncation']:.2e}"
                )
            elif count % 20 == 0 or count == N - 1:
                print(f"  sweep {s:02d} {direction:5s}: finished {count:3d}/{N-1} bonds", flush=True)

        e = current_energy(adtso.mps, mpo)
        dt = time.perf_counter() - t0
        max_trunc = max((x["truncation"] for x in bond_reports), default=0.0)
        max_grad = max((x["grad_norm"] for x in bond_reports), default=0.0)
        bd = bond_dims(adtso.mps)
        rec = {
            "sweep": s,
            "direction": direction,
            "energy": e,
            "energy_per_site": e / N,
            "delta_to_bethe_per_site": e / N - BETHE_E_INF,
            "delta_energy": e - history[-1]["energy"],
            "max_bond": max(bd) if bd else 1,
            "bond_dims": bd,
            "max_trunc": max_trunc,
            "max_grad_norm": max_grad,
            "runtime_s": dt,
            "bond_reports": bond_reports if print_bonds else None,
        }
        history.append(rec)
        print(
            f"sweep={s:02d} dir={direction:5s} E={e:.12f}  E/N={e/N:.12f}  "
            f"dE={rec['delta_energy']:+.3e}  dE/N_to_Bethe={e/N - BETHE_E_INF:+.3e}  "
            f"max_trunc={max_trunc:.2e}  max_grad={max_grad:.2e}  max_bond={rec['max_bond']}  "
            f"time={dt:.2f}s"
        )
        print(f"  bond_dims = {bd}")
        mem = cuda_memory_summary(device)
        if mem:
            print(f"  GPU memory: {mem}")

    total = time.perf_counter() - t_all
    final = history[-1]
    print("=" * 88)
    print(f"FINAL E                 = {final['energy']:.16f}")
    print(f"FINAL E / site          = {final['energy_per_site']:.16f}")
    print(f"FINAL E / bond          = {final['energy'] / (N - 1):.16f}")
    print(f"E/site - Bethe e_inf    = {final['delta_to_bethe_per_site']:.16f}")
    print(f"final max trunc         = {final['max_trunc']:.3e}")
    print(f"final max grad norm     = {final.get('max_grad_norm', 0.0):.3e}")
    print(f"final max bond          = {final['max_bond']}")
    print(f"total runtime           = {total:.3f} s")
    mem = cuda_memory_summary(device)
    if mem:
        print(f"final GPU memory: {mem}")
    print("ED status               = skipped by design")
    print("classical DMRG/Lanczos  = not used")
    print("=" * 88)

    result = {
        "N": N,
        "chi": chi,
        "sweeps": sweeps,
        "device": device,
        "dtype": str(dtype),
        "init": init,
        "optimizer": optimizer_name,
        "local_steps": local_steps,
        "lr": lr,
        "lbfgs_iters": lbfgs_iters,
        "cutoff": cutoff,
        "stabilization": stabilization,
        "grad_clip": grad_clip,
        "bethe_e_inf": BETHE_E_INF,
        "history": history,
        "final_energy": final["energy"],
        "final_energy_per_site": final["energy_per_site"],
        "final_delta_to_bethe_per_site": final["delta_to_bethe_per_site"],
        "final_bond_dims": bond_dims(adtso.mps),
        "total_runtime_s": total,
        "ed_skipped": True,
        "dmrg_lanczos_used": False,
    }
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        # Drop verbose bond reports if None values bother external tools.
        json_output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"JSON written to: {json_output}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="complex64",
                        choices=["complex64", "complex128", "float32", "float64"])
    parser.add_argument("--init", type=str, default="neel",
                        choices=["neel", "all_up", "all_down", "random"])
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["adam", "lbfgs"])
    parser.add_argument("--local-steps", type=int, default=1,
                        help="Adam: gradient steps per bond. LBFGS: optimizer.step calls per bond.")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-iters", type=int, default=10,
                        help="Internal LBFGS iterations per optimizer.step call.")
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--stabilization", type=str, default="tensor_norm",
                        choices=["none", "tensor_norm"])
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-bonds", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    run_ad_two_site_large(
        N=args.N,
        chi=args.chi,
        sweeps=args.sweeps,
        device=args.device,
        dtype=dtype,
        init=args.init,
        optimizer_name=args.optimizer,
        local_steps=args.local_steps,
        lr=args.lr,
        lbfgs_iters=args.lbfgs_iters,
        cutoff=args.cutoff,
        stabilization=args.stabilization,
        grad_clip=args.grad_clip,
        seed=args.seed,
        print_bonds=args.print_bonds,
        json_output=args.json_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
