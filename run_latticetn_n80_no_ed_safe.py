#!/usr/bin/env python3
"""Run latticeTN finite-OBC DMRG for a large Heisenberg chain with NO ED.

Usage from the latticeTN repository root:

  # Fast GPU smoke test
  PYTHONPATH=. python run_latticetn_n80_no_ed.py \
      --N 80 --chi 32 --sweeps 3 --device cuda --dtype complex64

  # More serious GPU run
  PYTHONPATH=. python run_latticetn_n80_no_ed.py \
      --N 80 --chi 64 --sweeps 6 --device cuda --dtype complex64 \
      --max-iter 80 --tol 1e-8

This script deliberately avoids latticetn.dmrg.run_dmrg(), because that driver
tries to build a dense Heisenberg Hamiltonian at the end for exact diagonalization
context. For N=80 that is impossible. Instead, this script calls two_site_sweep()
directly and computes only the native MPO Rayleigh energy.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

# Keep CPU BLAS from oversubscribing when using CPU. These must be set before
# importing torch to be maximally effective.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch as tc

from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.dmrg import two_site_sweep
from latticetn import contractions as K



def install_safe_lanczos_cpu_eigh() -> None:
    """Patch latticeTN Lanczos so the tiny Krylov T matrix is diagonalized on CPU.

    The DMRG tensors and Heff apply still run on the requested device, e.g. CUDA.
    Only the m x m tridiagonal Lanczos matrix, with m <= --max-iter, is copied
    to CPU float64 before torch.linalg.eigh. This avoids occasional cuSolver
    failures on ill-conditioned/repeated Ritz values.
    """
    import torch as tc
    import latticetn.lanczos as LZ

    def safe_lanczos_lowest_eigenpair(
        apply,
        dim: int,
        dtype=tc.complex128,
        device="cpu",
        max_iter: int | None = None,
        tol: float = 1e-12,
        seed: int = 0,
        num_restarts: int = 3,
    ):
        D = int(dim)
        if max_iter is None:
            max_iter = min(D, 40)
        max_iter_eff = int(min(D, max_iter))
        g = tc.Generator(device=device).manual_seed(int(seed))
        best_E = None
        best_V = None

        def _run_once(v0: tc.Tensor):
            Q = []
            alphas = []
            betas = []
            q = v0 / (tc.linalg.norm(v0) + 1e-30)
            Q.append(q)

            prev_ritz = None
            for j in range(max_iter_eff):
                w = apply(Q[j]).reshape(-1)
                a = tc.dot(Q[j].conj(), w).real
                alphas.append(a)

                w = w - a * Q[j]
                for qj in Q:
                    w = w - tc.dot(qj.conj(), w) * qj
                for qj in Q:
                    w = w - tc.dot(qj.conj(), w) * qj

                b = tc.linalg.norm(w)
                betas.append(float(b.detach().cpu()))

                # Check Ritz convergence occasionally using CPU eigh.
                m = len(alphas)
                if m >= 2:
                    T_cpu = tc.zeros(m, m, dtype=tc.float64, device="cpu")
                    for ii in range(m):
                        T_cpu[ii, ii] = float(alphas[ii].detach().cpu())
                    for ii in range(m - 1):
                        T_cpu[ii, ii + 1] = betas[ii]
                        T_cpu[ii + 1, ii] = betas[ii]
                    try:
                        E_tmp, _ = tc.linalg.eigh(T_cpu)
                    except RuntimeError:
                        eps = 1e-12 * max(1.0, float(T_cpu.abs().max()))
                        T_cpu = T_cpu + eps * tc.eye(m, dtype=tc.float64)
                        E_tmp, _ = tc.linalg.eigh(T_cpu)
                    ritz = float(E_tmp[0])
                    if prev_ritz is not None and abs(ritz - prev_ritz) < tol:
                        break
                    prev_ritz = ritz

                if b < 1e-14:
                    break
                if len(Q) >= max_iter_eff:
                    break
                qn = w / b
                Q.append(qn)

            m = len(alphas)
            T_cpu = tc.zeros(m, m, dtype=tc.float64, device="cpu")
            for ii in range(m):
                T_cpu[ii, ii] = float(alphas[ii].detach().cpu())
            for ii in range(m - 1):
                T_cpu[ii, ii + 1] = betas[ii]
                T_cpu[ii + 1, ii] = betas[ii]
            try:
                E_cpu, V_cpu = tc.linalg.eigh(T_cpu)
            except RuntimeError:
                eps = 1e-12 * max(1.0, float(T_cpu.abs().max()))
                T_cpu = T_cpu + eps * tc.eye(m, dtype=tc.float64)
                E_cpu, V_cpu = tc.linalg.eigh(T_cpu)

            E0 = E_cpu[0].to(device=device)
            coeffs = V_cpu[:, 0].to(device=device, dtype=tc.float64)
            vec = sum(float(coeffs[k].detach().cpu()) * Q[k] for k in range(m))
            vec = vec / (tc.linalg.norm(vec) + 1e-30)
            return E0, vec

        for r in range(int(num_restarts)):
            v0 = tc.randn(D, dtype=dtype, device=device, generator=g) + 1j * tc.randn(
                D, dtype=dtype, device=device, generator=g
            )
            E, V = _run_once(v0)
            if best_E is None or float(E.detach().cpu()) < float(best_E.detach().cpu()):
                best_E = E
                best_V = V

        return best_E, best_V

    LZ.lanczos_lowest_eigenpair = safe_lanczos_lowest_eigenpair


def parse_dtype(name: str) -> tc.dtype:
    table = {
        "complex64": tc.complex64,
        "c64": tc.complex64,
        "complex128": tc.complex128,
        "c128": tc.complex128,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "dtype must be one of: complex64, c64, complex128, c128"
        ) from exc


def resolve_device(name: str) -> str:
    name = name.lower()
    if name == "auto":
        return "cuda" if tc.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not tc.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return name


def sync_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        tc.cuda.synchronize()


def gpu_mem(prefix: str, device: str) -> str:
    if not device.startswith("cuda"):
        return ""
    dev = tc.device(device)
    alloc = tc.cuda.memory_allocated(dev) / 1024**3
    reserved = tc.cuda.memory_reserved(dev) / 1024**3
    max_alloc = tc.cuda.max_memory_allocated(dev) / 1024**3
    return f"{prefix} cuda_mem allocated={alloc:.3f}GB reserved={reserved:.3f}GB max_alloc={max_alloc:.3f}GB"


def bond_dims(tensors: list[tc.Tensor]) -> list[int]:
    return [int(tensors[i].shape[2]) for i in range(len(tensors) - 1)]


def run_large_dmrg_no_ed(
    *,
    N: int,
    chi: int,
    sweeps: int,
    device: str,
    dtype: tc.dtype,
    seed: int,
    max_iter: int,
    tol: float,
    quiet: bool,
) -> dict:
    tc.manual_seed(seed)
    if device.startswith("cuda"):
        tc.cuda.manual_seed_all(seed)
        tc.cuda.reset_peak_memory_stats(tc.device(device))

    mps = MPS(N, 2, chi, dtype=dtype, device=device)
    mpo = MPO.from_bonds(N, 2, dtype=dtype, device=device).generate_heisenberg(J=1.0)

    tensors = [x.detach().clone() for x in mps.tensors]
    history: list[dict] = []
    e_inf = 0.25 - math.log(2.0)

    print("=" * 96)
    print("latticeTN large-N DMRG, NO exact diagonalization")
    print("Model: spin-1/2 Heisenberg chain, open boundary, H = sum_i S_i . S_{i+1}, J=1")
    print(f"N={N}  chi={chi}  sweeps={sweeps}  solver=lanczos  max_iter={max_iter}  tol={tol:g}")
    print(f"device={device}  dtype={dtype}  torch={tc.__version__}")
    if device.startswith("cuda"):
        idx = tc.device(device).index
        if idx is None:
            idx = tc.cuda.current_device()
        print(f"GPU: {tc.cuda.get_device_name(idx)}")
    print(f"Bethe thermodynamic-limit reference e_inf = 1/4 - ln(2) = {e_inf:.16f}")
    print("Note: for finite open chains, E/N is not expected to equal e_inf exactly.")
    print("=" * 96)

    sync_if_needed(device)
    total_t0 = time.perf_counter()

    for s in range(sweeps):
        direction = "right" if (s % 2 == 0) else "left"
        sync_if_needed(device)
        t0 = time.perf_counter()

        tensors, last_local_E, trunc_errs = two_site_sweep(
            tensors,
            mpo,
            chi=chi,
            direction=direction,
            solver="lanczos",
            lanczos_kwargs={"max_iter": max_iter, "tol": tol},
        )

        mps_cur = MPS.from_tensors(
            tensors,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        E = float(K.rayleigh_energy_native(mps_cur, mpo))

        sync_if_needed(device)
        dt = time.perf_counter() - t0
        dims = bond_dims(tensors)
        max_trunc = max(trunc_errs) if trunc_errs else 0.0
        rec = {
            "sweep": s,
            "direction": direction,
            "energy": E,
            "energy_per_site": E / N,
            "energy_per_bond": E / max(1, N - 1),
            "delta_to_bethe_per_site": E / N - e_inf,
            "local_last_energy": float(last_local_E),
            "max_trunc": float(max_trunc),
            "max_bond": max(dims) if dims else 1,
            "bond_dims": dims,
            "seconds": dt,
        }
        history.append(rec)

        print(
            f"sweep={s:02d} dir={direction:5s} "
            f"E={E:.12f}  E/N={E / N:.12f}  "
            f"dE/N_to_Bethe={E / N - e_inf:+.3e}  "
            f"max_trunc={max_trunc:.2e}  max_bond={rec['max_bond']}  "
            f"time={dt:.2f}s"
        )
        if not quiet:
            print("  bond_dims =", dims)
            mem_line = gpu_mem(" ", device)
            if mem_line:
                print(mem_line)
        sys.stdout.flush()

    sync_if_needed(device)
    total_dt = time.perf_counter() - total_t0
    final = history[-1]

    print("=" * 96)
    print(f"FINAL E                 = {final['energy']:.16f}")
    print(f"FINAL E / site          = {final['energy_per_site']:.16f}")
    print(f"FINAL E / bond          = {final['energy_per_bond']:.16f}")
    print(f"E/site - Bethe e_inf    = {final['delta_to_bethe_per_site']:.16f}")
    print(f"final max trunc         = {final['max_trunc']:.3e}")
    print(f"final max bond          = {final['max_bond']}")
    print(f"total runtime           = {total_dt:.3f} s")
    mem_line = gpu_mem("final", device)
    if mem_line:
        print(mem_line)
    print("ED status               = skipped by design")
    print("=" * 96)

    return {"history": history, "final": final, "runtime": total_dt}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--chi", type=int, default=64)
    p.add_argument("--sweeps", type=int, default=6)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--dtype", type=parse_dtype, default=tc.complex64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=80, help="Lanczos iterations per two-site local solve")
    p.add_argument("--tol", type=float, default=1e-8, help="Lanczos convergence tolerance")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    p.add_argument("--quiet", action="store_true", help="do not print bond dimensions/memory each sweep")
    p.add_argument("--safe-lanczos", action="store_true", default=True, help="diagonalize the tiny Lanczos T matrix on CPU float64 to avoid CUDA eigh failures")
    p.add_argument("--no-safe-lanczos", dest="safe_lanczos", action="store_false", help="use the package's original CUDA eigh path for Lanczos T")
    args = p.parse_args()

    device = resolve_device(args.device)
    tc.set_num_threads(args.threads)
    if args.safe_lanczos:
        install_safe_lanczos_cpu_eigh()
        print("Safe Lanczos patch: ON, Krylov T matrix uses CPU float64 eigh; DMRG tensors stay on the selected device.")

    run_large_dmrg_no_ed(
        N=args.N,
        chi=args.chi,
        sweeps=args.sweeps,
        device=device,
        dtype=args.dtype,
        seed=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
