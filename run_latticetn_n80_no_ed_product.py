#!/usr/bin/env python3
"""Large-N Heisenberg DMRG sanity run for latticeTN, with NO exact diagonalization.

Designed for Windows/PowerShell and GPU runs.  It avoids the package's run_dmrg()
post-processing path because that path tries to build a dense ED reference.

Important stability choices:
  1. Initialize from a normalized Neel/product MPS by default, not the package's
     random high-bond MPS.  Random unscaled complex64 MPS tensors can overflow
     during the first QR canonicalization for N~80.
  2. Patch the local Lanczos solver so the tiny tridiagonal Ritz problem is
     solved on CPU with scipy.linalg.eigh_tridiagonal when SciPy is available.
     The DMRG tensors and Heff applies remain on --device.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Literal

import torch as tc

from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.dmrg import two_site_sweep
from latticetn import contractions as K
import latticetn.lanczos as LZ


def parse_dtype(s: str) -> tc.dtype:
    s = str(s).lower().replace("torch.", "")
    table = {
        "complex64": tc.complex64,
        "cfloat": tc.complex64,
        "complex128": tc.complex128,
        "cdouble": tc.complex128,
    }
    if s not in table:
        raise argparse.ArgumentTypeError("Use complex64 or complex128")
    return table[s]


def resolve_device(s: str) -> str:
    s = str(s).lower()
    if s == "auto":
        return "cuda" if tc.cuda.is_available() else "cpu"
    if s.startswith("cuda") and not tc.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False")
    return s


def sync_if_needed(device: str) -> None:
    if str(device).startswith("cuda"):
        tc.cuda.synchronize(tc.device(device))


def gpu_mem(prefix: str, device: str) -> str:
    if not str(device).startswith("cuda"):
        return ""
    dev = tc.device(device)
    alloc = tc.cuda.memory_allocated(dev) / 1024**3
    reserved = tc.cuda.memory_reserved(dev) / 1024**3
    peak = tc.cuda.max_memory_allocated(dev) / 1024**3
    return f"{prefix} GPU memory: allocated={alloc:.3f} GB reserved={reserved:.3f} GB peak={peak:.3f} GB"


def bond_dims(tensors: list[tc.Tensor]) -> list[int]:
    return [int(tensors[i].shape[2]) for i in range(len(tensors) - 1)]


def make_product_mps_tensors(
    N: int,
    dtype: tc.dtype,
    device: str,
    pattern: Literal["neel", "up", "down", "x"] = "neel",
) -> list[tc.Tensor]:
    """Return normalized bond-dim-1 product-state tensors, shape (1,2,1)."""
    tensors: list[tc.Tensor] = []
    invsqrt2 = 1.0 / math.sqrt(2.0)
    for i in range(N):
        A = tc.zeros((1, 2, 1), dtype=dtype, device=device)
        if pattern == "neel":
            A[0, i % 2, 0] = 1.0
        elif pattern == "up":
            A[0, 0, 0] = 1.0
        elif pattern == "down":
            A[0, 1, 0] = 1.0
        elif pattern == "x":
            A[0, 0, 0] = invsqrt2
            A[0, 1, 0] = invsqrt2
        else:
            raise ValueError(f"unknown product-state pattern {pattern!r}")
        tensors.append(A)
    return tensors


def make_random_lowbond_mps_tensors(
    N: int,
    chi_init: int,
    dtype: tc.dtype,
    device: str,
    seed: int,
    scale: float = 1e-2,
) -> list[tc.Tensor]:
    """Small-amplitude random MPS, then caller can canonicalize via DMRG sweeps.

    This is intentionally low-bond and scaled; do not use the package's MPS(N,2,chi)
    random init for long complex64 chains because it can overflow in QR sweeps.
    """
    g = tc.Generator(device=device).manual_seed(seed)
    bonds = [1]
    for i in range(1, N):
        bonds.append(min(chi_init, 2 ** min(i, N - i)))
    bonds.append(1)
    tensors: list[tc.Tensor] = []
    for i in range(N):
        shape = (bonds[i], 2, bonds[i + 1])
        real = tc.randn(shape, dtype=tc.float32 if dtype == tc.complex64 else tc.float64, device=device, generator=g)
        imag = tc.randn(shape, dtype=tc.float32 if dtype == tc.complex64 else tc.float64, device=device, generator=g)
        A = (real + 1j * imag).to(dtype) * scale
        tensors.append(A)
    return tensors


def _lowest_tridiag_eigenpair(alphas: list[float], betas: list[float]) -> tuple[float, list[float]]:
    """Small symmetric tridiagonal lowest eigenpair on CPU.

    Returns lowest eigenvalue and eigenvector components in the Lanczos basis.
    """
    import numpy as np

    a = np.asarray(alphas, dtype=np.float64)
    b = np.asarray(betas[: max(0, len(alphas) - 1)], dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("Lanczos tridiagonal contains NaN/Inf before diagonalization")
    try:
        from scipy.linalg import eigh_tridiagonal

        vals, vecs = eigh_tridiagonal(a, b, select="i", select_range=(0, 0), check_finite=True)
        return float(vals[0]), vecs[:, 0].astype(np.float64).tolist()
    except Exception:
        # Fallback for environments without SciPy. Matrix dimension is tiny.
        T = np.diag(a)
        if len(a) > 1:
            T += np.diag(b, 1) + np.diag(b, -1)
        vals, vecs = np.linalg.eigh(T)
        return float(vals[0]), vecs[:, 0].astype(np.float64).tolist()


def install_stable_lanczos_patch() -> None:
    """Monkey-patch latticetn.lanczos.lanczos_lowest_eigenpair.

    The package's original solver builds a tiny tridiagonal T and uses
    torch.linalg.eigh(T) on the selected device.  On CUDA/complex64, and
    sometimes even CPU torch for near-degenerate T, this may fail.  This patch
    keeps the expensive apply() calls on device, but solves the T problem using
    SciPy/NumPy on CPU.
    """

    def stable_lanczos_lowest_eigenpair(
        apply,
        dim: int,
        dtype=tc.complex128,
        device="cpu",
        max_iter: int | None = None,
        tol: float = 1e-12,
        seed: int = 0,
        num_restarts: int = 2,
    ) -> tuple[tc.Tensor, tc.Tensor]:
        D = int(dim)
        if max_iter is None:
            max_iter = min(D, 40)
        # Oversized Krylov spaces can become numerically redundant.  Cap at D.
        max_iter_eff = max(1, min(int(max_iter), D))
        g = tc.Generator(device=device).manual_seed(int(seed))
        best_E: tc.Tensor | None = None
        best_V: tc.Tensor | None = None

        def _rand_vec() -> tc.Tensor:
            real_dtype = tc.float32 if dtype == tc.complex64 else tc.float64
            vr = tc.randn(D, dtype=real_dtype, device=device, generator=g)
            vi = tc.randn(D, dtype=real_dtype, device=device, generator=g)
            return (vr + 1j * vi).to(dtype)

        def _run_once(v0: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
            Q: list[tc.Tensor] = []
            alphas: list[float] = []
            betas: list[float] = []
            q = v0.reshape(-1)
            qnrm = tc.linalg.norm(q)
            if not bool(tc.isfinite(qnrm).item()) or float(qnrm) == 0.0:
                raise FloatingPointError("initial Lanczos vector has invalid norm")
            q = q / qnrm
            Q.append(q)
            last_ritz: float | None = None

            for j in range(max_iter_eff):
                w = apply(Q[j]).reshape(-1)
                if not bool(tc.isfinite(w).all().item()):
                    raise FloatingPointError(
                        "Heff apply produced NaN/Inf. This usually means the MPS/envs overflowed; "
                        "use --init neel/product, --dtype complex128, or lower chi/max-iter."
                    )
                a_t = tc.dot(Q[j].conj(), w).real
                if not bool(tc.isfinite(a_t).item()):
                    raise FloatingPointError("Lanczos alpha became NaN/Inf")
                a = float(a_t.detach().cpu())
                alphas.append(a)

                w = w - a_t.to(dtype=w.dtype, device=w.device) * Q[j]
                # Full reorthogonalization, two passes.
                for _ in range(2):
                    for qj in Q:
                        w = w - tc.dot(qj.conj(), w) * qj
                b_t = tc.linalg.norm(w)
                if not bool(tc.isfinite(b_t).item()):
                    raise FloatingPointError("Lanczos beta became NaN/Inf")
                b = float(b_t.detach().cpu())

                # Periodically check Ritz convergence without requiring a full dense torch eigh.
                ritz, _ = _lowest_tridiag_eigenpair(alphas, betas)
                if last_ritz is not None and abs(ritz - last_ritz) < tol:
                    break
                last_ritz = ritz

                if b < 1e-14 or j == max_iter_eff - 1:
                    break
                betas.append(b)
                Q.append(w / b_t)

            E0_float, coeffs = _lowest_tridiag_eigenpair(alphas, betas)
            vec = tc.zeros_like(Q[0])
            for c, qk in zip(coeffs, Q):
                vec = vec + float(c) * qk
            vn = tc.linalg.norm(vec)
            if not bool(tc.isfinite(vn).item()) or float(vn) == 0.0:
                raise FloatingPointError("Ritz vector has invalid norm")
            vec = vec / vn
            E0 = tc.tensor(E0_float, dtype=tc.float64, device=device)
            return E0, vec

        last_err: Exception | None = None
        for _ in range(max(1, int(num_restarts))):
            try:
                E, V = _run_once(_rand_vec())
                if best_E is None or float(E.detach().cpu()) < float(best_E.detach().cpu()):
                    best_E, best_V = E, V
            except Exception as e:  # try another random start
                last_err = e
                continue
        if best_E is None or best_V is None:
            raise last_err if last_err is not None else RuntimeError("Lanczos failed without an exception")
        return best_E, best_V

    LZ.lanczos_lowest_eigenpair = stable_lanczos_lowest_eigenpair


def run_large_dmrg_no_ed(
    N: int,
    chi: int,
    sweeps: int,
    device: str,
    dtype: tc.dtype,
    seed: int,
    max_iter: int,
    tol: float,
    init: str,
    chi_init: int,
    quiet: bool,
) -> dict:
    tc.manual_seed(seed)
    if device.startswith("cuda"):
        tc.cuda.manual_seed_all(seed)
        tc.cuda.reset_peak_memory_stats(tc.device(device))

    mpo = MPO.from_bonds(N, 2, dtype=dtype, device=device).generate_heisenberg(J=1.0)
    if init in {"neel", "up", "down", "x"}:
        tensors = make_product_mps_tensors(N, dtype=dtype, device=device, pattern=init)  # type: ignore[arg-type]
    elif init == "random_lowbond":
        tensors = make_random_lowbond_mps_tensors(N, chi_init=chi_init, dtype=dtype, device=device, seed=seed)
    elif init == "package_random":
        # This is kept only for debugging.  It is NOT recommended for N~80 complex64.
        tensors = [x.detach().clone() for x in MPS(N, 2, chi, dtype=dtype, device=device).tensors]
    else:
        raise ValueError(f"unknown --init {init!r}")

    history: list[dict] = []
    e_inf = 0.25 - math.log(2.0)

    print("=" * 96)
    print("latticeTN large-N DMRG, NO exact diagonalization")
    print("Model: spin-1/2 Heisenberg chain, open boundary, H = sum_i S_i . S_{i+1}, J=1")
    print(f"N={N}  chi={chi}  sweeps={sweeps}  solver=lanczos  max_iter={max_iter}  tol={tol:g}")
    print(f"device={device}  dtype={dtype}  torch={tc.__version__}  init={init}")
    if device.startswith("cuda"):
        idx = tc.device(device).index
        if idx is None:
            idx = tc.cuda.current_device()
        print(f"GPU: {tc.cuda.get_device_name(idx)}")
    print(f"Bethe thermodynamic-limit reference e_inf = 1/4 - ln(2) = {e_inf:.16f}")
    print("Note: for finite open chains, E/N is not expected to equal e_inf exactly.")
    print("ED status: skipped by design; this script never builds a dense Hamiltonian.")
    print("=" * 96)
    sys.stdout.flush()

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
            lanczos_kwargs={"max_iter": max_iter, "tol": tol, "seed": seed + 1000 * s},
        )

        mps_cur = MPS.from_tensors(tensors, dtype=dtype, device=device, requires_grad=False)
        E_t = K.rayleigh_energy_native(mps_cur, mpo)
        if not bool(tc.isfinite(E_t).item()):
            raise FloatingPointError("global Rayleigh energy is NaN/Inf after sweep")
        E = float(E_t.detach().cpu())

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
    p.add_argument("--max-iter", type=int, default=40, help="Lanczos iterations per two-site local solve")
    p.add_argument("--tol", type=float, default=1e-7, help="Lanczos Ritz-value tolerance")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    p.add_argument("--quiet", action="store_true", help="do not print bond dimensions/memory each sweep")
    p.add_argument("--init", choices=["neel", "up", "down", "x", "random_lowbond", "package_random"], default="neel")
    p.add_argument("--chi-init", type=int, default=4, help="initial chi only for --init random_lowbond")
    p.add_argument("--no-stable-lanczos", action="store_true", help="do not patch Lanczos; use package original")
    args = p.parse_args()

    device = resolve_device(args.device)
    tc.set_num_threads(args.threads)
    if not args.no_stable_lanczos:
        install_stable_lanczos_patch()
        print("Stable Lanczos patch: ON; Krylov tridiagonal is solved by SciPy/NumPy on CPU, while DMRG tensors stay on the selected device.")
    else:
        print("Stable Lanczos patch: OFF; using package original Lanczos.")

    run_large_dmrg_no_ed(
        N=args.N,
        chi=args.chi,
        sweeps=args.sweeps,
        device=device,
        dtype=args.dtype,
        seed=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
        init=args.init,
        chi_init=args.chi_init,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
