"""Unified CPU/GPU benchmark registry for 1D models (Stage 7B).

A small, dependency-light registry that runs a solver on a model spec on CPU
and (opt-in) a V100/TITAN V GPU, and records the Stage-7A+ timing contract:

    model, N, chi, solver, device, device_name, dtype, runtime, speedup,
    final_energy, exact_error, below_ground, gpu_skip_reason.

It uses the **unified GPU selector** (``scripts/gpu_selector.py``) which
selects a GPU whose name contains ``V100`` or ``TITAN V``/``Titan V`` and
clean-skips (no fallback) otherwise. The GPU is NOT required to be faster;
speedup is recorded only to observe AD-TN GPU acceleration trends.

This is a **benchmark/recording layer, NOT a solver**. The AD mainline
(differentiable Rayleigh quotient + autograd + torch optimizer) is unchanged;
SVD/QR/canonicalization remain auxiliary stabilization; exact/DMRG/Lanczos
remain reference baselines.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
# scripts/ is a sibling of latticetn/; import the unified selector from there.
sys.path.insert(0, str(ROOT / "scripts"))

from gpu_selector import select_gpu, selection_report_dict  # noqa: E402

from .model_builder import ModelSpec, build_mpo
from .mps import MPS
from .ad_variational import ADVariationalMPS, train_ad_mps
from .operators import exact_ground_energy
from . import model_builder as MB

DTYPE = tc.complex128
BELOW_GROUND_TOL = 1e-6
ENERGY_AGREE_TOL = {4: 1e-6, 6: 1e-5}


@dataclass
class RunRecord:
    """One solver run on one (model, N, chi, device) cell."""
    model: str
    N: int
    chi: int
    solver: str
    device: str
    device_name: str
    dtype: str
    runtime: float
    speedup: float | None
    final_energy: float
    exact_error: float
    below_ground: bool
    gpu_skip_reason: str | None = None
    extra: dict = field(default_factory=dict)


def _exact_e0(spec: ModelSpec) -> float:
    H = MB.build_dense(spec)
    return float(exact_ground_energy(H)[0])


def _build_mps(N, chi, seed, device):
    tc.manual_seed(seed)
    return MPS(N, 2, chi, dtype=DTYPE, device=device)


def _copy_mps_into(dst, src):
    for a, b in zip(dst.tensors, src.tensors):
        a.data = b.data.to(device=dst.device).to(dtype=DTYPE)


def run_global_ad(spec: ModelSpec, chi: int, seed: int, device,
                  steps: int = 120, lr: float = 1e-2) -> tuple[dict, float]:
    """Run global AD-MPS on `spec`/`device`; return (train_result, runtime)."""
    tc.manual_seed(seed)
    mps_cpu = MPS(spec.N, 2, chi, dtype=DTYPE, device="cpu")
    mps = MPS(spec.N, 2, chi, dtype=DTYPE, device=device)
    _copy_mps_into(mps, mps_cpu)
    mpo = build_mpo(spec)
    # MPO is built on CPU by build_mpo; move to device if needed.
    if device != "cpu":
        mpo.tensors = [tc.nn.Parameter(t.to(device=device), requires_grad=False)
                       for t in mpo.tensors]
        mpo.device = device
    ad = ADVariationalMPS(mps, mpo)
    t0 = time.perf_counter()
    r = train_ad_mps(ad, num_steps=steps, lr=lr, optimizer="adam",
                     projection="tensor_norm")
    return r, time.perf_counter() - t0


def benchmark_model(spec: ModelSpec, chi: int, seed: int = 0,
                    steps: int = 120) -> dict:
    """Benchmark one model on CPU and (opt-in) V100/TITAN V GPU.

    Returns a dict with: model, N, chi, seed, dtype, exact_energy, cpu record,
    gpu record (or None with skip reason), speedup, device_info, gpu_skip_reason.
    """
    sel = select_gpu()
    device_info = selection_report_dict(sel)
    if not sel.skip_reason:
        gpu_device = sel.device
        gpu_skip_reason = None
    else:
        gpu_device = None
        gpu_skip_reason = sel.skip_reason
    # CPU-only default env when not opted in.
    if os.environ.get("LATTICETN_RUN_GPU", "") != "1":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    exact_e0 = _exact_e0(spec)

    # --- CPU ---
    r_cpu, t_cpu = run_global_ad(spec, chi, seed, "cpu", steps=steps)
    fe_cpu = float(r_cpu["final_energy"])
    cpu_rec = RunRecord(
        model=spec.name, N=spec.N, chi=chi, solver="global AD-MPS",
        device="cpu", device_name="cpu", dtype=str(DTYPE),
        runtime=t_cpu, speedup=None,
        final_energy=fe_cpu,
        exact_error=abs(fe_cpu - exact_e0),
        below_ground=bool(fe_cpu < exact_e0 - BELOW_GROUND_TOL),
        gpu_skip_reason=None,
    )

    gpu_rec = None
    speedup = None
    if gpu_skip_reason is None:
        r_gpu, t_gpu = run_global_ad(spec, chi, seed, gpu_device, steps=steps)
        fe_gpu = float(r_gpu["final_energy"])
        gpu_rec = RunRecord(
            model=spec.name, N=spec.N, chi=chi, solver="global AD-MPS",
            device=str(gpu_device), device_name=sel.gpu_name or str(gpu_device),
            dtype=str(DTYPE), runtime=t_gpu, speedup=None,
            final_energy=fe_gpu,
            exact_error=abs(fe_gpu - exact_e0),
            below_ground=bool(fe_gpu < exact_e0 - BELOW_GROUND_TOL),
        )
        speedup = float(t_cpu / t_gpu) if t_gpu > 0 else float("inf")
        cpu_rec.speedup = None
        gpu_rec.speedup = speedup

    return {
        "model": spec.name,
        "N": spec.N,
        "chi": chi,
        "seed": seed,
        "dtype": str(DTYPE),
        "exact_energy": exact_e0,
        "cpu": asdict(cpu_rec),
        "gpu": asdict(gpu_rec) if gpu_rec is not None else None,
        "gpu_skip_reason": gpu_skip_reason,
        "gpu_ran": gpu_skip_reason is None,
        "speedup": speedup,
        "cpu_gpu_energy_diff": (abs(fe_cpu - gpu_rec.final_energy)
                                if gpu_rec is not None else None),
        "device_info": device_info,
    }


def registry_report_dict(report: dict) -> dict:
    """Flatten a benchmark_model result for JSON/markdown."""
    return report


__all__ = [
    "RunRecord", "run_global_ad", "benchmark_model", "registry_report_dict",
]
