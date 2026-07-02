#!/usr/bin/env python3
"""Stage 6A CPU/GPU AD solver benchmark runner (the AD mainline benchmark).

Benchmarks the three AD mainline solvers of latticeTN on the open-boundary 1D
spin-1/2 Heisenberg chain on CPU and (opt-in) the machine's single GPU:

1. global AD-MPS            (latticetn.ad_variational.train_ad_mps, Adam)
2. one-site AD local opt    (latticetn.ad_local.train_ad_local, LBFGS)
3. two-site AD local opt    (latticetn.ad_two_site.train_ad_two_site, LBFGS)

plus reference baselines (NEVER the AD mainline):
4. classical DMRG           (latticetn.dmrg.run_dmrg, dense, CPU-only)
5. exact diagonalization    (latticetn.operators.exact_ground_energy, CPU-only)

GPU rules (this machine has exactly one GPU; see docs/AD_GPU_BENCHMARK_SPEC.md):
- GPU is OPT-IN: runs only when LATTICETN_RUN_GPU=1.
- When opted-in and torch.cuda.is_available() is True, use cuda:0.
- When opted-in but CUDA is unavailable, the GPU portion clean-skips (report
  records the skip; CPU portion still runs; exit 0).
- Default runs are CPU-only and never require a GPU.

CPU and GPU use the SAME seed, dtype, and solver config; the GPU MPS/MPO start
from tensors copied by value from the CPU build so the comparison is
apples-to-apples. No AD loss path is modified; this runner only CALLS the
existing train_ad_* functions with device-placed MPS/MPO.

Conventions: H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open boundary,
complex128. ED/DMRG are CPU-only reference baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn.ad_local import train_ad_local  # noqa: E402
from latticetn.ad_two_site import train_ad_two_site  # noqa: E402
from latticetn import dmrg as D  # noqa: E402  (reference baseline only)
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128

# Solver labels (AD mainline first; reference baselines flagged separately).
SOLVER_GLOBAL_AD = "global AD-MPS"
SOLVER_ONE_SITE_AD = "one-site AD local"
SOLVER_TWO_SITE_AD = "two-site AD local"
SOLVER_DMRG = "dense DMRG (reference)"
SOLVER_EXACT = "exact diagonalization (reference)"

# Tolerances for the CPU/GPU energy agreement and below-ground guard. These
# mirror the existing AD tolerances and are NOT widened here.
ENERGY_AGREE_TOL = {4: 1e-6, 6: 1e-5}
BELOW_GROUND_TOL = 1e-6


# ---------------------------------------------------------------------------
# Config + device selection
# ---------------------------------------------------------------------------

def env_run_gpu() -> bool:
    """GPU benchmark is opt-in via LATTICETN_RUN_GPU=1."""
    return os.environ.get("LATTICETN_RUN_GPU", "") == "1"


@dataclass
class CaseSpec:
    """One (N, chi) case with per-solver solver config."""
    N: int
    chi: int
    # global AD-MPS config
    global_steps: int
    global_lr: float
    # one-site AD config
    one_site_sweeps: int
    one_site_local_steps: int
    one_site_lr: float
    one_site_stabilization: str
    # two-site AD config
    two_site_sweeps: int
    two_site_local_steps: int
    two_site_lr: float
    two_site_max_bond: int


@dataclass
class BenchmarkConfig:
    """Top-level benchmark config (the --fast preset)."""
    cases: list = field(default_factory=list)
    seed: int = 0
    dtype: str = field(default_factory=lambda: str(DTYPE))
    # solver defaults (also stored per-case for the report)
    global_optimizer: str = "adam"
    one_site_optimizer: str = "lbfgs"
    two_site_optimizer: str = "lbfgs"


def fast_config() -> BenchmarkConfig:
    """The --fast preset: small N, short steps/sweeps, finishes quickly on CPU.

    Note: global AD-MPS is first-order Adam and needs more steps than the
    LBFGS local solvers to converge; the step counts below give a sensible
    energy within the --fast time budget. The benchmark contract is numerical
    parity + below-ground guard, not "every solver hits machine precision" —
    the energy error is reported honestly per row.
    """
    return BenchmarkConfig(
        cases=[
            CaseSpec(
                N=4, chi=4,
                global_steps=150, global_lr=1e-2,
                one_site_sweeps=2, one_site_local_steps=10,
                one_site_lr=1.0, one_site_stabilization="qr",
                two_site_sweeps=2, two_site_local_steps=10,
                two_site_lr=1.0, two_site_max_bond=4,
            ),
            CaseSpec(
                N=6, chi=8,
                global_steps=250, global_lr=1e-2,
                one_site_sweeps=3, one_site_local_steps=12,
                one_site_lr=1.0, one_site_stabilization="qr",
                two_site_sweeps=3, two_site_local_steps=12,
                two_site_lr=1.0, two_site_max_bond=8,
            ),
        ],
    )


def select_device() -> tuple[str, str | None]:
    """Resolve the GPU device string per the opt-in / clean-skip rules.

    Returns (device, skip_reason). When skip_reason is not None the GPU
    portion must clean-skip (device is "cpu" and unused for GPU runs).

    Note: ``torch.cuda.is_available()`` reports CUDA *build* support, which
    stays True even when ``CUDA_VISIBLE_DEVICES=""`` hides every device. We
    therefore also require ``device_count() > 0`` before using ``cuda:0``.
    """
    if not env_run_gpu():
        return "cpu", "LATTICETN_RUN_GPU != 1; GPU benchmark is opt-in and was not requested."
    if not tc.cuda.is_available() or tc.cuda.device_count() == 0:
        return "cpu", ("torch.cuda.is_available() is False or no visible CUDA "
                       "device (CUDA_VISIBLE_DEVICES may hide it); clean-skip.")
    # This machine has a single GPU; use cuda:0 (no name filtering).
    return "cuda:0", None


def gpu_device_info() -> dict:
    """Record GPU name / CUDA version / PyTorch version / device / dtype.

    Note: ``torch.cuda.is_available()`` reports CUDA *build* support, which
    stays True even when ``CUDA_VISIBLE_DEVICES=""`` hides every device. The
    number of *visible* devices is ``torch.cuda.device_count()``; we only
    report a gpu name / device when at least one device is actually visible.
    """
    cuda_built = bool(tc.cuda.is_available())
    n_vis = int(tc.cuda.device_count()) if cuda_built else 0
    info = {
        "torch_version": tc.__version__,
        "cuda_version": str(tc.version.cuda),
        "cuda_available": cuda_built,
        "device_count": n_vis,
        "gpu_name": None,
        "device": None,
        "dtype": str(DTYPE),
        "env_run_gpu": env_run_gpu(),
    }
    if n_vis > 0:
        info["gpu_name"] = tc.cuda.get_device_name(0)
        info["device"] = "cuda:0"
    return info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exact(N: int) -> float:
    """Exact Heisenberg ground energy for small N (CPU, numpy.linalg.eigh)."""
    H = heisenberg_dense(N, dtype=DTYPE, device="cpu")
    return float(exact_ground_energy(H)[0])


def _build_mps(N: int, chi: int, seed: int, device) -> MPS:
    """Build a fresh MPS on `device` with a pinned seed."""
    tc.manual_seed(seed)
    return MPS(N, 2, chi, dtype=DTYPE, device=device)


def _build_mpo(N: int, device) -> MPO:
    """Build the Heisenberg MPO on `device`."""
    return MPO.from_bonds(N, 2, dtype=DTYPE, device=device).generate_heisenberg(J=1.0)


def _copy_mps_into(dst: MPS, src: MPS) -> None:
    """Copy src tensors into dst by value (so CPU and GPU start identically)."""
    for a, b in zip(dst.tensors, src.tensors):
        a.data = b.data.to(device=dst.device).to(dtype=DTYPE)
    # MPO is rebuilt identically on each device (generate_heisenberg is
    # deterministic), so no copy is needed for the MPO.


def _energy_per_bond(final_energy: float, N: int) -> float:
    """Energy per bond = E / (N-1) for an open chain of N sites."""
    return float(final_energy) / (N - 1) if N > 1 else float(final_energy)


# ---------------------------------------------------------------------------
# Per-solver runners (AD mainline). Each takes a device and returns a result
# dict with final_energy, runtime_s, and solver/optimizer metadata.
# ---------------------------------------------------------------------------

def run_global_ad(spec: CaseSpec, device, seed: int) -> dict:
    mps = _build_mps(spec.N, spec.chi, seed, device)
    mpo = _build_mpo(spec.N, device)
    ad = ADVariationalMPS(mps, mpo)
    t0 = time.perf_counter()
    r = train_ad_mps(ad, num_steps=spec.global_steps, lr=spec.global_lr,
                     optimizer="adam")
    runtime = time.perf_counter() - t0
    return {
        "solver": SOLVER_GLOBAL_AD,
        "optimizer": "adam",
        "N": spec.N, "chi": spec.chi, "seed": seed,
        "num_steps": spec.global_steps, "lr": spec.global_lr,
        "final_energy": float(r["final_energy"]),
        "initial_energy": float(r["initial_energy"]),
        "runtime_s": runtime,
        "device": str(device),
    }


def run_one_site_ad(spec: CaseSpec, device, seed: int) -> dict:
    mps = _build_mps(spec.N, spec.chi, seed, device)
    mpo = _build_mpo(spec.N, device)
    t0 = time.perf_counter()
    r = train_ad_local(mps, mpo, num_sweeps=spec.one_site_sweeps,
                       local_steps=spec.one_site_local_steps,
                       lr=spec.one_site_lr, optimizer="lbfgs",
                       stabilization=spec.one_site_stabilization)
    runtime = time.perf_counter() - t0
    return {
        "solver": SOLVER_ONE_SITE_AD,
        "optimizer": "lbfgs",
        "N": spec.N, "chi": spec.chi, "seed": seed,
        "num_sweeps": spec.one_site_sweeps,
        "local_steps": spec.one_site_local_steps, "lr": spec.one_site_lr,
        "stabilization": spec.one_site_stabilization,
        "final_energy": float(r["final_energy"]),
        "initial_energy": float(r["initial_energy"]),
        "runtime_s": runtime,
        "device": str(device),
    }


def run_two_site_ad(spec: CaseSpec, device, seed: int) -> dict:
    mps = _build_mps(spec.N, spec.chi, seed, device)
    mpo = _build_mpo(spec.N, device)
    t0 = time.perf_counter()
    r = train_ad_two_site(mps, mpo, num_sweeps=spec.two_site_sweeps,
                          local_steps=spec.two_site_local_steps,
                          lr=spec.two_site_lr, optimizer="lbfgs",
                          max_bond_dim=spec.two_site_max_bond, cutoff=None)
    runtime = time.perf_counter() - t0
    return {
        "solver": SOLVER_TWO_SITE_AD,
        "optimizer": "lbfgs",
        "N": spec.N, "chi": spec.chi, "seed": seed,
        "num_sweeps": spec.two_site_sweeps,
        "local_steps": spec.two_site_local_steps, "lr": spec.two_site_lr,
        "max_bond_dim": spec.two_site_max_bond,
        "final_energy": float(r["final_energy"]),
        "initial_energy": float(r["initial_energy"]),
        "runtime_s": runtime,
        "device": str(device),
    }


# Reference baselines (CPU-only; never the AD mainline).

def run_dmrg_reference(spec: CaseSpec, seed: int = 0) -> dict:
    tc.manual_seed(seed)
    mps = MPS(spec.N, 2, spec.chi, dtype=DTYPE, device="cpu")
    mpo = _build_mpo(spec.N, "cpu")
    t0 = time.perf_counter()
    r = D.run_dmrg(mps, mpo, chi=spec.chi, num_sweeps=4, solver="dense")
    runtime = time.perf_counter() - t0
    return {
        "solver": SOLVER_DMRG,
        "optimizer": "none (classical DMRG)",
        "N": spec.N, "chi": spec.chi, "seed": seed,
        "final_energy": float(r["final_energy"]),
        "runtime_s": runtime,
        "device": "cpu",
        "is_reference": True,
    }


def run_exact_reference(N: int) -> dict:
    t0 = time.perf_counter()
    e0 = _exact(N)
    runtime = time.perf_counter() - t0
    return {
        "solver": SOLVER_EXACT,
        "optimizer": "none (numpy.linalg.eigh)",
        "N": N, "final_energy": float(e0),
        "runtime_s": runtime,
        "device": "cpu",
        "is_reference": True,
    }


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def _annotate(row: dict, exact_e0: float, dmrg_e: float | None) -> dict:
    """Add energy_error / energy_per_bond / below_ground to a solver row."""
    fe = row["final_energy"]
    row["exact_energy"] = exact_e0
    row["energy_error"] = float(abs(fe - exact_e0))
    row["energy_per_bond"] = _energy_per_bond(fe, row["N"])
    row["below_ground"] = bool(fe < exact_e0 - BELOW_GROUND_TOL)
    if dmrg_e is not None:
        row["dmrg_energy"] = dmrg_e
        row["diff_vs_dmrg"] = float(abs(fe - dmrg_e))
    return row


def run_benchmark(config: BenchmarkConfig) -> dict:
    """Run the CPU (and opt-in GPU) benchmark over all cases and solvers."""
    gpu_device, gpu_skip_reason = select_device()
    device_info = gpu_device_info()
    # Force CPU-only default env when GPU not opted in (matches other scores).
    if not env_run_gpu():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    report: dict = {
        "convention": ("H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, "
                       "open boundary, complex128"),
        "dtype": str(DTYPE),
        "seed": config.seed,
        "mainline_statement": (
            "This benchmark evaluates the AD mainline solvers (global AD-MPS, "
            "one-site AD local, two-site AD local) on CPU and the machine's "
            "single GPU. DMRG / Lanczos / exact diagonalization are classical "
            "reference baselines ONLY and are NOT part of the AD optimization "
            "path; they are reported for context."
        ),
        "device_info": device_info,
        "gpu_skip_reason": gpu_skip_reason,
        "gpu_ran": gpu_skip_reason is None,
        "cases": [],
        "checks": {},
        "pass": True,
        "known_limitations": [
            "Small systems only (N=4/6, chi=4/8); large-N GPU scaling is out of scope.",
            "GPU is opt-in (LATTICETN_RUN_GPU=1) and uses cuda:0; no GPU name filtering.",
            "Runtime/speedup are recorded but the GPU is NOT required to be faster: small systems are overhead-dominated (host<->device transfer, short sweeps).",
            "ED and DMRG are CPU-only reference baselines; the GPU runs only the AD solver optimization.",
            "CPU/GPU final energies must agree within tolerance; the GPU energy must not undershoot the exact ground beyond tolerance.",
        ],
    }

    for spec in config.cases:
        exact_e0 = _exact(spec.N)
        exact_ref = run_exact_reference(spec.N)
        dmrg_ref = run_dmrg_reference(spec, seed=0)

        # --- CPU runs (AD mainline) ---
        cpu_global = _annotate(run_global_ad(spec, "cpu", config.seed),
                               exact_e0, dmrg_ref["final_energy"])
        cpu_one = _annotate(run_one_site_ad(spec, "cpu", config.seed),
                            exact_e0, dmrg_ref["final_energy"])
        cpu_two = _annotate(run_two_site_ad(spec, "cpu", config.seed),
                            exact_e0, dmrg_ref["final_energy"])

        case_record: dict = {
            "N": spec.N, "chi": spec.chi,
            "exact": exact_ref,
            "dmrg_reference": dmrg_ref,
            "cpu": [cpu_global, cpu_one, cpu_two],
            "gpu": [],
            "speedups": [],
        }

        # --- GPU runs (opt-in) ---
        if gpu_skip_reason is None:
            # Build each solver on GPU with the SAME seed, copying CPU tensors
            # by value for an apples-to-apples starting point.
            def _gpu(spec, runner, label):
                # Build MPS on CPU with the pinned seed, then clone onto GPU so
                # CPU and GPU start from identical tensors.
                tc.manual_seed(config.seed)
                mps_cpu = MPS(spec.N, 2, spec.chi, dtype=DTYPE, device="cpu")
                mps_gpu = MPS(spec.N, 2, spec.chi, dtype=DTYPE, device=gpu_device)
                _copy_mps_into(mps_gpu, mps_cpu)
                mpo_gpu = _build_mpo(spec.N, gpu_device)
                t0 = time.perf_counter()
                r = runner(mps_gpu, mpo_gpu, spec)
                runtime = time.perf_counter() - t0
                row = {
                    "solver": label, "optimizer": r.get("optimizer", "?"),
                    "N": spec.N, "chi": spec.chi, "seed": config.seed,
                    "final_energy": float(r["final_energy"]),
                    "initial_energy": float(r["initial_energy"]),
                    "runtime_s": runtime, "device": str(gpu_device),
                }
                return _annotate(row, exact_e0, dmrg_ref["final_energy"])

            gpu_global = _gpu(spec, _train_global_on_device, SOLVER_GLOBAL_AD)
            gpu_one = _gpu(spec, _train_one_site_on_device, SOLVER_ONE_SITE_AD)
            gpu_two = _gpu(spec, _train_two_site_on_device, SOLVER_TWO_SITE_AD)
            case_record["gpu"] = [gpu_global, gpu_one, gpu_two]

            # Speedups (CPU runtime / GPU runtime) per solver.
            for cpu_row, gpu_row in zip(case_record["cpu"], case_record["gpu"]):
                cpu_t = cpu_row["runtime_s"]
                gpu_t = gpu_row["runtime_s"]
                speedup = float(cpu_t / gpu_t) if gpu_t > 0 else float("inf")
                case_record["speedups"].append({
                    "solver": cpu_row["solver"],
                    "N": spec.N, "chi": spec.chi,
                    "cpu_runtime_s": cpu_t,
                    "gpu_runtime_s": gpu_t,
                    "speedup": speedup,
                    "cpu_final_energy": cpu_row["final_energy"],
                    "gpu_final_energy": gpu_row["final_energy"],
                    "cpu_gpu_energy_diff": float(abs(cpu_row["final_energy"]
                                                     - gpu_row["final_energy"])),
                })

        report["cases"].append(case_record)

    # --- checks ---
    checks: dict = {}
    for case_record in report["cases"]:
        N = case_record["N"]
        tol = ENERGY_AGREE_TOL.get(N, 1e-5)
        for cpu_row in case_record["cpu"]:
            key = (f"{cpu_row['solver']} N={N} cpu_not_below_ground "
                   f"({cpu_row['below_ground']})")
            checks[key.replace(" ", "_")] = not cpu_row["below_ground"]
        if case_record["gpu"]:
            for cpu_row, gpu_row in zip(case_record["cpu"], case_record["gpu"]):
                diff = abs(cpu_row["final_energy"] - gpu_row["final_energy"])
                checks[f"cpu_gpu_agree_{cpu_row['solver']}_N{N}".replace(" ", "_")] = (
                    diff < tol)
                checks[f"gpu_not_below_ground_{cpu_row['solver']}_N{N}".replace(" ", "_")] = (
                    not gpu_row["below_ground"])
    report["checks"] = checks
    report["pass"] = all(checks.values()) if checks else True
    return report


# Device-placed training closures used by the GPU path (call train_ad_* on an
# already-device-placed MPS/MPO; the AD modules inherit the device).

def _train_global_on_device(mps: MPS, mpo: MPO, spec: CaseSpec) -> dict:
    ad = ADVariationalMPS(mps, mpo)
    return train_ad_mps(ad, num_steps=spec.global_steps, lr=spec.global_lr,
                        optimizer="adam")


def _train_one_site_on_device(mps: MPS, mpo: MPO, spec: CaseSpec) -> dict:
    return train_ad_local(mps, mpo, num_sweeps=spec.one_site_sweeps,
                          local_steps=spec.one_site_local_steps,
                          lr=spec.one_site_lr, optimizer="lbfgs",
                          stabilization=spec.one_site_stabilization)


def _train_two_site_on_device(mps: MPS, mpo: MPO, spec: CaseSpec) -> dict:
    return train_ad_two_site(mps, mpo, num_sweeps=spec.two_site_sweeps,
                             local_steps=spec.two_site_local_steps,
                             lr=spec.two_site_lr, optimizer="lbfgs",
                             max_bond_dim=spec.two_site_max_bond, cutoff=None)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt(x, p=10):
    if isinstance(x, float):
        if x == float("inf"):
            return "inf"
        return f"{x:.{p}f}"
    return str(x)


def _e(x, p=6):
    if isinstance(x, float):
        if x == float("inf"):
            return "inf"
        return f"{x:.{p}e}"
    return str(x)


def render_report_md(report: dict) -> str:
    L: list[str] = []
    L.append("# Stage 6A CPU/GPU AD Solver Benchmark Report")
    L.append("")
    L.append("Generated by `scripts/run_ad_gpu_benchmark.py`.")
    L.append("")
    L.append("## Mainline statement")
    L.append("")
    L.append(f"> _{report['mainline_statement']}_")
    L.append("")
    L.append("The **GPU benchmark evaluates the AD mainline solvers** (global "
             "AD-MPS, one-site AD local, two-site AD local). **DMRG / Lanczos "
             "are reference baselines only**, never the AD mainline; they are "
             "reported for context and run on CPU.")
    L.append("")
    L.append("## Device info")
    L.append("")
    di = report["device_info"]
    L.append(f"- PyTorch version: `{di.get('torch_version')}`")
    L.append(f"- CUDA version: `{di.get('cuda_version')}`")
    L.append(f"- CUDA available: `{di.get('cuda_available')}`")
    L.append(f"- GPU device count: `{di.get('device_count')}`")
    L.append(f"- GPU name: `{di.get('gpu_name')}`")
    L.append(f"- Device used: `{di.get('device')}`")
    L.append(f"- dtype: `{di.get('dtype')}`")
    L.append(f"- `LATTICETN_RUN_GPU=1`: `{di.get('env_run_gpu')}`")
    L.append("")
    if report.get("gpu_skip_reason"):
        L.append(f"**GPU SKIPPED:** {report['gpu_skip_reason']}")
        L.append("")
        L.append("The GPU portion was not run (clean skip). The CPU benchmark "
                 "ran normally; the report records CPU-only results.")
        L.append("")
    else:
        L.append("**GPU RAN** on `cuda:0` (the machine's single GPU; no name "
                 "filtering).")
        L.append("")
    L.append("## Convention")
    L.append("")
    L.append(f"`{report['convention']}`")
    L.append(f"- dtype: `{report['dtype']}`")
    L.append(f"- seed: `{report['seed']}`")
    L.append("")

    # Reference baselines table (per case).
    L.append("## Reference baselines (NOT the AD mainline)")
    L.append("")
    L.append("| N | exact E0 | DMRG E | DMRG runtime_s |")
    L.append("|---:|---:|---:|---:|")
    for c in report["cases"]:
        ex = c["exact"]["final_energy"]
        dm = c["dmrg_reference"]["final_energy"]
        drt = c["dmrg_reference"]["runtime_s"]
        L.append(f"| {c['N']} | {ex:.10f} | {dm:.10f} | {drt:.3f} |")
    L.append("")
    L.append("Exact diagonalization (`numpy.linalg.eigh`) and dense DMRG are "
             "classical reference baselines; they are **not** part of the AD "
             "optimization path.")
    L.append("")

    # CPU/GPU comparison table per case.
    for c in report["cases"]:
        N = c["N"]
        chi = c["chi"]
        gpu_ran = bool(c["gpu"])
        L.append(f"## CPU/GPU comparison — N={N}, chi={chi}")
        L.append("")
        if gpu_ran:
            L.append("| solver | optimizer | device | final E | energy error | "
                     "E / bond | runtime_s | speedup | below ground |")
            L.append("|---|:---:|:---:|---:|---:|---:|---:|---:|:---:|")
            for cpu_row, gpu_row in zip(c["cpu"], c["gpu"]):
                spd = next((s for s in c["speedups"]
                            if s["solver"] == cpu_row["solver"]), None)
                speedup_s = _e(spd["speedup"], 3) if spd else "?"
                for row, dev in ((cpu_row, "cpu"), (gpu_row, str(gpu_row["device"]))):
                    L.append(f"| {row['solver']} | {row['optimizer']} | {dev} | "
                             f"{row['final_energy']:.10f} | "
                             f"{row['energy_error']:.2e} | "
                             f"{row['energy_per_bond']:.10f} | "
                             f"{row['runtime_s']:.3f} | "
                             f"{('—' if dev=='cpu' else speedup_s)} | "
                             f"{row['below_ground']} |")
            L.append("")
            L.append("Speedup = CPU runtime / GPU runtime. **The GPU is NOT "
                     "required to be faster**: small systems are "
                     "overhead-dominated (host<->device transfer, short sweeps).")
            L.append("")
        else:
            L.append("| solver | optimizer | device | final E | energy error | "
                     "E / bond | runtime_s | below ground |")
            L.append("|---|:---:|:---:|---:|---:|---:|---:|:---:|")
            for row in c["cpu"]:
                L.append(f"| {row['solver']} | {row['optimizer']} | cpu | "
                         f"{row['final_energy']:.10f} | "
                         f"{row['energy_error']:.2e} | "
                         f"{row['energy_per_bond']:.10f} | "
                         f"{row['runtime_s']:.3f} | {row['below_ground']} |")
            L.append("")
            L.append("_GPU portion skipped — see Device info above._")
            L.append("")

    # Speedup summary (only when GPU ran).
    any_gpu = any(c["speedups"] for c in report["cases"])
    if any_gpu:
        L.append("## Speedup summary (CPU runtime / GPU runtime)")
        L.append("")
        L.append("| solver | N | chi | CPU runtime_s | GPU runtime_s | "
                 "speedup | |CPU-GPU| energy diff |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for c in report["cases"]:
            for s in c["speedups"]:
                L.append(f"| {s['solver']} | {s['N']} | {s['chi']} | "
                         f"{s['cpu_runtime_s']:.3f} | "
                         f"{s['gpu_runtime_s']:.3f} | "
                         f"{_e(s['speedup'], 3)} | "
                         f"{s['cpu_gpu_energy_diff']:.2e} |")
        L.append("")

    L.append("## Overall pass/fail")
    L.append("")
    L.append(f"**pass: `{report['pass']}`**")
    L.append("")
    L.append("| check | passed |")
    L.append("|---|:---:|")
    for k, v in report["checks"].items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## Known limitations")
    L.append("")
    for lim in report["known_limitations"]:
        L.append(f"- {lim}")
    L.append("")
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Run the small --fast benchmark preset (default).")
    parser.add_argument("--list", action="store_true",
                        help="List the benchmark solvers and exit.")
    parser.add_argument("--markdown-output", type=Path,
                        default=ROOT / "docs" / "AD_GPU_BENCHMARK_REPORT.md")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("AD mainline solvers under benchmark:")
        for s in (SOLVER_GLOBAL_AD, SOLVER_ONE_SITE_AD, SOLVER_TWO_SITE_AD):
            print(f"  - {s}")
        print("Reference baselines (NOT the AD mainline):")
        for s in (SOLVER_DMRG, SOLVER_EXACT):
            print(f"  - {s}")
        print("Required docs:")
        for d in ("docs/AD_GPU_BENCHMARK_SPEC.md",
                  "docs/AD_GPU_BENCHMARK_PROTOCOL.md",
                  "docs/AD_GPU_BENCHMARK_REPORT.md",
                  "docs/CLAUDE_PROGRESS_AD_GPU_BENCHMARK.md"):
            print(f"  - {d}")
        return 0

    if not args.fast and not args.list:
        args.fast = True

    config = fast_config()
    report = run_benchmark(config)

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_report_md(report), encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, default=str),
                                    encoding="utf-8")
    if args.print:
        print(json.dumps(report, indent=2, default=str))
    print(f"ad gpu benchmark: pass={report['pass']} "
          f"gpu_ran={report['gpu_ran']} "
          f"report -> {args.markdown_output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
