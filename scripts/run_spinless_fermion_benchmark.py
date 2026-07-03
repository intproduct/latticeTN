#!/usr/bin/env python3
"""Stage 7A spinless-fermion t-V chain benchmark runner.

Benchmarks the three AD mainline solvers on the open-boundary 1D spinless
fermion t-V chain on CPU and (opt-in) a V100/TITAN V GPU:

1. global AD-MPS            (latticetn.ad_variational.train_ad_mps, Adam)
2. one-site AD local opt    (latticetn.ad_local.train_ad_local, LBFGS)
3. two-site AD local opt    (latticetn.ad_two_site.train_ad_two_site, LBFGS)

plus the reference baseline (NEVER the AD mainline):
4. exact diagonalization    (latticetn.operators.exact_ground_energy, CPU-only)

GPU rules (Stage 7A, unified selector — scripts/gpu_selector.py):
- GPU is OPT-IN: runs only when LATTICETN_RUN_GPU=1.
- The unified selector picks a GPU whose name contains ``V100`` or
  ``TITAN V``/``Titan V``. If none matches, the GPU portion clean-skips
  (report records the skip; CPU portion still runs; exit 0).
- Default runs are CPU-only and never require a GPU.

CPU and GPU use the SAME seed, dtype, and solver config; the GPU MPS/MPO
start from tensors copied by value from the CPU build so the comparison is
apples-to-apples. No AD loss path is modified; this runner only CALLS the
existing train_ad_* functions with device-placed MPS/MPO.

Conventions: open-boundary spinless fermion t-V chain,
    H = -t sum_i (c^d_i c_{i+1} + h.c.) + V sum_i (n_i-1/2)(n_{i+1}-1/2)
        - mu sum_i (n_i - 1/2),
d=2, complex128. ED is CPU-only reference baseline. This is 1D Jordan-Wigner
fermions, NOT a full graded fermionic tensor network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn.ad_local import train_ad_local  # noqa: E402
from latticetn.ad_two_site import train_ad_two_site  # noqa: E402
from latticetn.operators import spinless_fermion_dense, exact_ground_energy  # noqa: E402
from gpu_selector import select_gpu, selection_report_dict  # noqa: E402

DTYPE = tc.complex128

SOLVER_GLOBAL_AD = "global AD-MPS"
SOLVER_ONE_SITE_AD = "one-site AD local"
SOLVER_TWO_SITE_AD = "two-site AD local"
SOLVER_EXACT = "exact diagonalization (reference)"

ENERGY_AGREE_TOL = {4: 1e-6, 6: 1e-5}
BELOW_GROUND_TOL = 1e-6


@dataclass
class CaseSpec:
    N: int
    chi: int
    t: float
    V: float
    mu: float
    global_steps: int
    global_lr: float
    one_site_sweeps: int
    one_site_local_steps: int
    one_site_lr: float
    one_site_stabilization: str
    two_site_sweeps: int
    two_site_local_steps: int
    two_site_lr: float
    two_site_max_bond: int


@dataclass
class BenchmarkConfig:
    cases: list = field(default_factory=list)
    seed: int = 0
    dtype: str = field(default_factory=lambda: str(DTYPE))
    global_optimizer: str = "adam"
    one_site_optimizer: str = "lbfgs"
    two_site_optimizer: str = "lbfgs"


def fast_config() -> BenchmarkConfig:
    """The --fast preset: small N, short steps/sweeps, finishes quickly on CPU."""
    return BenchmarkConfig(
        cases=[
            CaseSpec(
                N=4, chi=4, t=1.0, V=0.5, mu=0.0,
                global_steps=120, global_lr=1e-2,
                one_site_sweeps=2, one_site_local_steps=10,
                one_site_lr=1.0, one_site_stabilization="qr",
                two_site_sweeps=2, two_site_local_steps=10,
                two_site_lr=1.0, two_site_max_bond=4,
            ),
            CaseSpec(
                N=6, chi=8, t=1.0, V=0.5, mu=0.0,
                global_steps=180, global_lr=1e-2,
                one_site_sweeps=3, one_site_local_steps=12,
                one_site_lr=1.0, one_site_stabilization="qr",
                two_site_sweeps=3, two_site_local_steps=12,
                two_site_lr=1.0, two_site_max_bond=8,
            ),
        ],
    )


def _exact(spec: CaseSpec) -> float:
    H = spinless_fermion_dense(spec.N, t=spec.t, V=spec.V, mu=spec.mu,
                               dtype=DTYPE, device="cpu")
    return float(exact_ground_energy(H)[0])


def _build_mps(N, chi, seed, device):
    tc.manual_seed(seed)
    return MPS(N, 2, chi, dtype=DTYPE, device=device)


def _build_mpo(spec, device):
    return MPO.from_bonds(spec.N, 2, dtype=DTYPE, device=device). \
        generate_spinless_fermion(t=spec.t, V=spec.V, mu=spec.mu)


def _copy_mps_into(dst, src):
    for a, b in zip(dst.tensors, src.tensors):
        a.data = b.data.to(device=dst.device).to(dtype=DTYPE)


def _energy_per_bond(fe, N):
    return float(fe) / (N - 1) if N > 1 else float(fe)


def run_global_ad(spec, device, seed):
    mps = _build_mps(spec.N, spec.chi, seed, device)
    mpo = _build_mpo(spec, device)
    ad = ADVariationalMPS(mps, mpo)
    t0 = time.perf_counter()
    r = train_ad_mps(ad, num_steps=spec.global_steps, lr=spec.global_lr,
                     optimizer="adam")
    runtime = time.perf_counter() - t0
    return {"solver": SOLVER_GLOBAL_AD, "optimizer": "adam",
            "N": spec.N, "chi": spec.chi, "t": spec.t, "V": spec.V,
            "mu": spec.mu, "seed": seed,
            "final_energy": float(r["final_energy"]),
            "initial_energy": float(r["initial_energy"]),
            "runtime_s": runtime, "device": str(device)}


def run_one_site_ad(spec, device, seed):
    mps = _build_mps(spec.N, spec.chi, seed, device)
    mpo = _build_mpo(spec, device)
    t0 = time.perf_counter()
    r = train_ad_local(mps, mpo, num_sweeps=spec.one_site_sweeps,
                       local_steps=spec.one_site_local_steps,
                       lr=spec.one_site_lr, optimizer="lbfgs",
                       stabilization=spec.one_site_stabilization)
    runtime = time.perf_counter() - t0
    return {"solver": SOLVER_ONE_SITE_AD, "optimizer": "lbfgs",
            "N": spec.N, "chi": spec.chi, "t": spec.t, "V": spec.V,
            "mu": spec.mu, "seed": seed,
            "final_energy": float(r["final_energy"]),
            "initial_energy": float(r["initial_energy"]),
            "runtime_s": runtime, "device": str(device)}


def run_two_site_ad(spec, device, seed):
    mps = _build_mps(spec.N, spec.chi, seed, device)
    mpo = _build_mpo(spec, device)
    t0 = time.perf_counter()
    r = train_ad_two_site(mps, mpo, num_sweeps=spec.two_site_sweeps,
                          local_steps=spec.two_site_local_steps,
                          lr=spec.two_site_lr, optimizer="lbfgs",
                          max_bond_dim=spec.two_site_max_bond, cutoff=None)
    runtime = time.perf_counter() - t0
    return {"solver": SOLVER_TWO_SITE_AD, "optimizer": "lbfgs",
            "N": spec.N, "chi": spec.chi, "t": spec.t, "V": spec.V,
            "mu": spec.mu, "seed": seed,
            "final_energy": float(r["final_energy"]),
            "initial_energy": float(r["initial_energy"]),
            "runtime_s": runtime, "device": str(device)}


def run_exact_reference(spec):
    t0 = time.perf_counter()
    e0 = _exact(spec)
    runtime = time.perf_counter() - t0
    return {"solver": SOLVER_EXACT, "optimizer": "none (numpy.linalg.eigh)",
            "N": spec.N, "final_energy": float(e0),
            "runtime_s": runtime, "device": "cpu", "is_reference": True}


def _annotate(row, exact_e0):
    fe = row["final_energy"]
    row["exact_energy"] = exact_e0
    row["energy_error"] = float(abs(fe - exact_e0))
    row["energy_per_bond"] = _energy_per_bond(fe, row["N"])
    row["below_ground"] = bool(fe < exact_e0 - BELOW_GROUND_TOL)
    return row


def _train_global_on_device(mps, mpo, spec):
    ad = ADVariationalMPS(mps, mpo)
    return train_ad_mps(ad, num_steps=spec.global_steps, lr=spec.global_lr,
                        optimizer="adam")


def _train_one_site_on_device(mps, mpo, spec):
    return train_ad_local(mps, mpo, num_sweeps=spec.one_site_sweeps,
                          local_steps=spec.one_site_local_steps,
                          lr=spec.one_site_lr, optimizer="lbfgs",
                          stabilization=spec.one_site_stabilization)


def _train_two_site_on_device(mps, mpo, spec):
    return train_ad_two_site(mps, mpo, num_sweeps=spec.two_site_sweeps,
                             local_steps=spec.two_site_local_steps,
                             lr=spec.two_site_lr, optimizer="lbfgs",
                             max_bond_dim=spec.two_site_max_bond, cutoff=None)


def run_benchmark(config: BenchmarkConfig) -> dict:
    sel = select_gpu()
    device_info = selection_report_dict(sel)
    if not sel.skip_reason:
        gpu_device = sel.device
        gpu_skip_reason = None
    else:
        gpu_device = "cpu"
        gpu_skip_reason = sel.skip_reason
    # Force CPU-only default env when GPU not opted in.
    if os.environ.get("LATTICETN_RUN_GPU", "") != "1":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    report = {
        "convention": (
            "H = -t sum_i (c^d_i c_{i+1} + h.c.) + V sum_i (n_i-1/2)(n_{i+1}-1/2)"
            " - mu sum_i (n_i-1/2); open boundary; d=2; complex128; "
            "1D Jordan-Wigner fermions (NOT graded fermionic tensors)."),
        "dtype": str(DTYPE),
        "seed": config.seed,
        "mainline_statement": (
            "This benchmark evaluates the AD mainline solvers (global AD-MPS, "
            "one-site AD local, two-site AD local) on the spinless fermion t-V "
            "chain on CPU and (opt-in) a V100/TITAN V GPU. Exact "
            "diagonalization is a classical reference baseline ONLY and is NOT "
            "part of the AD optimization path. The AD loss path is unchanged; "
            "only the Hamiltonian/MPO/operator layer is new (Stage 7A)."),
        "device_info": device_info,
        "gpu_skip_reason": gpu_skip_reason,
        "gpu_ran": gpu_skip_reason is None,
        "cases": [],
        "checks": {},
        "pass": True,
        "known_limitations": [
            "1D Jordan-Wigner fermions only; NOT a full graded fermionic tensor "
            "network. No Hubbard, no TDVP, no finite-temperature.",
            "Open-boundary spinless t-V chain only (d=2).",
            "Small systems only (N=4/6, chi=4/8); large-N scaling is out of scope.",
            "GPU is opt-in (LATTICETN_RUN_GPU=1) and selects only V100/TITAN V "
            "via the unified selector; clean-skips otherwise (no fallback).",
            "Runtime/speedup are recorded but the GPU is NOT required to be "
            "faster: small systems are overhead-dominated.",
            "ED is a CPU-only reference baseline; the GPU runs only the AD "
            "solver optimization.",
        ],
    }

    for spec in config.cases:
        exact_e0 = _exact(spec)
        exact_ref = run_exact_reference(spec)

        cpu_global = _annotate(run_global_ad(spec, "cpu", config.seed), exact_e0)
        cpu_one = _annotate(run_one_site_ad(spec, "cpu", config.seed), exact_e0)
        cpu_two = _annotate(run_two_site_ad(spec, "cpu", config.seed), exact_e0)

        case_record = {
            "N": spec.N, "chi": spec.chi, "t": spec.t, "V": spec.V,
            "mu": spec.mu,
            "exact": exact_ref,
            "cpu": [cpu_global, cpu_one, cpu_two],
            "gpu": [],
            "speedups": [],
        }

        if gpu_skip_reason is None:
            def _gpu(spec, runner, label):
                tc.manual_seed(config.seed)
                mps_cpu = MPS(spec.N, 2, spec.chi, dtype=DTYPE, device="cpu")
                mps_gpu = MPS(spec.N, 2, spec.chi, dtype=DTYPE,
                              device=gpu_device)
                _copy_mps_into(mps_gpu, mps_cpu)
                mpo_gpu = _build_mpo(spec, gpu_device)
                t0 = time.perf_counter()
                r = runner(mps_gpu, mpo_gpu, spec)
                runtime = time.perf_counter() - t0
                row = {"solver": label, "optimizer": r.get("optimizer", "?"),
                       "N": spec.N, "chi": spec.chi, "t": spec.t, "V": spec.V,
                       "mu": spec.mu, "seed": config.seed,
                       "final_energy": float(r["final_energy"]),
                       "initial_energy": float(r["initial_energy"]),
                       "runtime_s": runtime, "device": str(gpu_device)}
                return _annotate(row, exact_e0)

            gpu_global = _gpu(spec, _train_global_on_device, SOLVER_GLOBAL_AD)
            gpu_one = _gpu(spec, _train_one_site_on_device, SOLVER_ONE_SITE_AD)
            gpu_two = _gpu(spec, _train_two_site_on_device, SOLVER_TWO_SITE_AD)
            case_record["gpu"] = [gpu_global, gpu_one, gpu_two]
            for cpu_row, gpu_row in zip(case_record["cpu"], case_record["gpu"]):
                cpu_t = cpu_row["runtime_s"]
                gpu_t = gpu_row["runtime_s"]
                speedup = float(cpu_t / gpu_t) if gpu_t > 0 else float("inf")
                case_record["speedups"].append({
                    "solver": cpu_row["solver"], "N": spec.N, "chi": spec.chi,
                    "cpu_runtime_s": cpu_t, "gpu_runtime_s": gpu_t,
                    "speedup": speedup,
                    "cpu_final_energy": cpu_row["final_energy"],
                    "gpu_final_energy": gpu_row["final_energy"],
                    "cpu_gpu_energy_diff": float(abs(
                        cpu_row["final_energy"] - gpu_row["final_energy"])),
                })

        report["cases"].append(case_record)

    checks = {}
    for c in report["cases"]:
        N = c["N"]
        tol = ENERGY_AGREE_TOL.get(N, 1e-5)
        for cpu_row in c["cpu"]:
            checks[f"{cpu_row['solver']}_N{N}_cpu_not_below_ground".replace(" ", "_")] = (
                not cpu_row["below_ground"])
        if c["gpu"]:
            for cpu_row, gpu_row in zip(c["cpu"], c["gpu"]):
                diff = abs(cpu_row["final_energy"] - gpu_row["final_energy"])
                checks[f"cpu_gpu_agree_{cpu_row['solver']}_N{N}".replace(" ", "_")] = (diff < tol)
                checks[f"gpu_not_below_ground_{cpu_row['solver']}_N{N}".replace(" ", "_")] = (
                    not gpu_row["below_ground"])
    report["checks"] = checks
    report["pass"] = all(checks.values()) if checks else True
    return report


def _e(x, p=6):
    if isinstance(x, float):
        return "inf" if x == float("inf") else f"{x:.{p}e}"
    return str(x)


def render_report_md(report: dict) -> str:
    L = []
    L.append("# Stage 7A Spinless Fermion (t-V) Chain Benchmark Report")
    L.append("")
    L.append("Generated by `scripts/run_spinless_fermion_benchmark.py`.")
    L.append("")
    L.append("## Mainline statement")
    L.append("")
    L.append(f"> _{report['mainline_statement']}_")
    L.append("")
    L.append("This is **1D Jordan-Wigner fermions**, NOT a full graded "
             "fermionic tensor network. The JW parity string `F = (-1)^n` is "
             "the key. The AD loss path is unchanged; Stage 7A adds only the "
             "Hamiltonian/MPO/operator layer. Exact diagonalization is a "
             "reference baseline only.")
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
    L.append(f"- Allowed GPU filters: `{di.get('allowed_filters')}`")
    L.append(f"- Discovery source: `{di.get('source')}`")
    L.append(f"- All GPUs: `{di.get('all_gpus')}`")
    L.append(f"- Matched GPUs: `{di.get('matched_gpus')}`")
    L.append("")
    if report.get("gpu_skip_reason"):
        L.append(f"**GPU SKIPPED:** {report['gpu_skip_reason']}")
        L.append("")
        L.append("The GPU portion was not run (clean skip). The CPU benchmark "
                 "ran normally; the report records CPU-only results.")
        L.append("")
    else:
        L.append(f"**GPU RAN** on `{di.get('device')}` "
                 f"(`{di.get('gpu_name')}`).")
        L.append("")
    L.append("## Convention")
    L.append("")
    L.append(f"`{report['convention']}`")
    L.append(f"- dtype: `{report['dtype']}`")
    L.append(f"- seed: `{report['seed']}`")
    L.append("")

    L.append("## Reference baseline (NOT the AD mainline)")
    L.append("")
    L.append("| N | t | V | mu | exact E0 | ED runtime_s |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for c in report["cases"]:
        ex = c["exact"]["final_energy"]
        L.append(f"| {c['N']} | {c['t']} | {c['V']} | {c['mu']} | "
                 f"{ex:.10f} | {c['exact']['runtime_s']:.3f} |")
    L.append("")
    L.append("Exact diagonalization (`numpy.linalg.eigh`) is a classical "
             "reference baseline; it is **not** part of the AD optimization "
             "path. Reference baselines only — never the AD mainline.")
    L.append("")

    for c in report["cases"]:
        N = c["N"]
        chi = c["chi"]
        gpu_ran = bool(c["gpu"])
        L.append(f"## CPU/GPU comparison — N={N}, chi={chi}, "
                 f"t={c['t']}, V={c['V']}, mu={c['mu']}")
        L.append("")
        if gpu_ran:
            L.append("| solver | optimizer | device | final E | energy error | "
                     "E / bond | runtime_s | speedup | below ground |")
            L.append("|---|:---:|:---:|---:|---:|---:|---:|---:|:---:|")
            for cpu_row, gpu_row in zip(c["cpu"], c["gpu"]):
                spd = next((s for s in c["speedups"]
                            if s["solver"] == cpu_row["solver"]), None)
                speedup_s = _e(spd["speedup"], 3) if spd else "?"
                for row, dev in ((cpu_row, "cpu"),
                                 (gpu_row, str(gpu_row["device"]))):
                    L.append(f"| {row['solver']} | {row['optimizer']} | {dev} | "
                             f"{row['final_energy']:.10f} | "
                             f"{row['energy_error']:.2e} | "
                             f"{row['energy_per_bond']:.10f} | "
                             f"{row['runtime_s']:.3f} | "
                             f"{('—' if dev == 'cpu' else speedup_s)} | "
                             f"{row['below_ground']} |")
            L.append("")
            L.append("Speedup = CPU runtime / GPU runtime. **The GPU is NOT "
                     "required to be faster**: small systems are "
                     "overhead-dominated.")
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
                        default=ROOT / "docs" / "FERMION_REPORT.md")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("AD mainline solvers under benchmark:")
        for s in (SOLVER_GLOBAL_AD, SOLVER_ONE_SITE_AD, SOLVER_TWO_SITE_AD):
            print(f"  - {s}")
        print("Reference baseline (NOT the AD mainline):")
        print(f"  - {SOLVER_EXACT}")
        print("Required docs:")
        for d in ("docs/FERMION_SPEC.md", "docs/FERMION_PROTOCOL.md",
                  "docs/FERMION_REPORT.md", "docs/CLAUDE_PROGRESS_FERMION.md",
                  "docs/GPU_TESTING_PROTOCOL.md"):
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
    print(f"fermion benchmark: pass={report['pass']} "
          f"gpu_ran={report['gpu_ran']} "
          f"report -> {args.markdown_output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
