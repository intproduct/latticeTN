#!/usr/bin/env python3
"""Stage 2.5 GPU readiness smoke runner (shared logic).

This module is the single source of truth for GPU device selection and the
GPU correctness smoke checks. It is imported by:

- ``scripts/gpu_score.py`` (orchestrator that writes ``docs/GPU_REPORT.md``),
- ``tests/test_gpu_device_parity.py`` and ``tests/test_gpu_heisenberg_smoke.py``
  (opt-in pytest tests).

Design rules (see docs/GPU_TESTING_PROTOCOL.md):

- GPU tests are OPT-IN. Nothing runs unless ``LATTICETN_RUN_GPU=1`` is set.
- We never default to ``cuda:0``. We select a GPU whose name contains
  ``LATTICETN_GPU_NAME_FILTER`` (default ``"Pro 4000 Blackwell"``). If no GPU
  matches, we DO NOT fall back to another GPU: the smoke test is cleanly
  skipped and the report records that no matching GPU was found.
- GPU discovery prefers ``nvidia-smi``; if unavailable, falls back to
  ``torch.cuda.get_device_name``.

Physics conventions are unchanged from Stage 1/2:
- H = J * sum_i S_i . S_{i+1}, S = sigma/2, J = 1.0, open boundary.
- dtype complex128.

Autograd rule: the energy path (``mps.energy_with_MPO`` and the short
optimization) does NOT use ``.detach()``/``.data``/unnecessary ``.item()``.
Energy *comparison* values for the report are extracted OUTSIDE the gradient
computation as plain Python floats, only after backward; that is the report
path, not the differentiable energy path. Normalization reuses the Stage 1
``_full_normalize`` routine, which mutates ``.data`` under ``no_grad`` OUTSIDE
the energy path (allowed by CLAUDE.md, since it preserves a valid optimizer
reference).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

DEFAULT_NAME_FILTER = "Pro 4000 Blackwell"

# Reuse the Stage 1 solve-time normalization (operates under no_grad + .data,
# OUTSIDE the differentiable energy path). We import it lazily inside
# run_smoke to keep module import cheap and GPU-free.


def env_run_gpu() -> bool:
    return os.environ.get("LATTICETN_RUN_GPU", "") == "1"


def env_name_filter() -> str:
    return os.environ.get("LATTICETN_GPU_NAME_FILTER", DEFAULT_NAME_FILTER)


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------

def _nvidia_smi_gpus() -> list[dict] | None:
    """Return [{index, name, memory_total_mb}, ...] via nvidia-smi, or None."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    out = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        try:
            mem = float(parts[2])
        except ValueError:
            mem = None
        out.append({"index": idx, "name": name, "memory_total_mb": mem})
    return out if out else None


def _torch_gpus() -> list[dict]:
    """Return [{index, name, memory_total_mb}, ...] via torch.cuda."""
    import torch as tc
    out = []
    if not tc.cuda.is_available():
        return out
    for i in range(tc.cuda.device_count()):
        name = tc.cuda.get_device_name(i)
        try:
            props = tc.cuda.get_device_properties(i)
            mem = float(props.total_memory) / (1024.0 * 1024.0)
        except Exception:
            mem = None
        out.append({"index": i, "name": name, "memory_total_mb": mem})
    return out


def discover_gpus() -> tuple[bool, list[dict], str]:
    """Discover visible GPUs.

    Returns (cuda_available, gpus, source) where source is 'nvidia-smi' or
    'torch.cuda'. nvidia-smi is preferred when available; otherwise torch is
    used. torch reports real CUDA visibility (after any CUDA_VISIBLE_DEVICES
    remap), while nvidia-smi reports physical devices.
    """
    import torch as tc
    cuda_available = bool(tc.cuda.is_available())
    nv = _nvidia_smi_gpus()
    if nv is not None:
        return cuda_available, nv, "nvidia-smi"
    return cuda_available, _torch_gpus(), "torch.cuda"


def select_gpu(gpus: list[dict], name_filter: str) -> dict | None:
    """Return the first GPU whose name contains name_filter (case-insensitive).

    Returns None if none match. Does NOT fall back to other GPUs.
    """
    nf = name_filter.lower()
    for g in gpus:
        if nf and nf in g["name"].lower():
            return g
    return None


# ---------------------------------------------------------------------------
# Smoke correctness checks
# ---------------------------------------------------------------------------

def run_smoke(N: int = 4, chi: int = 4, steps: int = 20, lr: float = 1e-2,
              seed: int = 0, J: float = 1.0) -> dict:
    """Run the GPU correctness smoke checks on the selected matched GPU.

    Returns a structured report dict. This function assumes the caller has
    already gated on env_run_gpu() and selected a matched GPU via
    select_gpu(); if you call it with no matched device it will fall through to
    a CPU-only baseline (used to populate the CPU columns of the report) only
    when explicitly asked.

    The dict always carries the device-selection facts; the actual GPU
    numerical checks are only populated when a matched CUDA GPU is available.
    """
    import torch as tc
    from latticetn.mpo import MPO
    from latticetn.mps import MPS
    from latticetn.operators import heisenberg_dense, exact_ground_energy
    from run_heisenberg_small import _full_normalize

    DTYPE = tc.complex128
    name_filter = env_name_filter()
    cuda_available, gpus, source = discover_gpus()
    matched = select_gpu(gpus, name_filter)

    report: dict = {
        "cuda_available": cuda_available,
        "gpu_source": source,
        "all_gpus": gpus,
        "name_filter": name_filter,
        "matched_gpus": ([g for g in gpus
                          if name_filter.lower() in g["name"].lower()]
                         if name_filter else []),
        "used_gpu_index": None,
        "used_gpu_name": None,
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_current_device": None,
        "torch_version": tc.__version__,
        "N": N,
        "chi": chi,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "dtype": str(DTYPE),
        "cpu_energy": None,
        "gpu_energy": None,
        "cpu_gpu_energy_diff": None,
        "mpo_dense_match": None,
        "backward_ok": None,
        "all_grads_not_none": None,
        "energy_decreased": None,
        "below_exact_ground": None,
        "final_energy": None,
        "exact_energy": None,
        "no_device_mixing": None,
        "checks_pass": None,
        "skip_reason": None,
        "pass": False,
        "known_limitations": [
            "GPU smoke is a correctness check only; not a performance benchmark.",
            "MPS observables/gpu path reuses Stage 1 contractions; large-N GPU scaling is out of scope.",
            "Only GPUs matching LATTICETN_GPU_NAME_FILTER are used; other visible GPUs are deliberately ignored.",
        ],
    }

    # --- gating ----------------------------------------------------------
    if not env_run_gpu():
        report["skip_reason"] = (
            "LATTICETN_RUN_GPU != 1; GPU smoke is opt-in and was not requested."
        )
        return report

    if not cuda_available:
        report["skip_reason"] = "torch.cuda.is_available() is False; no CUDA on this machine."
        return report

    if matched is None:
        report["skip_reason"] = (
            f"No GPU matching name contains '{name_filter}' was found. "
            "Not falling back to any other GPU."
        )
        return report

    # --- run on the matched GPU only ------------------------------------
    gpu_index = matched["index"]
    # NOTE: nvidia-smi indices are PHYSICAL indices. If CUDA_VISIBLE_DEVICES is
    # already set by the caller, torch's logical indices may differ. We honor
    # the caller-set CUDA_VISIBLE_DEVICES; otherwise we set it to the matched
    # physical index so torch exposes exactly that one device as logical 0.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # CUDA_VISIBLE_DEVICES must be set BEFORE the first CUDA call to take
        # effect. We already queried device names above (first CUDA call), so
        # late-setting may not remap. We therefore fall back to addressing the
        # device by its CURRENT torch logical index, computed below.
    # Resolve the matched device to its current torch logical index by name.
    torch_logical_index = None
    for i in range(tc.cuda.device_count()):
        if tc.cuda.get_device_name(i) == matched["name"]:
            torch_logical_index = i
            break
    if torch_logical_index is None:
        # Names didn't line up (e.g. CUDA_VISIBLE_DEVICES already remapped and
        # the matched device is now hidden). Treat as a clean skip rather than
        # using cuda:0.
        report["skip_reason"] = (
            f"Matched GPU '{matched['name']}' (physical index {gpu_index}) is "
            "not visible to torch under the current CUDA_VISIBLE_DEVICES; not "
            "falling back to another GPU."
        )
        return report

    device = tc.device(f"cuda:{torch_logical_index}")
    report["used_gpu_index"] = torch_logical_index
    report["used_gpu_name"] = matched["name"]
    report["torch_current_device"] = str(device)

    tc.manual_seed(seed)

    # CPU reference build (same seed/convention).
    mps_cpu = MPS(N, 2, chi, dtype=DTYPE, device="cpu")
    mpo_cpu = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=J)

    # GPU build with the SAME initial tensors (copy by value) so the energy
    # comparison is apples-to-apples, not different RNG draws.
    mps_gpu = MPS(N, 2, chi, dtype=DTYPE, device=device)
    mpo_gpu = MPO.from_bonds(N, 2, dtype=DTYPE, device=device).generate_heisenberg(J=J)
    # copy CPU tensors onto GPU to guarantee identical starting point
    for a, b in zip(mps_gpu.tensors, mps_cpu.tensors):
        a.data = b.data.to(device=device)
    for a, b in zip(mpo_gpu.tensors, mpo_cpu.tensors):
        if b is not None:
            a.data = b.data.to(device=device)

    # Normalize both identically (Stage 1 routine; mutates .data under no_grad).
    _full_normalize(mps_cpu)
    _full_normalize(mps_gpu)

    H_cpu = heisenberg_dense(N, J=J, device="cpu")
    exact_e0, _ = exact_ground_energy(H_cpu)
    report["exact_energy"] = exact_e0

    # Check 1: CPU/GPU MPO dense matrix equality.
    H_mpo_cpu = mpo_cpu.to_dense()
    H_mpo_gpu = mpo_gpu.to_dense().to("cpu")
    mpo_match = bool(tc.allclose(H_mpo_cpu, H_mpo_gpu, atol=1e-10, rtol=1e-10))
    report["mpo_dense_match"] = mpo_match

    # Check 2: CPU/GPU random-MPS energy equality (fully differentiable path).
    e_cpu = mps_cpu.energy_with_MPO(mpo_cpu)
    e_gpu = mps_gpu.energy_with_MPO(mpo_gpu)
    report["cpu_energy"] = float(e_cpu)
    report["gpu_energy"] = float(e_gpu)
    report["cpu_gpu_energy_diff"] = float(abs(e_cpu - e_gpu))

    # Check 3: GPU backward() runs.
    # Re-evaluate on GPU for a fresh autograd graph (the comparison values above
    # were read as plain floats, NOT inside a grad path). This new scalar feeds
    # backward.
    e_gpu_bp = mps_gpu.energy_with_MPO(mpo_gpu)
    e_gpu_bp.backward()
    report["backward_ok"] = True

    # Check 4: all MPS parameters have non-None grad on GPU.
    grads_ok = all(p.grad is not None for p in mps_gpu.tensors)
    report["all_grads_not_none"] = bool(grads_ok)
    # all grads live on the GPU -> no host/device mixing on the param side
    grad_device_ok = all(
        (p.grad is None) or (p.grad.device.type == "cuda") for p in mps_gpu.tensors
    )

    # Check 7 (partial): no CPU/CUDA tensor mixing in the MPS/MPO tensors.
    mps_device_ok = all(t.device.type == "cuda" for t in mps_gpu.tensors)
    mpo_device_ok = all((w is None) or (w.device.type == "cuda")
                        for w in mpo_gpu.tensors)
    report["no_device_mixing"] = bool(mps_device_ok and mpo_device_ok
                                      and grad_device_ok)

    # Zero the grads we just created before the short optimization loop.
    for p in mps_gpu.tensors:
        if p.grad is not None:
            p.grad = None

    # Check 5: short-step variational optimization lowers the energy on GPU.
    opt = tc.optim.Adam(mps_gpu.tensors, lr=lr)
    e_start = float(mps_gpu.energy_with_MPO(mpo_gpu))
    e_prev = e_start
    e_last = e_start
    decreased = False
    for _ in range(steps):
        e = mps_gpu.energy_with_MPO(mpo_gpu)
        opt.zero_grad()
        e.backward()
        opt.step()
        _full_normalize(mps_gpu)
        e_last = float(e)
        if e_last < e_prev - 1e-9:
            decreased = True
        e_prev = e_last
    e_final = float(mps_gpu.energy_with_MPO(mpo_gpu))
    report["final_energy"] = e_final
    report["energy_decreased"] = bool(decreased and (e_final < e_start + 1e-9))

    # Check 6: GPU final energy must not fall below exact ground (beyond tol).
    below = bool(e_final < exact_e0 - 1e-6)
    report["below_exact_ground"] = below

    checks = [
        mpo_match,
        report["cpu_gpu_energy_diff"] < 1e-8,
        report["backward_ok"],
        report["all_grads_not_none"],
        report["energy_decreased"],
        (not below),
        report["no_device_mixing"],
    ]
    report["checks_pass"] = {k: v for k, v in zip([
        "mpo_dense_match",
        "cpu_gpu_energy_match",
        "backward_ok",
        "all_grads_not_none",
        "energy_decreased",
        "not_below_exact_ground",
        "no_device_mixing",
    ], checks)}
    report["pass"] = bool(all(checks))
    return report


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report_md(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Stage 2.5 GPU Readiness Report")
    lines.append("")
    lines.append("Generated by `scripts/run_gpu_smoke.py` / `scripts/gpu_score.py`.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- CUDA available: `{report.get('cuda_available')}`")
    lines.append(f"- GPU discovery source: `{report.get('gpu_source')}`")
    lines.append(f"- GPU name filter (`LATTICETN_GPU_NAME_FILTER`): "
                 f"`{report.get('name_filter')}`")
    lines.append(f"- `CUDA_VISIBLE_DEVICES` env: "
                 f"`{report.get('cuda_visible_devices_env')}`")
    lines.append(f"- PyTorch current device: `{report.get('torch_current_device')}`")
    lines.append(f"- PyTorch version: `{report.get('torch_version')}`")
    lines.append("")
    lines.append("## All visible GPUs")
    lines.append("")
    lines.append("| physical index | name | memory (MB) |")
    lines.append("|---:|---|---:|")
    for g in report.get("all_gpus") or []:
        mem = g.get("memory_total_mb")
        mem_s = f"{mem:.1f}" if isinstance(mem, (int, float)) else "?"
        lines.append(f"| {g['index']} | {g['name']} | {mem_s} |")
    lines.append("")
    lines.append("## Matched GPUs")
    lines.append("")
    matched = report.get("matched_gpus") or []
    if not matched:
        lines.append("No GPU matching name contains "
                     f"`{report.get('name_filter')}` was found.")
    else:
        lines.append("| physical index | name |")
        lines.append("|---:|---|")
        for g in matched:
            lines.append(f"| {g['index']} | {g['name']} |")
    lines.append("")
    lines.append("## GPU actually used")
    lines.append("")
    lines.append(f"- Used GPU index (torch logical): `{report.get('used_gpu_index')}`")
    lines.append(f"- Used GPU name: `{report.get('used_gpu_name')}`")
    lines.append("")
    lines.append("## Smoke run parameters")
    lines.append("")
    lines.append(f"- N: `{report.get('N')}`")
    lines.append(f"- chi: `{report.get('chi')}`")
    lines.append(f"- steps: `{report.get('steps')}`")
    lines.append(f"- lr: `{report.get('lr')}`")
    lines.append(f"- seed: `{report.get('seed')}`")
    lines.append(f"- dtype: `{report.get('dtype')}`")
    lines.append("")
    lines.append("## Numerical results")
    lines.append("")
    lines.append(f"- exact ground energy: `{report.get('exact_energy')}`")
    lines.append(f"- CPU energy: `{report.get('cpu_energy')}`")
    lines.append(f"- GPU energy: `{report.get('gpu_energy')}`")
    lines.append(f"- |CPU - GPU| energy diff: `{report.get('cpu_gpu_energy_diff')}`")
    lines.append(f"- GPU final energy (after short opt): "
                 f"`{report.get('final_energy')}`")
    lines.append(f"- MPO dense CPU/GPU match: `{report.get('mpo_dense_match')}`")
    lines.append(f"- backward() OK on GPU: `{report.get('backward_ok')}`")
    lines.append(f"- all MPS params grad not None: "
                 f"`{report.get('all_grads_not_none')}`")
    lines.append(f"- energy decreased during short opt: "
                 f"`{report.get('energy_decreased')}`")
    lines.append(f"- below exact ground energy: `{report.get('below_exact_ground')}`")
    lines.append(f"- no CPU/CUDA tensor mixing: `{report.get('no_device_mixing')}`")
    lines.append("")
    lines.append("## Per-check status")
    lines.append("")
    cp = report.get("checks_pass") or {}
    if cp:
        lines.append("| check | passed |")
        lines.append("|---|:---|")
        for k, v in cp.items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("Not run.")
    lines.append("")
    lines.append("## Overall status")
    lines.append("")
    if report.get("skip_reason"):
        lines.append(f"**SKIPPED:** {report['skip_reason']}")
        lines.append("")
        lines.append("GPU numerical tests were not run (clean skip).")
    else:
        lines.append(f"**pass: `{report.get('pass')}`**")
    lines.append("")
    lines.append("## Known limitations")
    lines.append("")
    for lim in report.get("known_limitations") or []:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)
