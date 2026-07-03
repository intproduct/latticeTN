"""Unified GPU selector for latticeTN (Stage 7A onward).

A single, shared device-selection utility used by the fermion score scripts and
tests (and available to any future stage). It selects a GPU whose name contains
one of the allowed substrings, and **clean-skips** (returns no device) when no
matching GPU is present — it never falls back to ``cuda:0`` or any other GPU.

Selection rule (Stage 7A):
    Pick a GPU whose name (case-insensitive) contains ``"V100"`` OR
    ``"TITAN V"`` (the ``"Titan V"`` spelling also matches because the match is
    case-insensitive against ``"titan v"``). These are the GPUs admissible for
    Stage 7A CPU/GPU timing.

Opt-in:
    GPU use is opt-in via ``LATTICETN_RUN_GPU=1``. Without it, the selector
    returns a clean-skip reason and no device. With it but no CUDA / no
    matching GPU, it also clean-skips and records the reason.

Discovery:
    1. ``nvidia-smi --query-gpu=index,name --format=csv,noheader`` (preferred).
    2. Fallback to ``torch.cuda.get_device_name(i)`` for each visible device
       when ``nvidia-smi`` is unavailable.

The device actually used is resolved by RE-MATCHING the name against the
current torch-visible devices, so the returned torch logical index always
points at the matched GPU regardless of ordering differences between
``nvidia-smi`` and torch.

This module is importable from tests (it has no heavy side effects beyond
``import torch``) and from scripts. It does NOT run any GPU work on import.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

import torch as tc

# Allowed GPU name substrings (case-insensitive). A device is admissible if its
# name contains ANY of these. "Titan V" is covered by "titan v" because the
# match is case-insensitive, but we list both spellings for clarity.
ALLOWED_GPU_NAME_FILTERS: tuple[str, ...] = ("V100", "TITAN V", "Titan V")


def env_run_gpu() -> bool:
    """GPU use is opt-in via ``LATTICETN_RUN_GPU=1``."""
    return os.environ.get("LATTICETN_RUN_GPU", "") == "1"


def _matches(name: str) -> bool:
    """True if ``name`` contains any allowed filter (case-insensitive)."""
    low = name.lower()
    return any(f.lower() in low for f in ALLOWED_GPU_NAME_FILTERS)


def _nvidia_smi_gpus() -> list[dict] | None:
    """List visible GPUs via nvidia-smi, or None if nvidia-smi is unavailable.

    Each entry: ``{"index": int, "name": str}``.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    out: list[dict] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        out.append({"index": idx, "name": parts[1]})
    return out if out else None


def _torch_gpus() -> list[dict]:
    """List visible GPUs via torch.cuda (fallback when no nvidia-smi)."""
    out: list[dict] = []
    if not tc.cuda.is_available():
        return out
    for i in range(tc.cuda.device_count()):
        out.append({"index": i, "name": tc.cuda.get_device_name(i)})
    return out


def discover_gpus() -> tuple[bool, list[dict], str]:
    """Discover visible GPUs. Returns ``(cuda_available, gpus, source)``.

    ``source`` is ``"nvidia-smi"`` or ``"torch.cuda"``. ``cuda_available`` is
    ``torch.cuda.is_available()`` (CUDA build support). ``gpus`` is a list of
    ``{"index", "name"}``; empty when none visible.
    """
    cuda_available = bool(tc.cuda.is_available())
    smi = _nvidia_smi_gpus()
    if smi is not None:
        return cuda_available, smi, "nvidia-smi"
    return cuda_available, _torch_gpus(), "torch.cuda"


@dataclass
class GPUSelection:
    """Result of a GPU selection request.

    Attributes
    ----------
    device : str | None
        Torch device string (e.g. ``"cuda:0"``) when a matching GPU was found
        and use was opted-in; otherwise ``None`` (clean-skip).
    skip_reason : str | None
        Non-None when the GPU path must clean-skip. ``None`` when a device was
        selected.
    gpu_name : str | None
        Name of the selected GPU (for reporting).
    all_gpus : list[dict]
        All discovered GPUs (``{"index", "name"}``).
    matched_gpus : list[dict]
        All GPUs whose name matched the allowed filters.
    source : str
        Discovery source (``"nvidia-smi"`` or ``"torch.cuda"``).
    cuda_available : bool
        ``torch.cuda.is_available()``.
    extra : dict
        Extra reporting fields (e.g. ``{"allowed_filters": [...]}``).
    """
    device: str | None = None
    skip_reason: str | None = None
    gpu_name: str | None = None
    all_gpus: list[dict] = field(default_factory=list)
    matched_gpus: list[dict] = field(default_factory=list)
    source: str = ""
    cuda_available: bool = False
    extra: dict = field(default_factory=dict)


def select_gpu() -> GPUSelection:
    """Select a GPU whose name contains an allowed filter; else clean-skip.

    Rules:
    1. If ``LATTICETN_RUN_GPU != 1``: clean-skip (opt-in not requested).
    2. If CUDA is unavailable or no device is visible: clean-skip.
    3. Discover GPUs (nvidia-smi preferred, torch fallback).
    4. Matched = GPUs whose name contains an allowed filter.
    5. If matched is non-empty: pick the first; resolve its torch logical index
       by re-matching the name against ``torch.cuda.get_device_name(i)``.
    6. If matched is empty: clean-skip (NO fallback to cuda:0 or any other GPU).
    """
    sel = GPUSelection(extra={"allowed_filters": list(ALLOWED_GPU_NAME_FILTERS)})
    if not env_run_gpu():
        sel.skip_reason = "LATTICETN_RUN_GPU != 1; GPU path is opt-in and was not requested."
        return sel
    if not tc.cuda.is_available() or tc.cuda.device_count() == 0:
        sel.skip_reason = (
            "torch.cuda.is_available() is False or no visible CUDA device "
            "(CUDA_VISIBLE_DEVICES may hide it); GPU path clean-skips."
        )
        return sel
    cuda_available, gpus, source = discover_gpus()
    sel.all_gpus = gpus
    sel.source = source
    sel.cuda_available = cuda_available
    matched = [g for g in gpus if _matches(g["name"])]
    sel.matched_gpus = matched
    if not matched:
        sel.skip_reason = (
            "No GPU matching any of "
            f"{list(ALLOWED_GPU_NAME_FILTERS)} was found among visible devices "
            f"({[g['name'] for g in gpus]}); not falling back to any other GPU."
        )
        return sel
    chosen = matched[0]
    sel.gpu_name = chosen["name"]
    # Resolve the torch logical index by re-matching the name (so the index is
    # correct regardless of nvidia-smi vs torch ordering).
    torch_idx = None
    for i in range(tc.cuda.device_count()):
        if tc.cuda.get_device_name(i) == chosen["name"]:
            torch_idx = i
            break
    if torch_idx is None:
        # Name not found among torch-visible devices (e.g. nvidia-smi saw a
        # GPU that CUDA_VISIBLE_DEVICES hides from torch). Clean-skip rather
        # than guess an index.
        sel.skip_reason = (
            f"Matched GPU '{chosen['name']}' via {source} but it is not visible "
            "to torch.cuda (CUDA_VISIBLE_DEVICES may hide it); clean-skip."
        )
        sel.gpu_name = chosen["name"]
        return sel
    sel.device = f"cuda:{torch_idx}"
    return sel


def selection_report_dict(sel: GPUSelection) -> dict:
    """A flat dict of the selection for JSON/markdown reporting."""
    return {
        "torch_version": tc.__version__,
        "cuda_version": str(tc.version.cuda),
        "cuda_available": sel.cuda_available,
        "device_count": int(tc.cuda.device_count()) if tc.cuda.is_available() else 0,
        "gpu_name": sel.gpu_name,
        "device": sel.device,
        "source": sel.source,
        "all_gpus": sel.all_gpus,
        "matched_gpus": sel.matched_gpus,
        "allowed_filters": list(ALLOWED_GPU_NAME_FILTERS),
        "env_run_gpu": env_run_gpu(),
        "skip_reason": sel.skip_reason,
        "gpu_ran": sel.skip_reason is None,
    }


if __name__ == "__main__":
    sel = select_gpu()
    import json
    print(json.dumps(selection_report_dict(sel), indent=2, default=str))
