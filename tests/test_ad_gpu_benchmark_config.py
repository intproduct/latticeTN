"""Stage 6A benchmark config + device-selection tests (CPU-only, always run).

These tests do NOT require a GPU and do NOT run the benchmark optimization.
They verify the benchmark config parsing and the opt-in / clean-skip device
selection logic. See docs/AD_GPU_BENCHMARK_PROTOCOL.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch as tc  # noqa: E402

from run_ad_gpu_benchmark import (  # noqa: E402
    BenchmarkConfig,
    CaseSpec,
    DTYPE,
    ENERGY_AGREE_TOL,
    SOLVER_DMRG,
    SOLVER_EXACT,
    SOLVER_GLOBAL_AD,
    SOLVER_ONE_SITE_AD,
    SOLVER_TWO_SITE_AD,
    BELOW_GROUND_TOL,
    env_run_gpu,
    fast_config,
    gpu_device_info,
    select_device,
)


def test_fast_config_has_small_cases():
    """--fast preset must be small (N=4/6) so the score finishes quickly on CPU."""
    cfg = fast_config()
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.dtype == str(tc.complex128)
    assert cfg.dtype == str(DTYPE)
    ns = [c.N for c in cfg.cases]
    assert ns == [4, 6]
    for c in cfg.cases:
        assert isinstance(c, CaseSpec)
        assert c.chi in (4, 8)
        # short steps/sweeps so --fast stays fast (global AD-MPS is first-order
        # Adam and needs more steps than the LBFGS local solvers).
        assert c.global_steps <= 300
        assert c.one_site_sweeps <= 4
        assert c.two_site_sweeps <= 4
        assert c.one_site_local_steps <= 20
        assert c.two_site_local_steps <= 20


def test_solver_labels_distinguish_mainline_from_reference():
    """The three AD solvers are the mainline; DMRG/ED are reference baselines."""
    mainline = {SOLVER_GLOBAL_AD, SOLVER_ONE_SITE_AD, SOLVER_TWO_SITE_AD}
    reference = {SOLVER_DMRG, SOLVER_EXACT}
    assert mainline.isdisjoint(reference)
    assert len(mainline) == 3
    assert len(reference) == 2
    # DMRG/EXACT labels must clearly flag themselves as reference.
    for r in reference:
        assert "reference" in r.lower()


def test_tolerances_not_widened():
    """CPU/GPU agreement tolerances must match the existing AD tolerances."""
    assert ENERGY_AGREE_TOL[4] == 1e-6
    assert ENERGY_AGREE_TOL[6] == 1e-5
    assert BELOW_GROUND_TOL == 1e-6


def test_env_run_gpu_defaults_off():
    """Without LATTICETN_RUN_GPU=1 the GPU is opt-out (must be off)."""
    key = "LATTICETN_RUN_GPU"
    saved = os.environ.pop(key, None)
    try:
        os.environ.pop(key, None)
        assert env_run_gpu() is False
        os.environ[key] = "0"
        assert env_run_gpu() is False
        os.environ[key] = ""
        assert env_run_gpu() is False
        os.environ[key] = "1"
        assert env_run_gpu() is True
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


def test_select_device_clean_skip_when_not_opted_in(monkeypatch):
    """No LATTICETN_RUN_GPU=1 -> GPU portion clean-skips (device stays cpu)."""
    monkeypatch.delenv("LATTICETN_RUN_GPU", raising=False)
    device, skip = select_device()
    assert device == "cpu"
    assert skip is not None
    assert "LATTICETN_RUN_GPU" in skip


def test_select_device_cuda0_when_opted_in_and_available(monkeypatch):
    """Opted in AND a visible CUDA device -> cuda:0 (single GPU, no name filter).

    Note: the score's CPU-only env may set CUDA_VISIBLE_DEVICES="" which hides
    every device (device_count == 0) even though cuda.is_available() is True
    (build support). select_device only returns cuda:0 when a device is
    actually visible, so we gate on device_count() > 0 here too.
    """
    monkeypatch.setenv("LATTICETN_RUN_GPU", "1")
    if tc.cuda.device_count() == 0:
        pytest.skip("No visible CUDA device (CUDA_VISIBLE_DEVICES may hide it); "
                    "clean-skip path is covered by "
                    "test_select_device_clean_skip_when_cuda_unavailable")
    device, skip = select_device()
    assert device == "cuda:0"
    assert skip is None


def test_select_device_clean_skip_when_cuda_unavailable(monkeypatch):
    """Opted in but no visible CUDA device -> clean skip (device stays cpu).

    Covers both the no-CUDA-build case and the CUDA_VISIBLE_DEVICES="" case
    (build support present but no device visible).
    """
    monkeypatch.setenv("LATTICETN_RUN_GPU", "1")
    if tc.cuda.device_count() > 0:
        pytest.skip("A CUDA device is visible on this machine; the cuda:0 path "
                    "is covered by test_select_device_cuda0_when_opted_in_and_available")
    device, skip = select_device()
    assert device == "cpu"
    assert skip is not None
    assert "cuda.is_available" in skip or "CUDA" in skip


def test_gpu_device_info_records_required_fields():
    """The report must record GPU name / CUDA version / torch version / device / dtype.

    Note: ``cuda_available`` reflects CUDA *build* support and stays True even
    when ``CUDA_VISIBLE_DEVICES=""`` hides every device (device_count == 0 then);
    gpu name / device are only populated when a device is actually visible.
    """
    info = gpu_device_info()
    for key in ("torch_version", "cuda_version", "cuda_available",
                "device_count", "gpu_name", "device", "dtype", "env_run_gpu"):
        assert key in info
    assert info["dtype"] == str(DTYPE)
    assert info["cuda_available"] == bool(tc.cuda.is_available())
    n_vis = int(tc.cuda.device_count())
    if n_vis > 0:
        assert info["gpu_name"] == tc.cuda.get_device_name(0)
        assert info["device"] == "cuda:0"
        assert info["device_count"] == n_vis
    else:
        assert info["gpu_name"] is None
        assert info["device"] is None
        assert info["device_count"] == 0
