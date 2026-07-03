"""Stage 7B: unified benchmark registry (CPU/GPU timing) tests.

``latticetn.benchmarking.benchmark_model`` records the Stage-7A+ timing
contract (model, N, chi, solver, device, device_name, dtype, runtime, speedup,
final_energy, exact_error, below_ground, gpu_skip_reason) on CPU and (opt-in)
a V100/TITAN V GPU via the unified ``scripts/gpu_selector.py``. With no
``LATTICETN_RUN_GPU=1`` (or no matching GPU), the GPU portion clean-skips
(still exit 0). The GPU is NOT required to be faster; speedup is recorded only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.model_builder import (  # noqa: E402
    heisenberg_model, spinless_fermion_tv_model,
)
from latticetn.benchmarking import benchmark_model, RunRecord  # noqa: E402
from gpu_selector import select_gpu  # noqa: E402

DTYPE = tc.complex128


REQUIRED_CPU_FIELDS = {
    "model", "N", "chi", "solver", "device", "device_name", "dtype",
    "runtime", "speedup", "final_energy", "exact_error", "below_ground",
    "gpu_skip_reason",
}


def test_benchmark_model_records_cpu_timing():
    spec = heisenberg_model(4, J=1.0)
    r = benchmark_model(spec, chi=4, seed=0, steps=40)
    # CPU record always present
    assert r["cpu"] is not None
    for f in REQUIRED_CPU_FIELDS:
        assert f in r["cpu"], f
    assert r["cpu"]["device"] == "cpu"
    assert r["cpu"]["device_name"] == "cpu"
    assert r["cpu"]["dtype"] == str(DTYPE)
    assert r["cpu"]["runtime"] > 0
    assert r["cpu"]["solver"] == "global AD-MPS"
    # exact_error and below_ground recorded
    assert r["cpu"]["exact_error"] >= 0
    assert isinstance(r["cpu"]["below_ground"], bool)
    assert r["exact_energy"] is not None


def test_benchmark_model_both_presets_run():
    for spec in [heisenberg_model(4, J=1.0),
                 spinless_fermion_tv_model(4, t=1.0, V=0.5, mu=0.0)]:
        r = benchmark_model(spec, chi=4, seed=0, steps=40)
        assert r["model"] == spec.name
        assert r["N"] == 4
        assert r["cpu"]["final_energy"] is not None


def test_benchmark_model_cpu_not_below_ground():
    for spec in [heisenberg_model(4, J=1.0),
                 spinless_fermion_tv_model(4, t=1.0, V=0.5, mu=0.0)]:
        r = benchmark_model(spec, chi=4, seed=0, steps=60)
        assert not r["cpu"]["below_ground"], (spec.name, r["cpu"])


def test_benchmark_model_gpu_clean_skip_when_not_opted_in():
    if os.environ.get("LATTICETN_RUN_GPU", "") == "1":
        pytest.skip("GPU opted in; the clean-skip path is not exercised.")
    spec = heisenberg_model(4, J=1.0)
    r = benchmark_model(spec, chi=4, seed=0, steps=40)
    # When not opted in, GPU must clean-skip with a reason.
    assert r["gpu"] is None
    assert r["gpu_skip_reason"] is not None
    assert r["gpu_ran"] is False


def test_benchmark_model_gpu_parity_and_timing_when_opted_in():
    sel = select_gpu()
    if sel.skip_reason is not None:
        pytest.skip(sel.skip_reason)
    spec = heisenberg_model(4, J=1.0)
    r = benchmark_model(spec, chi=4, seed=0, steps=60)
    assert r["gpu"] is not None
    assert r["gpu"]["device"].startswith("cuda")
    assert r["gpu"]["device_name"] is not None
    # parity within tolerance
    tol = 1e-6
    assert abs(r["cpu"]["final_energy"] - r["gpu"]["final_energy"]) < tol
    # GPU not below ground
    assert not r["gpu"]["below_ground"]
    # speedup recorded (not required > 1)
    assert r["speedup"] is not None
    assert r["speedup"] > 0
    # the GPU name must be V100/TITAN V
    low = (r["gpu"]["device_name"] or "").lower()
    assert ("v100" in low) or ("titan v" in low)
