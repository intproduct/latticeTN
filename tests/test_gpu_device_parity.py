"""Stage 2.5 GPU device-parity smoke tests (OPT-IN).

These tests are NOT part of the default validation/benchmark paths. They only
run when the environment variable ``LATTICETN_RUN_GPU=1`` is set, and only on a
GPU whose name contains ``LATTICETN_GPU_NAME_FILTER`` (default
``"Pro 4000 Blackwell"``). They never default to ``cuda:0``.

Without ``LATTICETN_RUN_GPU=1``, without CUDA, or with no matching GPU, every
test in this module cleanly skips (so a plain ``pytest`` run on a CPU-only or
no-matching-GPU machine is unaffected).

See docs/GPU_TESTING_PROTOCOL.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gpu_smoke import (  # noqa: E402
    discover_gpus,
    env_name_filter,
    env_run_gpu,
    run_smoke,
    select_gpu,
)


def _skip_reason():
    if not env_run_gpu():
        return "LATTICETN_RUN_GPU != 1; GPU tests are opt-in."
    cuda_available, gpus, _ = discover_gpus()
    if not cuda_available:
        return "torch.cuda.is_available() is False."
    matched = select_gpu(gpus, env_name_filter())
    if matched is None:
        return (f"No GPU matching name contains '{env_name_filter()}' was found; "
                "not falling back to any other GPU.")
    return None


@pytest.fixture(scope="module")
def smoke_report():
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason, allow_module_level=False)
    return run_smoke(N=4, chi=4, steps=20, lr=1e-2, seed=0)


def test_gpu_opt_in_or_skip(smoke_report):
    # Reaching here means the smoke ran on a matched GPU.
    assert smoke_report is not None


def test_used_a_matching_gpu_not_cuda0(smoke_report):
    import torch as tc
    # The runner must have selected the name-matched GPU, not blindly cuda:0.
    assert smoke_report["used_gpu_name"] is not None
    name_filter = env_name_filter().lower()
    assert name_filter in smoke_report["used_gpu_name"].lower()
    # Confirm the device is a CUDA device.
    assert tc.device(smoke_report["torch_current_device"]).type == "cuda"


def test_cpu_gpu_mpo_dense_match(smoke_report):
    assert smoke_report["mpo_dense_match"] is True


def test_cpu_gpu_energy_match(smoke_report):
    diff = smoke_report["cpu_gpu_energy_diff"]
    assert diff is not None and diff < 1e-8


def test_backward_runs_on_gpu(smoke_report):
    assert smoke_report["backward_ok"] is True


def test_all_mps_params_have_grad(smoke_report):
    assert smoke_report["all_grads_not_none"] is True


def test_no_cpu_cuda_mixed_tensors(smoke_report):
    assert smoke_report["no_device_mixing"] is True
