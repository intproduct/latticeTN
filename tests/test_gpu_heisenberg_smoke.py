"""Stage 2.5 GPU Heisenberg optimization smoke tests (OPT-IN).

Companion to tests/test_gpu_device_parity.py. Focuses on the variational
Heisenberg path on GPU: short optimization energy descent and the below-ground
guard. Same opt-in / GPU-selection rules apply (see docs/GPU_TESTING_PROTOCOL.md).
"""

from __future__ import annotations

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


def test_gpu_short_optimization_lowers_energy(smoke_report):
    assert smoke_report["energy_decreased"] is True


def test_gpu_final_energy_not_below_exact_ground(smoke_report):
    # Variational energy must not undershoot the exact ground energy beyond tol.
    assert smoke_report["below_exact_ground"] is False
    assert smoke_report["final_energy"] >= smoke_report["exact_energy"] - 1e-6


def test_gpu_smoke_overall_pass(smoke_report):
    assert smoke_report["pass"] is True
