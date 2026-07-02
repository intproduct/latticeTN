"""Stage 6A benchmark smoke tests.

Runs the actual benchmark at the smallest preset. The CPU portion always runs
(CPU-only, fast). The GPU parity assertions clean-skip unless
``LATTICETN_RUN_GPU=1`` AND CUDA is available; when they run, CPU/GPU final
energies must agree within tolerance and the GPU energy must not undershoot the
exact ground beyond tolerance.

See docs/AD_GPU_BENCHMARK_PROTOCOL.md.
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
    ENERGY_AGREE_TOL,
    SOLVER_DMRG,
    SOLVER_EXACT,
    SOLVER_GLOBAL_AD,
    SOLVER_ONE_SITE_AD,
    SOLVER_TWO_SITE_AD,
    BELOW_GROUND_TOL,
    env_run_gpu,
    fast_config,
    run_benchmark,
)


def _gpu_reason():
    """Return None if the GPU benchmark should run, else a skip reason.

    Matches select_device(): opt-in AND a visible CUDA device. cuda.is_available()
    alone is insufficient because the score's CPU-only env may set
    CUDA_VISIBLE_DEVICES="" which hides every device (device_count == 0).
    """
    if not env_run_gpu():
        return "LATTICETN_RUN_GPU != 1; GPU benchmark is opt-in."
    if not tc.cuda.is_available() or tc.cuda.device_count() == 0:
        return ("torch.cuda.is_available() is False or no visible CUDA device "
                "(CUDA_VISIBLE_DEVICES may hide it); GPU benchmark clean-skips.")
    return None


@pytest.fixture(scope="module")
def benchmark_report():
    """Run the full --fast benchmark once (CPU always; GPU if opted-in)."""
    # Make the run deterministic and CPU-thread-stable.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    return run_benchmark(fast_config())


# --- CPU results (always run) ----------------------------------------------

def test_cpu_report_has_both_cases(benchmark_report):
    cases = benchmark_report["cases"]
    assert [c["N"] for c in cases] == [4, 6]
    for c in cases:
        assert c["chi"] in (4, 8)


def test_cpu_ad_solvers_finish_with_finite_energy(benchmark_report):
    """Each AD mainline solver must finish with a finite final energy on CPU."""
    for c in benchmark_report["cases"]:
        cpu_rows = c["cpu"]
        assert len(cpu_rows) == 3
        solvers = {r["solver"] for r in cpu_rows}
        assert solvers == {SOLVER_GLOBAL_AD, SOLVER_ONE_SITE_AD, SOLVER_TWO_SITE_AD}
        for row in cpu_rows:
            assert row["final_energy"] == row["final_energy"]  # not NaN
            assert tc.isfinite(tc.tensor(row["final_energy"])).item()
            assert row["runtime_s"] > 0.0
            assert row["device"] == "cpu"


def test_cpu_runtime_and_speedup_fields_present(benchmark_report):
    """runtime_s must exist on every row; speedup must exist when GPU ran."""
    for c in benchmark_report["cases"]:
        for row in c["cpu"]:
            assert "runtime_s" in row
            assert "energy_per_bond" in row
            assert "energy_error" in row
            assert "below_ground" in row
        if c["speedups"]:
            for s in c["speedups"]:
                assert "cpu_runtime_s" in s
                assert "gpu_runtime_s" in s
                assert "speedup" in s
                assert s["gpu_runtime_s"] > 0.0


def test_cpu_ad_not_below_ground(benchmark_report):
    """CPU AD final energies must not undershoot the exact ground beyond tol."""
    for c in benchmark_report["cases"]:
        for row in c["cpu"]:
            assert row["below_ground"] is False, (
                f"{row['solver']} N={c['N']} below ground: "
                f"{row['final_energy']} < {row['exact_energy']}")


def test_reference_baselines_present_and_flagged(benchmark_report):
    """ED and DMRG reference baselines must be present and flagged reference."""
    for c in benchmark_report["cases"]:
        ex = c["exact"]
        dm = c["dmrg_reference"]
        assert ex["solver"] == SOLVER_EXACT
        assert dm["solver"] == SOLVER_DMRG
        assert ex["is_reference"] is True
        assert dm["is_reference"] is True
        assert ex["device"] == "cpu"
        assert dm["device"] == "cpu"


def test_mainline_statement_mentions_ad_and_reference(benchmark_report):
    s = benchmark_report["mainline_statement"].lower()
    assert "ad mainline" in s
    assert "reference baselines" in s


# --- GPU parity (opt-in; clean-skip otherwise) -----------------------------

def test_gpu_parity_when_run(benchmark_report):
    """When the GPU ran: CPU/GPU final energies agree and GPU not below ground."""
    reason = _gpu_reason()
    if reason is not None:
        pytest.skip(reason)
    assert benchmark_report["gpu_ran"] is True
    assert benchmark_report["gpu_skip_reason"] is None
    for c in benchmark_report["cases"]:
        assert len(c["gpu"]) == 3
        N = c["N"]
        tol = ENERGY_AGREE_TOL.get(N, 1e-5)
        for cpu_row, gpu_row in zip(c["cpu"], c["gpu"]):
            diff = abs(cpu_row["final_energy"] - gpu_row["final_energy"])
            assert diff < tol, (
                f"{cpu_row['solver']} N={N}: |CPU-GPU| energy diff {diff:.2e} "
                f">= tol {tol:.0e}")
            assert gpu_row["below_ground"] is False, (
                f"{cpu_row['solver']} N={N} GPU below ground: "
                f"{gpu_row['final_energy']} < {gpu_row['exact_energy']}")
            assert gpu_row["device"] == "cuda:0"
            assert gpu_row["runtime_s"] > 0.0


def test_gpu_clean_skip_when_not_run(benchmark_report):
    """When the GPU did NOT run: gpu_ran is False and a skip reason is recorded."""
    reason = _gpu_reason()
    if reason is None:
        pytest.skip("GPU benchmark is opted-in and CUDA is available; the "
                    "clean-skip path is not exercised here.")
    assert benchmark_report["gpu_ran"] is False
    assert benchmark_report["gpu_skip_reason"] is not None
    for c in benchmark_report["cases"]:
        assert c["gpu"] == []
        assert c["speedups"] == []


def test_overall_pass_flag(benchmark_report):
    """The benchmark's overall pass flag must be True (CPU always; GPU parity
    only checked when it ran, which is reflected in the checks dict)."""
    assert benchmark_report["pass"] is True
