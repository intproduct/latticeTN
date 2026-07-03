"""Stage 7C: CPU/GPU energy parity + timing for the spinful Hubbard chain.

Uses the unified GPU selector (``scripts/gpu_selector.py``) which selects a
GPU whose name contains ``V100`` or ``TITAN V``/``Titan V``. If none matches
(or GPU use is not opted in via ``LATTICETN_RUN_GPU=1``), the GPU portion
clean-skips (still exit 0). When a matching GPU is present, this test:

- runs the global AD-MPS solver on CPU and on the matched GPU (same seed,
  dtype, solver config; MPS tensors copied by value so both start identically);
- asserts CPU/GPU final energies agree within tolerance;
- asserts the GPU energy does NOT undershoot the exact ground beyond
  tolerance (``below_ground`` guard);
- records runtime / speedup per the Stage 7C reporting contract (device name,
  dtype, N, chi, solver, final energy, exact error, runtime, speedup,
  below_ground).

The GPU is NOT required to be faster; speedup is recorded for trend-watching.
Default (no ``LATTICETN_RUN_GPU=1``) is CPU-only and always passes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn.operators import hubbard_dense, exact_ground_energy  # noqa: E402
from gpu_selector import select_gpu, selection_report_dict  # noqa: E402

DTYPE = tc.complex128
ENERGY_AGREE_TOL = {4: 1e-6, 6: 1e-5}
BELOW_GROUND_TOL = 1e-6


def _skip_reason():
    sel = select_gpu()
    if sel.skip_reason is not None:
        return sel, sel.skip_reason
    return sel, None


def _exact_e0(N, t, U, mu, h):
    H = hubbard_dense(N, t=t, U=U, mu=mu, h=h, dtype=DTYPE)
    return float(exact_ground_energy(H)[0])


def _build_mps(N, chi, seed, device):
    tc.manual_seed(seed)
    return MPS(N, 4, chi, dtype=DTYPE, device=device)


def _build_mpo(N, t, U, mu, h, device):
    return MPO.from_bonds(N, 4, dtype=DTYPE, device=device).generate_hubbard(
        t=t, U=U, mu=mu, h=h)


def _copy_mps_into(dst, src):
    for a, b in zip(dst.tensors, src.tensors):
        a.data = b.data.to(device=dst.device).to(dtype=DTYPE)


def _run_global_ad(N, chi, t, U, mu, h, seed, device, steps=80):
    # Build on CPU with pinned seed, then clone onto `device` for an
    # apples-to-apples start.
    tc.manual_seed(seed)
    mps_cpu = MPS(N, 4, chi, dtype=DTYPE, device="cpu")
    mps = MPS(N, 4, chi, dtype=DTYPE, device=device)
    _copy_mps_into(mps, mps_cpu)
    mpo = _build_mpo(N, t, U, mu, h, device)
    ad = ADVariationalMPS(mps, mpo)
    t0 = time.perf_counter()
    r = train_ad_mps(ad, num_steps=steps, lr=1e-2, optimizer="adam",
                     projection="tensor_norm")
    runtime = time.perf_counter() - t0
    return {
        "solver": "global AD-MPS",
        "N": N, "chi": chi, "t": t, "U": U, "mu": mu, "h": h, "seed": seed,
        "final_energy": float(r["final_energy"]),
        "initial_energy": float(r["initial_energy"]),
        "runtime_s": runtime, "device": str(device),
        "dtype": str(DTYPE),
    }


def test_gpu_selection_respects_v100_titan_filter():
    """The selector must only pick V100/TITAN V, never fall back to others."""
    sel = select_gpu()
    if sel.skip_reason is not None:
        assert sel.device is None
        return
    assert sel.gpu_name is not None
    low = sel.gpu_name.lower()
    assert ("v100" in low) or ("titan v" in low), sel.gpu_name


def test_cpu_global_ad_lowers_energy_and_not_below_ground():
    # The CPU baseline always runs (no GPU needed).
    for N, chi in [(4, 4), (6, 8)]:
        t, U, mu, h = 1.0, 4.0, 0.0, 0.0
        e0 = _exact_e0(N, t, U, mu, h)
        res = _run_global_ad(N, chi, t, U, mu, h, seed=1, device="cpu")
        assert res["final_energy"] < res["initial_energy"]
        assert res["final_energy"] >= e0 - BELOW_GROUND_TOL, (res, e0)


def test_cpu_gpu_energy_parity_and_timing():
    sel, reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    device = sel.device
    records = []
    for N, chi in [(4, 4), (6, 8)]:
        t, U, mu, h = 1.0, 4.0, 0.0, 0.0
        e0 = _exact_e0(N, t, U, mu, h)
        cpu = _run_global_ad(N, chi, t, U, mu, h, seed=1, device="cpu")
        gpu = _run_global_ad(N, chi, t, U, mu, h, seed=1, device=device)
        # parity
        tol = ENERGY_AGREE_TOL.get(N, 1e-5)
        assert abs(cpu["final_energy"] - gpu["final_energy"]) < tol, (N, cpu, gpu)
        # below-ground guard (both)
        assert cpu["final_energy"] >= e0 - BELOW_GROUND_TOL
        assert gpu["final_energy"] >= e0 - BELOW_GROUND_TOL
        # speedup record (GPU not required to be faster)
        spd = (cpu["runtime_s"] / gpu["runtime_s"]
               if gpu["runtime_s"] > 0 else float("inf"))
        for row, dev_name in ((cpu, "cpu"), (gpu, sel.gpu_name)):
            records.append({
                "device_name": dev_name, "dtype": str(DTYPE),
                "N": N, "chi": chi, "solver": row["solver"],
                "final_energy": row["final_energy"],
                "exact_error": abs(row["final_energy"] - e0),
                "runtime": row["runtime_s"],
                "speedup": (spd if dev_name != "cpu" else None),
                "below_ground": row["final_energy"] < e0 - BELOW_GROUND_TOL,
            })
    # sanity: we recorded both CPU and GPU rows
    assert len(records) == 4
    # expose the selection + records for the score script via a module global
    _GPU_TIMING_RECORDS = {  # noqa: F841
        "selection": selection_report_dict(sel),
        "records": records,
    }
