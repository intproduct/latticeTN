"""Opt-in traditional TDVP CPU/GPU parity smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gpu_selector import select_gpu  # noqa: E402
from latticetn.initial_states import neel_spin_state  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.tdvp import TDVP  # noqa: E402


def test_tdvp_cpu_gpu_device_parity_opt_in():
    selection = select_gpu()
    if selection.skip_reason is not None:
        pytest.skip(selection.skip_reason)
    assert selection.device is not None

    def evolve(device: str):
        initial = neel_spin_state(4, dtype=tc.complex128, device=device)
        mpo = MPO.from_bonds(4, 2, dtype=tc.complex128, device=device).generate_heisenberg()
        return TDVP(
            initial,
            mpo,
            dt=0.01,
            method="two_site",
            device=device,
            max_bond_dim=4,
            truncation_tol=0.0,
        ).evolve(steps=2)

    cpu = evolve("cpu")
    gpu = evolve(selection.device)
    assert all(tensor.device.type == "cuda" for tensor in gpu.mps.tensors)
    cpu_state = cpu.mps.to_dense()
    gpu_state = gpu.mps.to_dense().cpu()
    phase = tc.vdot(cpu_state, gpu_state)
    phase = phase / abs(phase)
    assert tc.allclose(cpu_state * phase, gpu_state, atol=1e-9, rtol=1e-9)
    assert abs(cpu.energy_history[-1] - gpu.energy_history[-1]) < 1e-9
