"""Stage 3B native MPO energy contraction tests.

Verifies the native MPO expectation / Rayleigh energy against:
- the Stage 1 ``MPS.energy_with_MPO`` path,
- the dense-state energy,
and checks gradients flow through the differentiable native energy path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn import contractions as K  # noqa: E402
from latticetn.operators import heisenberg_dense  # noqa: E402

DTYPE = tc.complex128


def _mps_and_mpo(N=6, chi=4, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return mps, mpo


def test_native_rayleigh_matches_stage1_energy_path():
    mps, mpo = _mps_and_mpo()
    e_classic = float(mps.energy_with_MPO(mpo))
    e_native = float(K.rayleigh_energy_native(mps, mpo))
    assert abs(e_classic - e_native) < 1e-9


def test_native_rayleigh_matches_dense_state_energy():
    mps, mpo = _mps_and_mpo()
    psi = mps.to_dense().detach()
    H = heisenberg_dense(mps.N, dtype=DTYPE)
    # <psi|H|psi> / <psi|psi> on the dense state
    e_dense = complex(psi.conj() @ H @ psi).real / complex(psi.conj() @ psi).real
    e_native = float(K.rayleigh_energy_native(mps, mpo))
    assert abs(e_dense - e_native) < 1e-9


def test_native_mpo_numerator_matches_dense_numerator():
    mps, mpo = _mps_and_mpo()
    psi = mps.to_dense().detach()
    H = heisenberg_dense(mps.N, dtype=DTYPE)
    num_dense = complex(psi.conj() @ H @ psi)
    num_native = complex(K.native_mpo_numerator(mps, mpo))
    assert abs(num_dense - num_native) < 1e-6


def test_native_energy_backward_grads_not_none():
    mps, mpo = _mps_and_mpo()
    e = K.rayleigh_energy_native(mps, mpo)
    e.backward()
    assert all(p.grad is not None for p in mps.tensors)


def test_native_energy_does_not_touch_to_dense_or_no_grad_on_path():
    # The differentiable native energy path must not introduce no_grad. We only
    # sanity-check it returns a differentiable scalar with a grad_fn.
    mps, mpo = _mps_and_mpo()
    e = K.rayleigh_energy_native(mps, mpo)
    assert e.requires_grad
    assert e.grad_fn is not None
