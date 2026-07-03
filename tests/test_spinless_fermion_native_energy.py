"""Stage 7A: native (MPS/MPO contraction) Rayleigh energy == dense energy for
the spinless fermion t-V chain.

The native differentiable contraction ``contractions.rayleigh_energy_native``
must compute the same Rayleigh quotient ``<psi|H|psi>/<psi|psi>`` as the dense
state-vector energy, on a random MPS and the fermion MPO. This confirms the
fermion MPO plugs into the existing AD mainline loss path unchanged (the loss
path is operator-agnostic; only the Hamiltonian/MPO layer is new).
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
from latticetn.operators import spinless_fermion_dense  # noqa: E402

DTYPE = tc.complex128


def _mps_mpo(N, chi, t, V, mu, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_spinless_fermion(
        t=t, V=V, mu=mu)
    return mps, mpo


def _dense_energy(psi, H):
    num = psi.conj() @ H @ psi
    den = psi.conj() @ psi
    return (num / den).real


def test_native_rayleigh_matches_dense_energy():
    for N in [2, 3, 4, 6]:
        for (t, V, mu) in [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.3)]:
            mps, mpo = _mps_mpo(N, 8, t, V, mu)
            H = spinless_fermion_dense(N, t=t, V=V, mu=mu, dtype=DTYPE)
            e_native = float(K.rayleigh_energy_native(mps, mpo))
            e_dense = float(_dense_energy(mps.to_dense(), H))
            assert abs(e_native - e_dense) < 1e-9, (N, t, V, mu, e_native, e_dense)


def test_native_rayleigh_matches_stage1_energy_path():
    # Also matches MPS.energy_with_MPO (the Stage-1 dense-path convenience).
    for N in [3, 4, 5]:
        mps, mpo = _mps_mpo(N, 6, 1.0, 0.5, 0.3)
        e_native = float(K.rayleigh_energy_native(mps, mpo))
        e_classic = float(mps.energy_with_MPO(mpo))
        assert abs(e_native - e_classic) < 1e-9


def test_native_energy_backward_grads_not_none():
    mps, mpo = _mps_mpo(4, 6, 1.0, 0.5, 0.3)
    e = K.rayleigh_energy_native(mps, mpo)
    e.backward()
    assert all(p.grad is not None for p in mps.tensors)
    assert all(tc.isfinite(p.grad).all() for p in mps.tensors)


def test_native_energy_is_differentiable_scalar():
    mps, mpo = _mps_mpo(4, 6, 1.0, 0.5, 0.3)
    e = K.rayleigh_energy_native(mps, mpo)
    assert e.requires_grad
    assert e.grad_fn is not None


def test_native_energy_invariant_under_mps_scaling():
    # Rayleigh quotient is scale-invariant.
    mps, mpo = _mps_mpo(4, 6, 1.0, 0.5, 0.3, seed=11)
    e0 = float(K.rayleigh_energy_native(mps, mpo))
    scaled = MPS(4, 2, 6, dtype=DTYPE)
    scaled.tensors = [t.clone() * 2.3 for t in mps.tensors]
    e1 = float(K.rayleigh_energy_native(scaled, mpo))
    assert abs(e0 - e1) < 1e-9, (e0, e1)
