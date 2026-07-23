"""Stage 4: energy_with_MPO must be the Rayleigh quotient <psi|H|psi>/<psi|psi>,
be differentiable, match the dense energy, and respect <psi|psi> normalization.
"""

from __future__ import annotations

import numpy as np
import torch as tc

from latticetn.mpo import MPO
from latticetn.mps import MPS
from latticetn.operators import heisenberg_dense, tfi_dense

DTYPE = tc.complex128


def _dense_energy(psi, H):
    num = psi.conj() @ H @ psi
    den = psi.conj() @ psi
    return (num / den).real


def test_energy_is_rayleigh_quotient_matches_dense():
    tc.manual_seed(7)
    for N in [2, 3, 4, 6]:
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
        H = heisenberg_dense(N)
        mps = MPS(N, 2, 8, dtype=DTYPE)
        e_mpo = float(mps.energy_with_MPO(mpo).detach())
        e_dense = float(_dense_energy(mps.to_dense(), H))
        assert abs(e_mpo - e_dense) < 1e-9, (N, e_mpo, e_dense)


def test_energy_invariant_under_mps_normalization():
    # Energy is a Rayleigh quotient -> invariant to scaling psi by a constant,
    # even when the scaling differs per tensor isn't a scalar; test global scale.
    tc.manual_seed(3)
    N = 4
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
    mps = MPS(N, 2, 6, dtype=DTYPE)
    e0 = float(mps.energy_with_MPO(mpo))
    scaled = MPS(N, 2, 6, dtype=DTYPE)
    scaled.tensors = [t.clone() * 3.7 for t in mps.tensors]
    e1 = float(scaled.energy_with_MPO(mpo))
    assert abs(e0 - e1) < 1e-9, (e0, e1)


def test_energy_autograd_grad_exists_and_finite():
    tc.manual_seed(1)
    N = 4
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
    mps = MPS(N, 2, 6, dtype=DTYPE)
    e = mps.energy_with_MPO(mpo)
    grads = tc.autograd.grad(e, mps.tensors, allow_unused=False)
    for g in grads:
        assert g is not None
        assert tc.isfinite(g).all()
        assert float(g.abs().sum()) > 0


def test_energy_does_not_break_graph_for_backprop():
    # Ensure no .item/.detach/.data inside the differentiable path: a backward()
    # through the energy must populate .grad on the tensors.
    tc.manual_seed(4)
    N = 4
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
    mps = MPS(N, 2, 6, dtype=DTYPE)
    e = mps.energy_with_MPO(mpo)
    e.backward()
    for t in mps.tensors:
        assert t.grad is not None
        assert float(t.grad.abs().sum()) > 0


def test_energy_tfi_matches_dense():
    tc.manual_seed(5)
    N = 4
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_tfi(J=1.0, h=0.5)
    H = tfi_dense(N, J=1.0, h=0.5)
    mps = MPS(N, 2, 8, dtype=DTYPE)
    e_mpo = float(mps.energy_with_MPO(mpo))
    e_dense = float(_dense_energy(mps.to_dense(), H))
    assert abs(e_mpo - e_dense) < 1e-9, (e_mpo, e_dense)
