"""Stage 7B: model_builder spinless fermion t-V preset alignment.

The spinless fermion t-V preset built via ``model_builder`` must produce a
dense Hamiltonian byte-identical to the existing
``operators.spinless_fermion_dense`` reference (Stage 7A, Jordan-Wigner, open
boundary, complex128). The fermionic terms MUST keep the JW parity string
(they do NOT degrade to hard-core-boson terms).
"""

from __future__ import annotations

import torch as tc

from latticetn.model_builder import (
    spinless_fermion_tv_model, build_dense, build_mpo,
    FermionHopTerm, DensityDensityTerm, OnsiteTerm,
)
from latticetn.operators import spinless_fermion_dense, exact_ground_energy
from latticetn.mps import MPS
from latticetn import contractions as K

DTYPE = tc.complex128

CASES = [
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.5, 0.3),
    (0.7, 1.2, -0.4),
    (1.0, -0.5, 0.0),
    (1.0, 2.0, -0.7),
]


def test_fermion_model_dense_matches_reference():
    for N in [2, 3, 4, 5, 6]:
        for (t, V, mu) in CASES:
            spec = spinless_fermion_tv_model(N, t=t, V=V, mu=mu)
            H = build_dense(spec)
            H_ref = spinless_fermion_dense(N, t=t, V=V, mu=mu, dtype=DTYPE)
            assert tc.allclose(H, H_ref, atol=1e-12), (N, t, V, mu)


def test_fermion_model_statistics_is_fermion():
    spec = spinless_fermion_tv_model(4, t=1.0, V=0.5, mu=0.0)
    assert spec.statistics == "fermion"
    assert spec.dim == 2
    assert spec.N == 4


def test_fermion_model_terms_are_fermionic():
    """The preset must contain a FermionHopTerm (JW), a DensityDensityTerm,
    and an OnsiteTerm — and the hop term carries the JW string (not a
    hard-core-boson TwoSiteTerm)."""
    spec = spinless_fermion_tv_model(4, t=1.0, V=0.5, mu=0.3)
    has_hop = any(isinstance(t, FermionHopTerm) for t in spec.terms)
    has_dd = any(isinstance(t, DensityDensityTerm) for t in spec.terms)
    has_onsite = any(isinstance(t, OnsiteTerm) for t in spec.terms)
    assert has_hop and has_dd and has_onsite


def test_fermion_model_ground_energy_matches_ed():
    for N in [2, 4, 6]:
        for (t, V, mu) in [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.3)]:
            spec = spinless_fermion_tv_model(N, t=t, V=V, mu=mu)
            E0, _ = exact_ground_energy(build_dense(spec))
            E0_ref, _ = exact_ground_energy(
                spinless_fermion_dense(N, t=t, V=V, mu=mu, dtype=DTYPE))
            assert abs(E0 - E0_ref) < 1e-9, (N, t, V, mu, E0, E0_ref)


def test_fermion_model_not_hardcore_boson():
    """For N>=3 the fermion dense H differs from a hard-core-boson build
    (the JW parity string is real). The preset's dense must NOT match a
    no-parity bosonic hop build for N>=3."""
    from latticetn.fermion_operators import fermion_operators
    from latticetn.operators import _kron
    ops = fermion_operators(dtype=DTYPE)
    I, c, cdag = ops["I"], ops["c"], ops["cdag"]

    def hardcore_boson(N, t):
        H = tc.zeros((2 ** N, 2 ** N), dtype=DTYPE)
        for i in range(N - 1):
            term = None
            for k in range(N):
                g = cdag if k == i else (c if k == i + 1 else I)
                term = g if term is None else _kron(term, g)
            H = H + (-t) * (term + term.conj().T)
        return H

    for N in [2, 3, 4, 5]:
        spec = spinless_fermion_tv_model(N, t=1.0, V=0.0, mu=0.0)
        Hf = build_dense(spec)
        Hb = hardcore_boson(N, 1.0)
        if N == 2:
            assert tc.allclose(Hf, Hb, atol=1e-12), N
        else:
            assert not tc.allclose(Hf, Hb, atol=1e-9), N


def test_fermion_model_native_rayleigh_matches_dense_energy():
    for N in [2, 4, 6]:
        for (t, V, mu) in [(1.0, 0.0, 0.0), (1.0, 0.5, 0.3)]:
            spec = spinless_fermion_tv_model(N, t=t, V=V, mu=mu)
            mpo = build_mpo(spec)
            tc.manual_seed(0)
            mps = MPS(N, 2, 8, dtype=DTYPE)
            e_native = float(K.rayleigh_energy_native(mps, mpo))
            psi = mps.to_dense()
            H = build_dense(spec)
            e_dense = float(((psi.conj() @ H @ psi) / (psi.conj() @ psi)).real)
            assert abs(e_native - e_dense) < 1e-9, (N, t, V, mu,
                                                     e_native, e_dense)
