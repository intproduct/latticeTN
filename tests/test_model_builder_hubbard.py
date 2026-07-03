"""Stage 7C: model_builder Hubbard preset alignment.

The Hubbard preset built via ``model_builder`` must produce a dense
Hamiltonian byte-identical to the existing ``operators.hubbard_dense``
reference (Stage 7C, Jordan-Wigner, open boundary, complex128). The fermionic
terms MUST keep the JW parity (they do NOT degrade to spin / hard-core-boson
terms).
"""

from __future__ import annotations

import torch as tc

from latticetn.model_builder import (
    hubbard_model, build_dense, build_mpo,
    FermionHopTerm, DensityDensityTerm, OnsiteTerm,
)
from latticetn.operators import hubbard_dense, exact_ground_energy
from latticetn.mps import MPS
from latticetn import contractions as K

DTYPE = tc.complex128

CASES = [
    (1.0, 4.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.7, 2.0, 0.3, 0.1),
    (1.0, 4.0, 0.2, -0.3),
    (1.0, 8.0, 0.0, 0.0),
    (0.5, 1.0, -0.2, 0.4),
    (1.0, 2.0, -0.7, 0.0),
]


def test_hubbard_model_dense_matches_reference():
    # Dense alignment for N=2..4 (per Stage 7C spec); N=5,6 dense builds are
    # expensive at d=4 (4^N matrix with 2N-mode Kronecker) and are covered by
    # the lighter ground-energy test below.
    for N in [2, 3, 4]:
        for (t, U, mu, h) in CASES:
            spec = hubbard_model(N, t=t, U=U, mu=mu, h=h)
            H = build_dense(spec)
            H_ref = hubbard_dense(N, t=t, U=U, mu=mu, h=h, dtype=DTYPE)
            assert tc.allclose(H, H_ref, atol=1e-12), (N, t, U, mu, h)


def test_hubbard_model_statistics_is_fermion():
    spec = hubbard_model(4, t=1.0, U=4.0, mu=0.0, h=0.0)
    assert spec.statistics == "fermion"
    assert spec.dim == 4
    assert spec.N == 4


def test_hubbard_model_terms_are_fermionic():
    """The preset must contain FermionHopTerm(s) and DensityDensityTerm and
    OnsiteTerm (the hop terms carry the JW string, not a hard-core-boson
    TwoSiteTerm)."""
    spec = hubbard_model(4, t=1.0, U=4.0, mu=0.3, h=0.1)
    has_hop = any(isinstance(t, FermionHopTerm) for t in spec.terms)
    has_dd = any(isinstance(t, DensityDensityTerm) for t in spec.terms)
    has_onsite = any(isinstance(t, OnsiteTerm) for t in spec.terms)
    assert has_hop and has_dd and has_onsite


def test_hubbard_model_ground_energy_matches_ed():
    # N=2,4 (d=4 dense + ED is expensive at N=6; the MPO/dense alignment
    # tests already cover the operator layer up to N=4).
    for N in [2, 4]:
        for (t, U, mu, h) in [(1.0, 4.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0),
                              (1.0, 0.5, 0.3, 0.1)]:
            spec = hubbard_model(N, t=t, U=U, mu=mu, h=h)
            E0, _ = exact_ground_energy(build_dense(spec))
            E0_ref, _ = exact_ground_energy(
                hubbard_dense(N, t=t, U=U, mu=mu, h=h, dtype=DTYPE))
            assert abs(E0 - E0_ref) < 1e-9, (N, t, U, mu, h, E0, E0_ref)


def test_hubbard_model_mpo_dense_matches_model_dense():
    # MPO.to_dense == build_dense for N=2..4 (per Stage 7C spec).
    for N in [2, 3, 4]:
        for (t, U, mu, h) in CASES:
            spec = hubbard_model(N, t=t, U=U, mu=mu, h=h)
            H_mpo = build_mpo(spec).to_dense()
            H_dense = build_dense(spec)
            assert tc.allclose(H_mpo, H_dense, atol=1e-12), (N, t, U, mu, h)


def test_hubbard_model_native_rayleigh_matches_dense_energy():
    # N=2,4 (d=4 dense + MPS.to_dense is expensive at N=6).
    for N in [2, 4]:
        for (t, U, mu, h) in [(1.0, 4.0, 0.0, 0.0), (1.0, 0.5, 0.3, 0.1)]:
            spec = hubbard_model(N, t=t, U=U, mu=mu, h=h)
            mpo = build_mpo(spec)
            tc.manual_seed(0)
            mps = MPS(N, 4, 8, dtype=DTYPE)
            e_native = float(K.rayleigh_energy_native(mps, mpo))
            psi = mps.to_dense()
            H = build_dense(spec)
            e_dense = float(((psi.conj() @ H @ psi) / (psi.conj() @ psi)).real)
            assert abs(e_native - e_dense) < 1e-9, (N, t, U, mu, h,
                                                     e_native, e_dense)


def test_hubbard_model_not_hardcore_boson():
    """For all N>=2 the Hubbard dense H differs from a no-parity spin /
    hard-core-boson hop build (the JW parity structure is real)."""
    from latticetn.fermion_operators import hubbard_local_operators
    from latticetn.operators import _kron
    hop = hubbard_local_operators(dtype=DTYPE)
    I4 = hop["I"]

    def hardcore_boson(N, t):
        H = tc.zeros((4 ** N, 4 ** N), dtype=DTYPE)
        for s in ("up", "down"):
            cdag = hop[f"cdag{s}"]
            c = hop[f"c{s}"]
            for i in range(N - 1):
                term = None
                for k in range(N):
                    g = cdag if k == i else (c if k == i + 1 else I4)
                    term = g if term is None else _kron(term, g)
                hc = None
                for k in range(N):
                    g = c if k == i else (cdag if k == i + 1 else I4)
                    hc = g if hc is None else _kron(hc, g)
                H = H + (-t) * (term + hc)
        return H

    for N in [2, 3, 4]:
        spec = hubbard_model(N, t=1.0, U=0.0, mu=0.0, h=0.0)
        Hf = build_dense(spec)
        Hb = hardcore_boson(N, 1.0)
        assert not tc.allclose(Hf, Hb, atol=1e-9), (
            N, float((Hf - Hb).abs().max()))
