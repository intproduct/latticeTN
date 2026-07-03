"""Stage 7B: model_builder Heisenberg preset alignment.

The Heisenberg preset built via ``model_builder`` must produce a dense
Hamiltonian byte-identical to the existing ``operators.heisenberg_dense``
reference (Stage 1 convention, S = sigma/2, J, open boundary, complex128).
"""

from __future__ import annotations

import torch as tc

from latticetn.model_builder import heisenberg_model, build_dense, build_mpo
from latticetn.operators import heisenberg_dense, exact_ground_energy
from latticetn.mps import MPS
from latticetn import contractions as K

DTYPE = tc.complex128


def test_heisenberg_model_dense_matches_reference():
    for N in [2, 3, 4, 5, 6]:
        spec = heisenberg_model(N, J=1.0)
        H = build_dense(spec)
        H_ref = heisenberg_dense(N, J=1.0)
        assert tc.allclose(H, H_ref, atol=1e-12), N


def test_heisenberg_model_j_scaling():
    for J in [0.5, 1.0, 2.0]:
        spec = heisenberg_model(5, J=J)
        assert tc.allclose(build_dense(spec), heisenberg_dense(5, J=J), atol=1e-12)


def test_heisenberg_model_statistics_is_boson():
    spec = heisenberg_model(4, J=1.0)
    assert spec.statistics == "boson"
    assert spec.dim == 2
    assert spec.N == 4


def test_heisenberg_model_ground_energy_matches_ed():
    for N in [2, 4, 6]:
        spec = heisenberg_model(N, J=1.0)
        E0, _ = exact_ground_energy(build_dense(spec))
        E0_ref, _ = exact_ground_energy(heisenberg_dense(N, J=1.0))
        assert abs(E0 - E0_ref) < 1e-9, (N, E0, E0_ref)


def test_heisenberg_model_terms_decompose_to_SdotS():
    """The Heisenberg preset's terms must reconstruct S_i . S_{i+1}.

    S.S = Sz Sz + (1/2)(S+ S- + S- S+); the preset stores exactly these three
    two-site terms with coeff J.
    """
    from latticetn.model_builder import TwoSiteTerm
    from latticetn.operators import spin_operators
    ops = spin_operators(dtype=DTYPE)
    spec = heisenberg_model(4, J=1.0)
    # collect (op_i, op_j, coeff) per two-site term
    pairs = [(t.op_i, t.op_j, t.coeff) for t in spec.terms
             if isinstance(t, TwoSiteTerm)]
    # there should be 3 terms: SzSz, S+S-, S-S+
    assert len(pairs) == 3
    # rebuild the two-site operator and compare to S.S
    SdotS = ops["Sz"] @ ops["Sz"] + 0.5 * (ops["S+"] @ ops["S-"]
                                            + ops["S-"] @ ops["S+"])
    acc = sum(coeff * (oi @ oj) for oi, oj, coeff in pairs)
    assert tc.allclose(acc, SdotS, atol=1e-12)


def test_heisenberg_native_rayleigh_matches_dense_energy():
    for N in [2, 4, 6]:
        spec = heisenberg_model(N, J=1.0)
        mpo = build_mpo(spec)
        tc.manual_seed(0)
        mps = MPS(N, 2, 8, dtype=DTYPE)
        e_native = float(K.rayleigh_energy_native(mps, mpo))
        psi = mps.to_dense()
        H = build_dense(spec)
        e_dense = float(((psi.conj() @ H @ psi) / (psi.conj() @ psi)).real)
        assert abs(e_native - e_dense) < 1e-9, (N, e_native, e_dense)
