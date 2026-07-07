"""Stage 7B: model_builder MPO -> dense alignment for both presets.

``build_mpo(spec).to_dense()`` must match ``build_dense(spec)`` for the
Heisenberg and spinless fermion t-V presets across N and parameter sets. The
spinless Hamiltonian uses the JW-reduced adjacent hopping product; nonlocal
fermionic observables still require explicit strings.
"""

from __future__ import annotations

import torch as tc

from latticetn.model_builder import (
    heisenberg_model, spinless_fermion_tv_model, build_dense, build_mpo,
)

DTYPE = tc.complex128


def test_heisenberg_mpo_dense_matches_model_dense():
    for N in [2, 3, 4, 5, 6]:
        spec = heisenberg_model(N, J=1.0)
        H_mpo = build_mpo(spec).to_dense()
        H_dense = build_dense(spec)
        assert tc.allclose(H_mpo, H_dense, atol=1e-12), N


def test_heisenberg_mpo_j_scaling():
    for J in [0.5, 1.0, 2.0]:
        spec = heisenberg_model(5, J=J)
        assert tc.allclose(build_mpo(spec).to_dense(), build_dense(spec),
                           atol=1e-12)


def test_fermion_mpo_dense_matches_model_dense():
    cases = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.3),
             (0.7, 1.2, -0.4), (1.0, -0.5, 0.0), (1.0, 2.0, -0.7)]
    for N in [2, 3, 4, 5, 6]:
        for (t, V, mu) in cases:
            spec = spinless_fermion_tv_model(N, t=t, V=V, mu=mu)
            H_mpo = build_mpo(spec).to_dense()
            H_dense = build_dense(spec)
            assert tc.allclose(H_mpo, H_dense, atol=1e-12), (N, t, V, mu)


def test_mpo_open_boundary_shapes():
    # Heisenberg MPO bond dim 5, spinless t-V MPO bond dim 5.
    hspec = heisenberg_model(5, J=1.0)
    hmpo = build_mpo(hspec)
    assert hmpo.tensors[0].shape[0] == 1
    assert hmpo.tensors[-1].shape[1] == 1
    assert hmpo.tensors[1].shape[0] == 5

    fspec = spinless_fermion_tv_model(5, t=1.0, V=1.0, mu=0.0)
    fmpo = build_mpo(fspec)
    assert fmpo.tensors[0].shape[0] == 1
    assert fmpo.tensors[-1].shape[1] == 1
    assert fmpo.tensors[1].shape[0] == 5


def test_mpo_dense_is_hermitian():
    for spec in [heisenberg_model(5, J=1.0),
                 spinless_fermion_tv_model(5, t=1.0, V=0.5, mu=0.3)]:
        H = build_mpo(spec).to_dense()
        assert tc.allclose(H, H.conj().T, atol=1e-12), spec.name


def test_build_mpo_unregistered_preset_raises():
    from latticetn.model_builder import ModelSpec
    spec = ModelSpec(N=4, dim=2, statistics="boson", name="unknown_model")
    try:
        build_mpo(spec)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for unregistered preset")


def test_build_dense_unregistered_preset_raises():
    from latticetn.model_builder import ModelSpec
    spec = ModelSpec(N=4, dim=2, statistics="boson", name="unknown_model")
    try:
        build_dense(spec)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for unregistered preset")
