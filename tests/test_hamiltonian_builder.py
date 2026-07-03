import torch as tc

from latticetn.hamiltonian_builder import build_mpo
from latticetn.model_registry import build_model_spec
from latticetn.mpo import MPO


DTYPE = tc.complex128


def _assert_same_dense(spec, old_mpo):
    new = build_mpo(spec, dtype=DTYPE, device="cpu")
    assert new.length == old_mpo.length
    assert new.dim == old_mpo.dim
    diff = (new.to_dense() - old_mpo.to_dense()).abs().max()
    assert float(diff) < 1e-10


def test_build_mpo_matches_heisenberg_generator():
    spec = build_model_spec("heisenberg", N=4, parameters={"J": 1.2})
    old = MPO.from_bonds(4, 2, dtype=DTYPE).generate_heisenberg(J=1.2)
    _assert_same_dense(spec, old)


def test_build_mpo_matches_tfi_generator():
    spec = build_model_spec("tfi", N=4, parameters={"J": 0.7, "h": 0.3})
    old = MPO.from_bonds(4, 2, dtype=DTYPE).generate_tfi(J=0.7, h=0.3)
    _assert_same_dense(spec, old)


def test_build_mpo_matches_spinless_generator():
    spec = build_model_spec("spinless_tv", N=4, parameters={"t": 1.0, "V": 0.2, "mu": 0.1})
    old = MPO.from_bonds(4, 2, dtype=DTYPE).generate_spinless_fermion(t=1.0, V=0.2, mu=0.1)
    _assert_same_dense(spec, old)


def test_build_mpo_matches_hubbard_generator():
    spec = build_model_spec("hubbard", N=3, parameters={"t": 1.0, "U": 2.0, "mu": 0.1, "h": 0.2})
    old = MPO.from_bonds(3, 4, dtype=DTYPE).generate_hubbard(t=1.0, U=2.0, mu=0.1, h=0.2)
    _assert_same_dense(spec, old)
