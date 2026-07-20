"""Matrix-free TDVP effective-Hamiltonian contraction checks."""

import torch as tc

from latticetn import canonical
from latticetn.dmrg import apply_heff as dmrg_apply_two_site
from latticetn.mpo import MPO
from latticetn.mps import MPS
from latticetn.tdvp import effective_hamiltonian as effective


DTYPE = tc.complex128


def _random_mixed_state(n=5, chi=4, center=2, seed=0):
    generator = tc.Generator().manual_seed(seed)
    mps = MPS(n, 2, chi, dtype=DTYPE, rng=generator)
    return canonical.mixed_canonical(mps, center)


def test_one_site_effective_hamiltonian_is_hermitian():
    center = 2
    mps = _random_mixed_state(center=center, seed=2)
    mpo = MPO.from_bonds(mps.N, 2, dtype=DTYPE).generate_heisenberg()
    left = effective.build_left_environments(mps.tensors, mpo.tensors)[center]
    right = effective.build_right_environments(mps.tensors, mpo.tensors)[center + 1]
    shape = mps.tensors[center].shape
    action = effective.one_site_action(left, mpo.tensors[center], right, shape)
    identity = tc.eye(action.dim, dtype=DTYPE)
    dense = tc.stack([action(identity[:, j]) for j in range(action.dim)], dim=1)
    assert tc.allclose(dense, dense.conj().transpose(0, 1), atol=1e-11, rtol=1e-11)


def test_two_site_action_reuses_verified_dmrg_leg_convention():
    bond = 1
    mps = _random_mixed_state(n=4, center=bond, seed=4)
    mpo = MPO.from_bonds(mps.N, 2, dtype=DTYPE).generate_heisenberg()
    left = effective.build_left_environments(mps.tensors, mpo.tensors)[bond]
    right = effective.build_right_environments(mps.tensors, mpo.tensors)[bond + 2]
    theta = tc.einsum("lsc,cer->lser", mps.tensors[bond], mps.tensors[bond + 1])
    got = effective.apply_two_site(
        left, mpo.tensors[bond], mpo.tensors[bond + 1], right, theta
    )
    expected = dmrg_apply_two_site(
        left, mpo.tensors[bond], mpo.tensors[bond + 1], right, theta
    )
    assert tc.allclose(got, expected, atol=1e-12, rtol=1e-12)
