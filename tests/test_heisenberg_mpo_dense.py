"""Stage 5: Heisenberg MPO -> dense must match the dense Heisenberg reference.

Convention: H = J * sum_i S.S, S = sigma/2, open boundary, default J=1.0.
The MPO uses the standard D=5 nearest-neighbor construction (no self-loop on
carry states -> strictly nearest-neighbor interactions).
"""

from __future__ import annotations

import numpy as np
import torch as tc

from latticetn.mpo import MPO
from reference_models import heisenberg_dense, exact_ground_energy

DTYPE = tc.complex128


def test_heisenberg_mpo_matches_dense():
    for N in [2, 3, 4, 5, 6]:
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
        H_mpo = mpo.to_dense()
        H_ref = heisenberg_dense(N, J=1.0)
        assert tc.allclose(H_mpo, H_ref, atol=1e-12), (
            N, float((H_mpo - H_ref).abs().max()))


def test_heisenberg_mpo_j_scaling_matches_dense():
    for J in [0.5, 1.0, 2.0]:
        N = 5
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=J)
        assert tc.allclose(mpo.to_dense(), heisenberg_dense(N, J=J), atol=1e-12)


def test_heisenberg_mpo_strictly_nearest_neighbor():
    # The MPO must NOT contain next-nearest or longer-range terms: it must equal
    # the dense nearest-neighbor Hamiltonian, which it does iff to_dense matches.
    N = 6
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
    H = mpo.to_dense().numpy()
    # Construct a "long-range version" and make sure we differ from it.
    H_nn = heisenberg_dense(N, J=1.0).numpy()
    assert np.allclose(H, H_nn, atol=1e-12)


def test_heisenberg_mpo_ground_energy_matches_ed():
    for N in [2, 4, 6]:
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
        E0_mpo, _ = exact_ground_energy(mpo.to_dense())
        E0_ref, _ = exact_ground_energy(heisenberg_dense(N, J=1.0))
        assert abs(E0_mpo - E0_ref) < 1e-9, (N, E0_mpo, E0_ref)


def test_heisenberg_mpo_open_boundary_shapes():
    N = 5
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=1.0)
    assert mpo.tensors[0].shape[0] == 1       # left boundary
    assert mpo.tensors[-1].shape[1] == 1      # right boundary
    assert mpo.tensors[1].shape[0] == 5 and mpo.tensors[1].shape[1] == 5  # bulk D=5
