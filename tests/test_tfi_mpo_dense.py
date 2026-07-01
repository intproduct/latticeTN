"""Stage 3: TFI MPO -> dense must match the dense TFI reference.

Convention (recorded in docs/CLAUDE_PROGRESS.md): H_TFI = -J Sz Sz - h sum_i Sx,
spin convention S = sigma/2, open boundary.
"""

from __future__ import annotations

import numpy as np
import torch as tc

from latticetn.mpo import MPO
from reference_models import tfi_dense, exact_ground_energy

DTYPE = tc.complex128


def test_tfi_mpo_matches_dense_several_sizes():
    for N in [2, 3, 4, 5]:
        for J, h in [(1.0, 0.5), (1.0, 1.0), (2.0, 0.3)]:
            mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_tfi(J=J, h=h)
            H_mpo = mpo.to_dense()
            H_ref = tfi_dense(N, J=J, h=h)
            assert tc.allclose(H_mpo, H_ref, atol=1e-12), (N, J, h,
                                                           float((H_mpo - H_ref).abs().max()))


def test_tfi_mpo_hermitian():
    for N in [2, 4, 6]:
        mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_tfi(J=1.0, h=0.6)
        H = mpo.to_dense()
        assert tc.allclose(H, H.conj().T, atol=1e-12)


def test_tfi_mpo_ground_energy_at_h0_is_classical():
    # h=0, J>0 -> H = -J Sz Sz, ground = fully aligned, E0 = -J * (N-1) * (1/4).
    N = 5
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_tfi(J=1.0, h=0.0)
    E0, _ = exact_ground_energy(mpo.to_dense())
    assert abs(E0 - (-(N - 1) * 0.25)) < 1e-9, E0


def test_tfi_mpo_open_boundary_shape():
    N = 4
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_tfi(J=1.0, h=0.5)
    # left/right boundary bonds are size 1
    assert mpo.tensors[0].shape[0] == 1
    assert mpo.tensors[-1].shape[1] == 1
    # bulk virtual dimension is 3 for TFI
    assert mpo.tensors[1].shape[0] == 3 and mpo.tensors[1].shape[1] == 3
