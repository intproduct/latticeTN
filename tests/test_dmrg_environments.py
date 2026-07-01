"""Stage 4A DMRG environment tests.

Verifies the MPO left/right environments: the three-leg environment tensors
have the expected shapes, and combined with the two-site W tensors they
reproduce the full native MPO numerator on a two-site mixed-canonical MPS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn import dmrg as D  # noqa: E402
from latticetn import contractions as K  # noqa: E402

DTYPE = tc.complex128


def _setup(N=5, chi=4, seed=0):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    return mps, mpo


def test_left_env_shape_is_three_leg():
    mps, mpo = _setup()
    t = D.mixed_canonical_two_site(mps, 1)
    tmp = MPS.from_tensors(t, dtype=DTYPE, device="cpu")
    L = D.mpo_left_env(tmp, mpo, up_to=1)
    assert L.dim() == 3
    # bra/ket leg = left bond of site 1; mpo leg = 5 for Heisenberg bulk
    lb = t[1].shape[0]
    assert L.shape[0] == lb and L.shape[2] == lb
    assert L.shape[1] == mpo.tensors[1].shape[0]


def test_right_env_shape_is_three_leg():
    mps, mpo = _setup()
    t = D.mixed_canonical_two_site(mps, 1)
    tmp = MPS.from_tensors(t, dtype=DTYPE, device="cpu")
    R = D.mpo_right_env(tmp, mpo, from_=3)
    assert R.dim() == 3
    rb = t[2].shape[2]
    assert R.shape[0] == rb and R.shape[2] == rb


def test_environments_recover_full_mpo_numerator():
    # <psi|H|psi> via L . W_i . W_{i+1} . R applied to Theta must equal the
    # full native MPO numerator on the (mixed-canonical) MPS.
    mps, mpo = _setup(N=4, seed=1)
    i = 1
    t = D.mixed_canonical_two_site(mps, i)
    tmp = MPS.from_tensors(t, dtype=DTYPE, device="cpu")
    L = D._left_mpo_env(t, mpo.tensors, up_to=i)
    R = D._right_mpo_env(t, mpo.tensors, from_=i + 2)
    Theta = D._theta_two_site(t, i)              # (l, s, s1, r)
    # apply H_eff once to Theta
    out = D.apply_heff(L, mpo.tensors[i], mpo.tensors[i + 1], R, Theta)
    num_local = complex(tc.einsum("lser,lser->", Theta.conj(), out))
    num_full = complex(K.native_mpo_numerator(tmp, mpo))
    assert abs(num_local - num_full) / max(1.0, abs(num_full)) < 1e-10
