"""Stage 3B native observable contraction tests.

Verifies native local/two-site/bond-energy/correlation contractions against the
dense-state references (Stage 2 observables) for small random MPS, including
non-commuting operator ordering (i>j).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn import contractions as K  # noqa: E402
from latticetn import observables as O  # noqa: E402
from latticetn.operators import spin_operators  # noqa: E402

DTYPE = tc.complex128


def _make(seed=0):
    tc.manual_seed(seed)
    return MPS(6, 2, 4, dtype=DTYPE)


def _dense(mps: MPS) -> tc.Tensor:
    return mps.to_dense().detach()


def test_native_local_sz_matches_dense():
    mps = _make(0)
    psi = _dense(mps)
    ops = spin_operators()
    for site in range(mps.N):
        d = float(O.dense_expect_local(psi, ops["Sz"], site, mps.N).real)
        n = float(K.native_local_expect(mps, ops["Sz"], site).real)
        assert abs(d - n) < 1e-9


def test_native_two_site_sz_matches_dense_including_ij_order():
    mps = _make(1)
    psi = _dense(mps)
    ops = spin_operators()
    for i, j in [(0, 1), (1, 3), (4, 2), (3, 5), (5, 0)]:
        d = complex(O.dense_expect_two_site(psi, ops["Sz"], i, ops["Sz"], j, mps.N)).real
        n = float(K.native_two_site_expect(mps, ops["Sz"], i, ops["Sz"], j).real)
        assert abs(d - n) < 1e-9, (i, j, d, n)


def test_native_non_commuting_operator_order_preserved():
    # <S+_i S-_j> must respect caller ordering (S+ and S- do not commute).
    mps = _make(2)
    psi = _dense(mps)
    ops = spin_operators()
    for i, j in [(1, 2), (2, 1)]:
        d = complex(O.dense_expect_two_site(psi, ops["S+"], i, ops["S-"], j, mps.N))
        n = complex(K.native_two_site_expect(mps, ops["S+"], i, ops["S-"], j))
        assert abs(d - n) < 1e-9, (i, j, d, n)


def test_native_bond_energy_matches_dense():
    mps = _make(3)
    psi = _dense(mps)
    for i in range(mps.N - 1):
        d = float(O.dense_bond_energy_heisenberg(psi, i, mps.N))
        n = float(K.native_bond_energy_heisenberg(mps, i))
        assert abs(d - n) < 1e-9, (i, d, n)


def test_native_correlation_matches_dense():
    mps = _make(4)
    psi = _dense(mps)
    ops = spin_operators()
    for i, j in [(0, 2), (1, 4), (0, 5)]:
        d = complex(O.dense_expect_two_site(psi, ops["Sz"], i, ops["Sz"], j, mps.N)).real
        n = float(K.native_correlation(mps, ops["Sz"], i, j).real)
        assert abs(d - n) < 1e-9, (i, j, d, n)
