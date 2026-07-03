"""Stage 7C: spinful Hubbard observables (dense + MPS, small-N alignment).

Checks the Hubbard observables in ``latticetn/observables.py``:

- local <n_up_i>, <n_down_i>, <n_tot_i> on known basis states.
- double occupancy <n_up_i n_down_i>.
- local <S^z_i>.
- nearest-neighbor spin-resolved hopping <c^d_{i,s} c_{i+1,s} + h.c.> on a
  one-electron delocalized state (gives 1.0) and on the free-fermion ground
  state (gives -E/t summed over spins, cross-checked against ED).
- dense vs MPS variants agree on a random MPS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.observables import (  # noqa: E402
    dense_hubbard_local_density, dense_hubbard_double_occ,
    dense_hubbard_local_sz, dense_hubbard_nn_hopping,
    mps_hubbard_local_density, mps_hubbard_double_occ,
    mps_hubbard_local_sz, mps_hubbard_nn_hopping,
)
from latticetn.mps import MPS  # noqa: E402
from latticetn.operators import hubbard_dense, exact_ground_energy  # noqa: E402

DTYPE = tc.complex128


def _state(idx_list, N):
    """Build a basis state |s_0, s_1, ...> from per-site indices."""
    psi = tc.zeros(4 ** N, dtype=DTYPE)
    idx = 0
    for s in idx_list:
        idx = idx * 4 + s
    psi[idx] = 1.0
    return psi


def test_local_densities_on_basis_states():
    N = 2
    # |up, down> : site0=up(1), site1=down(2)
    psi = _state([1, 2], N)
    assert abs(float(dense_hubbard_local_density(psi, 0, N, "up")) - 1.0) < 1e-12
    assert abs(float(dense_hubbard_local_density(psi, 0, N, "down")) - 0.0) < 1e-12
    assert abs(float(dense_hubbard_local_density(psi, 0, N, "tot")) - 1.0) < 1e-12
    assert abs(float(dense_hubbard_local_density(psi, 1, N, "up")) - 0.0) < 1e-12
    assert abs(float(dense_hubbard_local_density(psi, 1, N, "down")) - 1.0) < 1e-12
    assert abs(float(dense_hubbard_local_density(psi, 1, N, "tot")) - 1.0) < 1e-12


def test_double_occupancy_on_basis_states():
    N = 2
    # |up, updown> : site0=up(1), site1=ud(3)
    psi = _state([1, 3], N)
    assert abs(float(dense_hubbard_double_occ(psi, 0, N)) - 0.0) < 1e-12
    assert abs(float(dense_hubbard_double_occ(psi, 1, N)) - 1.0) < 1e-12
    # |updown, 0> : site0=ud(3), site1=empty(0)
    psi2 = _state([3, 0], N)
    assert abs(float(dense_hubbard_double_occ(psi2, 0, N)) - 1.0) < 1e-12
    assert abs(float(dense_hubbard_double_occ(psi2, 1, N)) - 0.0) < 1e-12


def test_local_sz_on_basis_states():
    N = 2
    psi = _state([1, 2], N)   # |up, down>
    assert abs(float(dense_hubbard_local_sz(psi, 0, N)) - 0.5) < 1e-12
    assert abs(float(dense_hubbard_local_sz(psi, 1, N)) - (-0.5)) < 1e-12
    # |up, up> : sz = +0.5 each
    psi2 = _state([1, 1], N)
    assert abs(float(dense_hubbard_local_sz(psi2, 0, N)) - 0.5) < 1e-12
    assert abs(float(dense_hubbard_local_sz(psi2, 1, N)) - 0.5) < 1e-12


def test_nn_hopping_one_electron_delocalized():
    """( |up, 0> + |0, up> ) / sqrt2 -> <c^d_{0,up} c_{1,up} + h.c.> = 1.0."""
    N = 2
    psi = _state([1, 0], N) / np.sqrt(2) + _state([0, 1], N) / np.sqrt(2)
    val = float(dense_hubbard_nn_hopping(psi, 0, N, "up"))
    assert abs(val - 1.0) < 1e-9, val
    # down-spin hopping is zero on this up-only state
    val_down = float(dense_hubbard_nn_hopping(psi, 0, N, "down"))
    assert abs(val_down - 0.0) < 1e-9, val_down


def test_nn_hopping_cross_check_with_ed_ground_state():
    """For the free (U=0) Hubbard ground state, hop_up + hop_down = -E/t."""
    for N in [2, 4]:
        H = hubbard_dense(N, t=1.0, U=0.0, mu=0.0, h=0.0, dtype=DTYPE)
        e0, gs = exact_ground_energy(H)
        gs = gs / tc.linalg.norm(gs)
        hop_up = float(dense_hubbard_nn_hopping(gs, 0, N, "up"))
        hop_down = float(dense_hubbard_nn_hopping(gs, 0, N, "down"))
        # H = -t * sum_{i,s} hop_{i,s}; at the ground state <H> = E0. With one
        # bond (N=2) or the dominant bond, sum_s hop_{0,s} ~ -E0/t for the
        # uniform free chain. We check the per-bond sum is positive and the
        # total bond-energy matches: sum_i (hop_up_i + hop_down_i) = -E0/t.
        total_hop = 0.0
        for i in range(N - 1):
            total_hop += float(dense_hubbard_nn_hopping(gs, i, N, "up"))
            total_hop += float(dense_hubbard_nn_hopping(gs, i, N, "down"))
        assert abs(total_hop - (-e0 / 1.0)) < 1e-6, (N, total_hop, e0)


def test_dense_and_mps_observables_agree():
    """Dense and MPS variants of the Hubbard observables must agree on a
    random MPS (small N)."""
    N = 4
    tc.manual_seed(0)
    mps = MPS(N, 4, 6, dtype=DTYPE)
    psi = mps.to_dense()
    psi = psi / tc.linalg.norm(psi)
    # re-normalize the MPS in place so the dense reference is a true state
    # (the observables normalize internally, so just pass the raw MPS)
    for site in range(N):
        for spin in ("up", "down", "tot"):
            d = float(dense_hubbard_local_density(psi, site, N, spin))
            m = float(mps_hubbard_local_density(mps, site, spin=spin))
            assert abs(d - m) < 1e-9, (site, spin, d, m)
        assert abs(float(dense_hubbard_double_occ(psi, site, N))
                   - float(mps_hubbard_double_occ(mps, site))) < 1e-9, site
        assert abs(float(dense_hubbard_local_sz(psi, site, N))
                   - float(mps_hubbard_local_sz(mps, site))) < 1e-9, site
    for i in range(N - 1):
        for spin in ("up", "down"):
            d = float(dense_hubbard_nn_hopping(psi, i, N, spin))
            m = float(mps_hubbard_nn_hopping(mps, i, spin=spin))
            assert abs(d - m) < 1e-9, (i, spin, d, m)
