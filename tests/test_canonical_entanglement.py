"""Stage 3A entanglement entropy tests (canonical vs dense SVD reference)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn import canonical as C  # noqa: E402
from latticetn.observables import dense_entanglement_entropy  # noqa: E402


def test_canonical_entropy_matches_dense_for_random_mps():
    tc.manual_seed(7)
    N = 5
    mps = MPS(N, 2, 4, dtype=tc.complex128)
    psi = mps.to_dense()
    psi = psi / tc.linalg.norm(psi)
    for cut in range(1, N):
        dense_S = float(dense_entanglement_entropy(psi, cut, N))
        canon_S = float(C.entanglement_entropy(mps, cut))
        assert abs(dense_S - canon_S) < 1e-9


def test_canonical_entropy_zero_for_product_state():
    # A product state has zero entanglement across every cut.
    N = 4
    # |0 0 0 0> as an MPS via from_dense
    psi = tc.zeros(2 ** N, dtype=tc.complex128)
    psi[0] = 1.0
    mps = C.from_dense(psi, N, chi=None)
    for cut in range(1, N):
        assert abs(float(C.entanglement_entropy(mps, cut))) < 1e-12


def test_canonical_entropy_max_for_bell_pair_cut():
    # N=2, (|00> + |11>)/sqrt2 -> entropy ln(2) at the only cut.
    psi = tc.zeros(4, dtype=tc.complex128)
    psi[0] = 1.0 / math.sqrt(2.0)
    psi[3] = 1.0 / math.sqrt(2.0)
    mps = C.from_dense(psi, 2, chi=None)
    assert abs(float(C.entanglement_entropy(mps, 1)) - math.log(2.0)) < 1e-9
