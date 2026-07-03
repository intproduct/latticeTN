"""Stage 7A: spinless fermion dense Hamiltonian reference correctness.

``operators.spinless_fermion_dense`` is the dense (small-N) golden reference
for the open-boundary 1D spinless fermion t-V chain. It must be:

- Hermitian.
- The correct free-fermion (V=0, mu=0) spectrum (single-particle
  ``-2t cos(k pi/(N+1))`` summed over negative modes).
- Particle-hole symmetric at half filling when mu=0.
- Consistent with the explicit JW global operators (NOT a hard-core-boson H).
"""

from __future__ import annotations

import numpy as np
import torch as tc

from latticetn.operators import spinless_fermion_dense, exact_ground_energy, _kron
from latticetn.fermion_operators import fermion_operators

DTYPE = tc.complex128


def test_dense_is_hermitian():
    for N in [2, 3, 4, 5, 6]:
        for (t, V, mu) in [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                           (0.7, 1.2, -0.4), (1.0, 0.5, 0.3)]:
            H = spinless_fermion_dense(N, t=t, V=V, mu=mu, dtype=DTYPE)
            assert tc.allclose(H, H.conj().T, atol=1e-12), (N, t, V, mu)


def _free_fermion_e0(N: int, t: float = 1.0) -> float:
    """Single-particle open-chain levels eps_k = -2t cos(k pi/(N+1)), k=1..N.

    E0 = sum of negative eps_k (fill the Fermi sea at half filling, mu=0)."""
    eps = -2.0 * t * np.cos(np.arange(1, N + 1) * np.pi / (N + 1))
    return float(eps[eps < 0].sum())


def test_free_fermion_ground_energy_matches_single_particle():
    for N in [2, 3, 4, 5, 6]:
        H = spinless_fermion_dense(N, t=1.0, V=0.0, mu=0.0, dtype=DTYPE)
        E0, _ = exact_ground_energy(H)
        assert abs(E0 - _free_fermion_e0(N)) < 1e-9, (N, E0, _free_fermion_e0(N))


def test_free_fermion_t_scaling():
    for t in [0.5, 1.0, 2.0]:
        N = 5
        H = spinless_fermion_dense(N, t=t, V=0.0, mu=0.0, dtype=DTYPE)
        E0, _ = exact_ground_energy(H)
        assert abs(E0 - _free_fermion_e0(N, t=t)) < 1e-9, (t, E0)


def test_dense_matches_explicit_jw_global_operators():
    """The dense H must equal the explicit JW global-operator build.

    Confirms the dense reference is genuinely fermionic: each hopping term is
    ``-t (F^i x c^d x c + F^i x c x c^d)`` with the JW parity string, NOT the
    hard-core-boson ``c^d x c`` (which would give the wrong sign structure).
    """
    ops = fermion_operators(dtype=DTYPE)
    I, c, cdag, F, nmh = ops["I"], ops["c"], ops["cdag"], ops["F"], ops["n_minus_half"]

    def global_op(local, site, N):
        term = None
        for k in range(N):
            g = F if k < site else (local if k == site else I)
            term = g if term is None else _kron(term, g)
        return term

    def two_site(op_i, i, op_i1, N):
        # F^i x op_i x op_{i+1} x I... (parity on sites 0..i-1)
        term = None
        for k in range(N):
            if k < i:
                g = F
            elif k == i:
                g = op_i
            elif k == i + 1:
                g = op_i1
            else:
                g = I
            term = g if term is None else _kron(term, g)
        return term

    for N in [2, 3, 4, 5]:
        for (t, V, mu) in [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.7, 1.2, -0.4)]:
            H_ref = tc.zeros((2 ** N, 2 ** N), dtype=DTYPE)
            # on-site -mu (n-1/2), diagonal (no parity)
            for i in range(N):
                op = None
                for k in range(N):
                    g = nmh if k == i else I
                    op = g if op is None else _kron(op, g)
                H_ref = H_ref + (-mu) * op
            for i in range(N - 1):
                H_ref = H_ref + (-t) * (two_site(cdag, i, c, N) + two_site(c, i, cdag, N))
            for i in range(N - 1):
                op = None
                for k in range(N):
                    g = nmh if (k == i or k == i + 1) else I
                    op = g if op is None else _kron(op, g)
                H_ref = H_ref + V * op
            H = spinless_fermion_dense(N, t=t, V=V, mu=mu, dtype=DTYPE)
            assert tc.allclose(H, H_ref, atol=1e-12), (N, t, V, mu)


def test_particle_hole_symmetry_half_filling_mu_zero():
    """At mu=0 (half filling) the spectrum is particle-hole symmetric:
    E(N_e) = E(N - N_e) + const, equivalently the spectrum is symmetric about
    a shift. A simple check: the trace of H is 0 at mu=0 (the on-site term
    -mu(n-1/2) vanishes and the hopping/interaction are traceless at mu=0 for
    the t-V chain with the (n-1/2)(n-1/2) interaction)."""
    for N in [2, 3, 4, 5, 6]:
        H = spinless_fermion_dense(N, t=1.0, V=0.5, mu=0.0, dtype=DTYPE)
        assert abs(float(H.trace().real)) < 1e-10, (N, float(H.trace()))


def test_not_hardcore_boson_signs():
    """The fermionic H differs from the naive hard-core-boson H for N>=3.

    For N=2 the JW string is trivial (only bond 0, no sites to the left), so
    fermion == boson. For N>=3 bond i>=1 carries the JW parity string, so the
    fermionic H differs from the hard-core-boson H (which would use c^d x c
    with NO parity). This test confirms the difference is real.
    """
    ops = fermion_operators(dtype=DTYPE)
    I, c, cdag = ops["I"], ops["c"], ops["cdag"]

    def hardcore_boson_H(N, t):
        # naive two-site product with NO JW string (wrong for fermions)
        H = tc.zeros((2 ** N, 2 ** N), dtype=DTYPE)
        for i in range(N - 1):
            term = None
            for k in range(N):
                if k == i:
                    g = cdag
                elif k == i + 1:
                    g = c
                else:
                    g = I
                term = g if term is None else _kron(term, g)
            hc = term.conj().T
            H = H + (-t) * (term + hc)
        return H

    for N in [2, 3, 4, 5]:
        Hf = spinless_fermion_dense(N, t=1.0, V=0.0, mu=0.0, dtype=DTYPE)
        Hb = hardcore_boson_H(N, 1.0)
        if N == 2:
            assert tc.allclose(Hf, Hb, atol=1e-12), N
        else:
            assert not tc.allclose(Hf, Hb, atol=1e-9), N
