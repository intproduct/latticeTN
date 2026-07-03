"""Stage 7A: spinless fermion local operator algebra (Jordan-Wigner).

Checks the canonical fermion algebra on the local 2x2 operators returned by
``fermion_operators``:

    {c, c^d} = I,  c^2 = (c^d)^2 = 0,  n = c^d c,  F = (-1)^n,  F^2 = I,
    F c = - c F  (parity anticommutes with c / c^d on the same site).

These are the building blocks of the Jordan-Wigner construction; the parity
string ``F`` is what makes fermionic operators on different sites anticommute
(see ``latticetn/fermion_operators.py``).
"""

from __future__ import annotations

import torch as tc

from latticetn.fermion_operators import fermion_operators

DTYPE = tc.complex128


def _ops():
    return fermion_operators(dtype=DTYPE, device="cpu")


def test_anticommutator_c_cdag_is_identity():
    ops = _ops()
    c, cdag, I = ops["c"], ops["cdag"], ops["I"]
    # {c, c^d} = c c^d + c^d c
    antic = c @ cdag + cdag @ c
    assert tc.allclose(antic, I, atol=1e-12), antic


def test_c_squared_is_zero():
    ops = _ops()
    c = ops["c"]
    assert tc.allclose(c @ c, tc.zeros_like(c), atol=1e-12)


def test_cdag_squared_is_zero():
    ops = _ops()
    cdag = ops["cdag"]
    assert tc.allclose(cdag @ cdag, tc.zeros_like(cdag), atol=1e-12)


def test_number_operator_is_cdag_c():
    ops = _ops()
    c, cdag, n = ops["c"], ops["cdag"], ops["n"]
    assert tc.allclose(cdag @ c, n, atol=1e-12)
    # n is diag(0, 1)
    assert tc.allclose(n, tc.tensor([[0, 0], [0, 1]], dtype=DTYPE), atol=1e-12)


def test_parity_F_squared_is_identity():
    ops = _ops()
    F, I = ops["F"], ops["I"]
    assert tc.allclose(F @ F, I, atol=1e-12)
    # F = diag(1, -1) = (-1)^n
    assert tc.allclose(F, tc.tensor([[1, 0], [0, -1]], dtype=DTYPE), atol=1e-12)


def test_F_anticommutes_with_c_and_cdag():
    """F c = - c F  and  F c^d = - c^d F  (same-site JW anticommutation)."""
    ops = _ops()
    F, c, cdag = ops["F"], ops["c"], ops["cdag"]
    assert tc.allclose(F @ c, -(c @ F), atol=1e-12)
    assert tc.allclose(F @ cdag, -(cdag @ F), atol=1e-12)


def test_n_minus_half_is_diagonal():
    ops = _ops()
    nmh, I = ops["n_minus_half"], ops["I"]
    expected = tc.tensor([[-0.5, 0], [0, 0.5]], dtype=DTYPE)
    assert tc.allclose(nmh, expected, atol=1e-12)
    # n - 1/2 = n - (1/2) I
    assert tc.allclose(nmh, ops["n"] - 0.5 * I, atol=1e-12)


def test_global_jw_operators_anticommute_on_different_sites():
    """Global c_i and c_j (i != j) must anticommute: {c_i, c_j} = 0.

    Builds the global operators with the explicit JW parity string
    ``c_i = F x ... x F x c x I ...`` and checks the anticommutator vanishes
    for several (i, j) pairs on a small chain. This is the defining property
    that makes the Hamiltonian fermionic, not hard-core-bosonic.
    """
    N = 4
    ops = _ops()
    I, c, F = ops["I"], ops["c"], ops["F"]

    def global_c(site):
        term = None
        for k in range(N):
            g = F if k < site else (c if k == site else I)
            term = g if term is None else tc.kron(term, g)
        return term

    for i in range(N):
        for j in range(i + 1, N):
            ci, cj = global_c(i), global_c(j)
            antic = ci @ cj + cj @ ci
            assert tc.allclose(antic, tc.zeros((2 ** N, 2 ** N), dtype=DTYPE),
                               atol=1e-12), (i, j)


def test_global_c_anticommutes_with_global_cdag_on_different_sites():
    """{c_i, c^d_j} = 0 for i != j (different sites)."""
    N = 4
    ops = _ops()
    I, c, cdag, F = ops["I"], ops["c"], ops["cdag"], ops["F"]

    def global_op(local, site):
        term = None
        for k in range(N):
            g = F if k < site else (local if k == site else I)
            term = g if term is None else tc.kron(term, g)
        return term

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            ci = global_op(c, i)
            cdagj = global_op(cdag, j)
            antic = ci @ cdagj + cdagj @ ci
            assert tc.allclose(antic, tc.zeros((2 ** N, 2 ** N), dtype=DTYPE),
                               atol=1e-12), (i, j)


def test_same_site_anticommutator_c_cdag_is_identity_globally():
    """{c_i, c^d_i} = I (globally, on the same site)."""
    N = 3
    ops = _ops()
    I, c, cdag, F = ops["I"], ops["c"], ops["cdag"], ops["F"]

    def global_op(local, site):
        term = None
        for k in range(N):
            g = F if k < site else (local if k == site else I)
            term = g if term is None else tc.kron(term, g)
        return term

    for i in range(N):
        ci = global_op(c, i)
        cdagi = global_op(cdag, i)
        antic = ci @ cdagi + cdagi @ ci
        assert tc.allclose(antic, tc.eye(2 ** N, dtype=DTYPE), atol=1e-12), i
