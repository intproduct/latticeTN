"""Spinless fermion local operators and Jordan-Wigner conventions.

Stage 7A adds the open-boundary 1D spinless fermion t-V chain

    H = -t * sum_i (c^d_i c_{i+1} + c^d_{i+1} c_i)
        + V * sum_i (n_i - 1/2)(n_{i+1} - 1/2)
        - mu * sum_i (n_i - 1/2)

with local basis |0> (empty), |1> (occupied); d=2; default complex128.

THIS IS NOT A FULL GRADED FERMIONIC TENSOR NETWORK. It is the 1D
Jordan-Wigner (JW) construction: fermionic operators are represented as
tensor products of *bosonic* (spinless, 2x2) local matrices with an explicit
parity string. The crucial object is the JW parity operator

    F = (-1)^n = diag(1, -1)

which anticommutes with c / c^d on the SAME site (F c = -c F) and is the
"string" carried left-to-right so that fermionic operators on different sites
anticommute:

    c_i  = F_0 x F_1 x ... x F_{i-1} x c x I x ... x I
    c^d_i = F_0 x F_1 x ... x F_{i-1} x c^d x I x ... x I

For NEAREST-NEIGHBOR hopping c^d_i c_{i+1} + h.c., the two parity strings
between sites i and i+1 cancel (F c = c on the left factor and F is squared to
I on the right factor), so the two-site hopping operator is the ordinary
bosonic product c^d_i (c)_{i+1} + h.c. with NO explicit parity string. This is
why the t-V chain admits a finite-bond-dimension MPO (see
``MPO.generate_spinless_fermion``). The dense reference
``operators.spinless_fermion_dense`` builds the *global* c_i / c^d_i WITH the
full parity string, so it is a genuine fermionic Hamiltonian, NOT a
hard-core-boson Hamiltonian — the JW string is the only thing making the two
agree at the two-site level.

Conventions are independent of the spin convention S = sigma/2 used elsewhere
in the package; the fermion module never mixes with ``spin_operators``.
"""

from __future__ import annotations

import torch as tc


def fermion_operators(dtype=tc.complex128, device="cpu") -> dict:
    """Return the spinless-fermion local 2x2 operators as a dict of tensors.

    Basis: |0> = (1, 0)^T (empty), |1> = (0, 1)^T (occupied). All matrices are
    real; returned in the requested complex dtype for consistency with the
    rest of the package.

    Returns
    -------
    dict with keys:
        ``I``            : 2x2 identity.
        ``c``            : annihilation operator c  (c |1> = |0>, c |0> = 0).
        ``cdag``         : creation operator c^d (c^d |0> = |1>, c^d |1> = 0).
        ``n``            : number operator n = c^d c  (n |1> = |1>, n |0> = 0).
        ``F``            : Jordan-Wigner parity F = (-1)^n = diag(1, -1).
        ``n_minus_half`` : n - 1/2 = diag(-1/2, +1/2).

    Algebra checks (see tests/test_fermion_operators.py):
        {c, c^d} = I,  c^2 = (c^d)^2 = 0,  n = c^d c,  F^2 = I,  F c = - c F.
    """
    I = tc.eye(2, dtype=dtype, device=device)
    # c |1> = |0>  ->  matrix with a single 1 in the (row0, col1) entry.
    c = tc.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
    # c^d |0> = |1>  ->  single 1 in the (row1, col0) entry.
    cdag = tc.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
    # n = c^d c = diag(0, 1).
    n = cdag @ c
    # F = (-1)^n = diag(1, -1). The Jordan-Wigner parity string.
    F = tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    # n - 1/2 used for the density-density interaction term.
    n_minus_half = n - 0.5 * I
    return {
        "I": I,
        "c": c,
        "cdag": cdag,
        "n": n,
        "F": F,
        "n_minus_half": n_minus_half,
    }


def hubbard_local_operators(dtype=tc.complex128, device="cpu") -> dict:
    """Return the spinful-Hubbard local 4x4 operators as a dict of tensors.

    Stage 7C adds the open-boundary 1D spinful Hubbard chain. Local basis
    (fixed convention):

        index 0 : |0>     (empty)
        index 1 : |up>    (one up electron)
        index 2 : |down>  (one down electron)
        index 3 : |up,down> (double occupancy)

    so ``d = 4``. The local Hilbert space is the tensor product of the up-spin
    mode and the down-spin mode, and the basis above is the tensor-product
    basis ordered as ``(up, down)`` with the up bit the *more significant*
    index. Concretely, the 4 basis states are the columns of
    ``|up> x |down>`` with up the high bit:

        |0>      = |0_up> x |0_down>   -> index 0
        |up>     = |1_up> x |0_down>   -> index 1
        |down>   = |0_up> x |1_down>   -> index 2
        |updown> = |1_up> x |1_down>   -> index 3

    The spin-resolved local operators are built from the spinless 2x2
    operators ``c``, ``c^d``, ``n``, ``F = (-1)^n`` of :func:`fermion_operators`
    via the **on-site Jordan-Wigner** construction, then re-indexed to the
    standard Hubbard basis. Concretely, the two modes on a site are JW-ordered
    ``up`` then ``down`` (so the down operator carries the up-mode parity
    ``F_up``), which makes the *on-site* canonical anticommutation relations
    hold exactly:

        c_up       = c   x I         (up mode, no parity needed — it is first)
        c^d_up     = c^d x I
        c_down     = F   x c         (down mode carries up parity F_up)
        c^d_down   = F   x c^d
        n_up       = n   x I
        n_down     = I   x n
        n_tot      = n_up + n_down
        sz         = (1/2)(n_up - n_down)
        double_occ = n_up * n_down
        parity     = F   x F = (-1)^(n_up + n_down)

    The Kronecker product ``A x B`` uses up as the LEFT (high) factor and down
    as the RIGHT (low) factor, so in the *JW-natural* basis the columns are
    ordered ``(0, down, up, updown)``. We then apply the index permutation
    that swaps the middle two axes so the public basis is the standard Hubbard
    one, ``|0>, |up>, |down>, |updown>`` (indices 0, 1, 2, 3). Under this
    permutation the on-site CARs are preserved (a unitary relabeling of basis
    states), and the number / sz / double-occ / parity operators come out
    diagonal exactly as expected:

        n_up      = diag(0, 1, 0, 1)
        n_down    = diag(0, 0, 1, 1)
        n_tot     = diag(0, 1, 1, 2)
        sz        = diag(0, +1/2, -1/2, 0)
        double_occ= diag(0, 0, 0, 1)
        parity    = diag(1, -1, -1, 1) = (-1)^{n_tot}

    with the standard fermionic actions ``c_up|up> = |0>``,
    ``c_up|updown> = |down>``, ``c_down|down> = |0>``,
    ``c_down|updown> = -|up>`` (the minus sign on the last is the correct JW
    sign, since the down mode is JW-ordered after the up mode on the site).

    The global anticommutation between different *sites* is supplied by the
    Jordan-Wigner parity string on every global mode left of the operator
    (see :func:`latticetn.operators.hubbard_dense`), with the global mode
    ordering fixed to **site-major** ``(0_up, 0_down, 1_up, 1_down, ...,
    (N-1)_up, (N-1)_down)`` — i.e. within each site the up mode comes first
    (global index ``2*site``) and the down mode second (``2*site + 1``),
    matching the on-site JW order, so the on-site and global JW strings are
    consistent.

    Returns
    -------
    dict with keys: ``I, cup, cdagup, cdown, cdagdown, nup, ndown, ntot,
    sz, double_occ, parity``. All are 4x4 tensors in the requested dtype, in
    the standard Hubbard basis ``|0>, |up>, |down>, |updown>``.
    """
    f = fermion_operators(dtype=dtype, device=device)
    I2 = f["I"]
    c2 = f["c"]
    cdag2 = f["cdag"]
    n2 = f["n"]
    F2 = f["F"]
    I4 = tc.eye(4, dtype=dtype, device=device)

    # On-site JW (up first, down second). Kronecker: up = left (high) factor,
    # down = right (low) factor. In this "JW-natural" basis the column order
    # is (0, down, up, updown); we permute to (0, up, down, updown) below.
    cup_raw = tc.kron(c2, I2)            # c on up
    cdagup_raw = tc.kron(cdag2, I2)
    cdown_raw = tc.kron(F2, c2)          # F_up x c_down (down carries up parity)
    cdagdown_raw = tc.kron(F2, cdag2)
    nup_raw = tc.kron(n2, I2)
    ndown_raw = tc.kron(I2, n2)
    # permutation swapping axes 1<->2: (0,down,up,updown) -> (0,up,down,updown)
    P = tc.tensor([[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]], dtype=dtype, device=device)
    # P is its own inverse and unitary; M_std = P M_raw P^dagger = P M_raw P.
    def _std(m):
        return P @ m @ P

    cup = _std(cup_raw)
    cdagup = _std(cdagup_raw)
    cdown = _std(cdown_raw)
    cdagdown = _std(cdagdown_raw)
    nup = _std(nup_raw)
    ndown = _std(ndown_raw)
    ntot = nup + ndown
    sz = 0.5 * (nup - ndown)
    double_occ = nup @ ndown
    # parity = F x F is diagonal diag(1,-1,-1,1) in BOTH bases (the swap of the
    # two middle axes leaves it invariant), so no permutation needed.
    parity = tc.kron(F2, F2)
    return {
        "I": I4,
        "cup": cup,
        "cdagup": cdagup,
        "cdown": cdown,
        "cdagdown": cdagdown,
        "nup": nup,
        "ndown": ndown,
        "ntot": ntot,
        "sz": sz,
        "double_occ": double_occ,
        "parity": parity,
    }
