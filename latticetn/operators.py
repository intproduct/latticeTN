"""Spin operators and dense reference Hamiltonians.

All operators use the SPIN convention S = sigma / 2 (see PHYSICS_SPEC.md).
"""

from __future__ import annotations

import numpy as np
import torch as tc


def spin_operators(dtype=tc.complex128, device="cpu") -> dict:
    """Return spin-1/2 operators {I, Sx, Sy, Sz, S+, S-} with S = sigma/2."""
    # Pauli matrices
    sx = tc.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    sy = tc.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    sz = tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    id2 = tc.eye(2, dtype=dtype, device=device)
    # spin = pauli / 2
    Sx = sx / 2.0
    Sy = sy / 2.0
    Sz = sz / 2.0
    Sp = (sx + 1j * sy) / 2.0  # S+ = Sx + i Sy = sigma_+
    Sm = (sx - 1j * sy) / 2.0  # S- = Sx - i Sy = sigma_-
    return {"I": id2, "Sx": Sx, "Sy": Sy, "Sz": Sz, "S+": Sp, "S-": Sm}


def pauli_matrices(dtype=tc.complex128, device="cpu") -> dict:
    """Return Pauli matrices {I, X, Y, Z} (sigma, = 2*S)."""
    sx = tc.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    sy = tc.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    sz = tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    id2 = tc.eye(2, dtype=dtype, device=device)
    return {"I": id2, "X": sx, "Y": sy, "Z": sz}


from .fermion_operators import fermion_operators, hubbard_local_operators


def _kron(a: tc.Tensor, b: tc.Tensor) -> tc.Tensor:
    """Kronecker product of two 2D tensors, preserving dtype/device."""
    return tc.kron(a, b)


def heisenberg_dense(
    N: int, J: float = 1.0, dtype=tc.complex128, device="cpu"
) -> tc.Tensor:
    """Dense Heisenberg Hamiltonian H = J * sum_i S.S_{i,i+1}, open boundary.

    S = sigma/2. Returns a (2**N, 2**N) matrix. Spin-spin term:
        Sx.Sx + Sy.Sy + Sz.Sz = Sz Sz + (1/2)(S+ S- + S- S+).
    """
    assert N >= 1
    ops = spin_operators(dtype=dtype, device=device)
    I = ops["I"]
    Sz = ops["Sz"]
    Sp = ops["S+"]
    Sm = ops["S-"]
    d = 2
    dim = d ** N
    H = tc.zeros((dim, dim), dtype=dtype, device=device)
    for i in range(N - 1):
        # build two-site operator on sites (i, i+1) embedded in full chain
        for op_a, op_b, coeff in [
            (Sz, Sz, 1.0),
            (Sp, Sm, 0.5),
            (Sm, Sp, 0.5),
        ]:
            term = None
            for k in range(N):
                g = None
                if k == i:
                    g = op_a
                elif k == i + 1:
                    g = op_b
                else:
                    g = I
                term = g if term is None else _kron(term, g)
            H = H + J * coeff * term
    return H


def tfi_dense(
    N: int, J: float = 1.0, h: float = 1.0, dtype=tc.complex128, device="cpu"
) -> tc.Tensor:
    """Dense transverse-field Ising Hamiltonian H = -J Sz Sz - h sum_i Sx_i.

    Uses the SAME spin convention S = sigma/2 (consistent with heisenberg_dense).
    Open boundary. Returns a (2**N, 2**N) matrix.
    """
    assert N >= 1
    ops = spin_operators(dtype=dtype, device=device)
    I = ops["I"]
    Sz = ops["Sz"]
    Sx = ops["Sx"]
    d = 2
    dim = d ** N
    H = tc.zeros((dim, dim), dtype=dtype, device=device)
    for i in range(N - 1):
        term = None
        for k in range(N):
            g = Sz if (k == i or k == i + 1) else I
            term = g if term is None else _kron(term, g)
        H = H - J * term
    for i in range(N):
        term = None
        for k in range(N):
            g = Sx if k == i else I
            term = g if term is None else _kron(term, g)
        H = H - h * term
    return H


def spinless_fermion_dense(
    N: int, t: float = 1.0, V: float = 0.0, mu: float = 0.0,
    dtype=tc.complex128, device="cpu",
) -> tc.Tensor:
    """Dense open-boundary spinless-fermion t-V Hamiltonian (JW, NOT bosonic).

        H = -t * sum_i (c^d_i c_{i+1} + c^d_{i+1} c_i)
            + V * sum_i (n_i - 1/2)(n_{i+1} - 1/2)
            - mu * sum_i (n_i - 1/2)

    with c, c^d, n the spinless fermion operators and n the number operator.
    Local basis |0> (empty), |1> (occupied), d=2. Returns a (2**N, 2**N)
    matrix.

    The global c_i / c^d_i are built WITH the explicit Jordan-Wigner parity
    string F = (-1)^n on every site to the left of site i:

        c_i  = F x ... x F x c  x I x ... x I
        c^d_i = F x ... x F x c^d x I x ... x I

    so that fermionic operators on different sites anticommute. This is a
    genuine fermionic Hamiltonian, NOT a hard-core-boson one. (For the
    nearest-neighbor terms the JW strings between the two sites cancel, so the
    dense matrix coincides with the naive two-site product — but we still
    build the global operators with the string to keep the fermionic identity
    explicit and to make the on-site term and any future longer-range term
    correct by construction.)
    """
    assert N >= 1
    ops = fermion_operators(dtype=dtype, device=device)
    I = ops["I"]
    c = ops["c"]
    cdag = ops["cdag"]
    nmh = ops["n_minus_half"]
    d = 2
    dim = d ** N
    H = tc.zeros((dim, dim), dtype=dtype, device=device)

    # on-site chemical potential term -mu * sum_i (n_i - 1/2). The number
    # operator is diagonal/local, so NO Jordan-Wigner parity string is needed
    # (a diagonal operator commutes with the parity string anyway).
    for i in range(N):
        H = H + (-mu) * _global_single(nmh, i, N, dtype, device)

    # nearest-neighbor hopping -t * (c^d_i c_{i+1} + c^d_{i+1} c_i). We build
    # the two-site operator with the explicit JW string on sites 0..i-1 (the
    # string between i and i+1 cancels). Concretely:
    #   c^d_i c_{i+1}  = F^i x c^d x c x I x ...
    #   c^d_{i+1} c_i  = F^i x c x c^d x I x ...
    for i in range(N - 1):
        def _two_site_hop(op_i, op_i1):
            term = None
            for k in range(N):
                if k < i:
                    g = ops["F"]
                elif k == i:
                    g = op_i
                elif k == i + 1:
                    g = op_i1
                else:
                    g = I
                term = g if term is None else _kron(term, g)
            return term
        # hop term: -t * (c^d_i c_{i+1} + h.c.) ; h.c. = c^d_{i+1} c_i
        H = H + (-t) * (_two_site_hop(cdag, c) + _two_site_hop(c, cdag))

    # density-density interaction V * (n_i - 1/2)(n_{i+1} - 1/2): diagonal,
    # no parity string needed.
    for i in range(N - 1):
        H = H + V * _global_two_diag(nmh, i, nmh, i + 1, N, dtype, device)

    return H


def _global_single(op, site, N, dtype, device) -> tc.Tensor:
    """Single-site operator op_{site} embedded in the chain with identity elsewhere.

    Used for the on-site chemical-potential term (n_i - 1/2), which is diagonal
    and needs no Jordan-Wigner string.
    """
    I = tc.eye(op.shape[0], dtype=dtype, device=device)
    term = None
    for k in range(N):
        g = op if k == site else I
        term = g if term is None else _kron(term, g)
    return term


def _global_two_diag(op1, site1, op2, site2, N, dtype, device) -> tc.Tensor:
    """Two-site diagonal operator op1_{site1} op2_{site2} embedded in the chain.

    Used for the density-density term (n_i - 1/2)(n_{i+1} - 1/2), which is
    diagonal so no Jordan-Wigner string is required.
    """
    I = tc.eye(op1.shape[0], dtype=dtype, device=device)
    term = None
    for k in range(N):
        if k == site1:
            g = op1
        elif k == site2:
            g = op2
        else:
            g = I
        term = g if term is None else _kron(term, g)
    return term


def exact_ground_energy(H: tc.Tensor) -> tuple[float, tc.Tensor]:
    """Return (E0, ground_state) of a dense Hermitian Hamiltonian.

    Uses numpy.linalg.eigh for CPU eigendecomposition. E0 is the lowest
    eigenvalue (float). The state is returned as a complex128 torch tensor.
    """
    H_np = H.detach().cpu().numpy()
    # symmetrize against tiny numerical non-Hermiticity
    H_np = 0.5 * (H_np + H_np.conj().T)
    w, v = np.linalg.eigh(H_np)
    E0 = float(w[0].real)
    state = tc.tensor(v[:, 0], dtype=H.dtype, device=H.device)
    return E0, state


# ---------------------------------------------------------------------------
# Spinful Hubbard chain (Stage 7C)
# ---------------------------------------------------------------------------

def _jw_global_mode(local2: tc.Tensor, g: int, Nmodes: int,
                    dtype, device) -> tc.Tensor:
    """Global 2-level operator acting on global mode ``g`` with the JW string.

    Builds ``F x ... x F x local2 x I x ... x I`` over ``Nmodes`` two-level
    modes, with the Jordan-Wigner parity ``F = (-1)^n`` on every mode left of
    ``g``, ``local2`` (a 2x2 operator) at ``g``, and identity elsewhere. This
    is the global fermionic operator for a single two-level mode in the
    natural (non-standard) per-site basis; used here as a building block and
    as an independent cross-check of the site-level standard-basis build.
    """
    ops = fermion_operators(dtype=dtype, device=device)
    I2 = ops["I"]
    F2 = ops["F"]
    term = None
    for k in range(Nmodes):
        if k < g:
            op = F2
        elif k == g:
            op = local2
        else:
            op = I2
        term = op if term is None else _kron(term, op)
    return term


def hubbard_dense(
    N: int, t: float = 1.0, U: float = 4.0, mu: float = 0.0, h: float = 0.0,
    dtype=tc.complex128, device="cpu",
) -> tc.Tensor:
    """Dense open-boundary spinful-Hubbard Hamiltonian (Jordan-Wigner).

        H = -t  sum_{i,sigma} (c^d_{i sigma} c_{i+1,sigma} + h.c.)
            + U  sum_i (n_{i up} - 1/2)(n_{i down} - 1/2)
            - mu sum_i (n_{i up} + n_{i down} - 1)
            - h  sum_i (n_{i up} - n_{i down})

    Local basis (fixed convention, see ``fermion_operators.hubbard_local_operators``):
        index 0 : |0>      (empty)
        index 1 : |up>     (one up electron)
        index 2 : |down>   (one down electron)
        index 3 : |up,down> (double occupancy)
    so ``d = 4`` and the dense matrix is ``(4**N, 4**N)``.

    Global mode ordering is fixed to **site-major**
    ``(0_up, 0_down, 1_up, 1_down, ..., (N-1)_up, (N-1)_down)``: there are
    ``2N`` two-level (single-spin) modes, the up mode of site ``i`` is global
    mode ``2*i`` and the down mode of site ``i`` is global mode ``2*i + 1``.

    This function explicitly builds the global fermionic operators with the
    Jordan-Wigner parity string ``F = (-1)^n`` on every global mode left of
    the operator, so fermionic operators on different modes anticommute. It is
    a GENUINE FERMIONIC Hamiltonian, NOT a spin model and NOT a hard-core-boson
    model: the hopping ``c^d_{i sigma} c_{i+1, sigma}`` carries the JW string
    on all global modes left of ``g_{i sigma}`` (which includes the *other
    spin* on the same site when ``sigma = down``, and all spins on earlier
    sites).

    The build is done at the SITE level in the standard Hubbard basis, which
    is algebraically identical to (and cross-checked against) the full 2N-mode
    JW build. The factorization is:

        c_{i, up}    (global) = P_0 x ... x P_{i-1} x cup_i    x I x ...
        c_{i, down}  (global) = P_0 x ... x P_{i-1} x cdown_i  x I x ...
        c^d_{i, up}  (global) = P_0 x ... x P_{i-1} x cdagup_i x I x ...
        c^d_{i, down}(global) = P_0 x ... x P_{i-1} x cdagdown_i x I x ...

    where ``P = F_up x F_down = (-1)^{n_up+n_down}`` is the per-site parity
    (the diagonal 4x4 ``parity`` operator) and ``cup/cdagup/cdown/cdagdown``
    are the local 4x4 Hubbard operators (which already contain the on-site JW
    structure: the down operator carries ``F_up`` internally, so the down
    mode's intra-site parity is built into the local matrix, and only the
    cross-site parity ``P`` on sites 0..i-1 needs to be threaded by the
    embedding). For the nearest-neighbor same-spin hop, the parity string
    between sites ``i`` and ``i+1`` cancels (it appears in both the left and
    the right global factor), reproducing the usual reduced two-site form;
    the cross-spin parity on site ``i`` for a down-hop is the ``F_up`` already
    inside ``cdown_i``/``cdagdown_i``.

    The diagonal terms (Hubbard ``U`` interaction, chemical potential ``mu``,
    field ``h``) commute with the parity string and need no JW string.
    """
    assert N >= 1
    hop = hubbard_local_operators(dtype=dtype, device=device)
    I4 = hop["I"]
    P = hop["parity"]               # per-site JW parity F_up x F_down
    cup = hop["cup"]
    cdagup = hop["cdagup"]
    cdown = hop["cdown"]
    cdagdown = hop["cdagdown"]
    nup = hop["nup"]
    ndown = hop["ndown"]
    nmh_up = nup - 0.5 * I4         # n_up - 1/2  (4x4, diagonal)
    nmh_down = ndown - 0.5 * I4     # n_down - 1/2
    ntot_m1 = nup + ndown - I4      # n_up + n_down - 1
    nup_m_ndown = nup - ndown       # n_up - n_down

    dim = 4 ** N
    H = tc.zeros((dim, dim), dtype=dtype, device=device)

    # --- Hopping: -t sum_{i,sigma} (c^d_{i sigma} c_{i+1,sigma} + h.c.) ---
    # Build the two global same-spin operators for the bond (i, i+1): the
    # left factor (c^d or c on site i) and the right factor (c or c^d on site
    # i+1). Each carries parity P on sites 0..i-1 (left) and 0..i (right),
    # but the PRODUCT only has P on sites 0..i-1 (the P on site i from the
    # right factor is squared away against... actually the right factor's P
    # on site i is the per-site parity that the left factor does NOT carry,
    # and since the left factor is c^d/c on site i (not P), the right factor
    # contributes P on site i which does NOT cancel — wait, no: the right
    # factor is c/c^d on site i+1, so its string is P on sites 0..i. The left
    # factor is c^d/c on site i, string P on sites 0..i-1. The product's
    # string on sites 0..i-1 is P*P = I (cancels), and on site i it is just
    # P (from the right factor) — which is the cross-site parity needed so
    # that, e.g., a down-hop on bond i correctly carries the up-mode parity
    # of sites 0..i. We build it directly as the product of the two global
    # operators to keep the fermionic identity explicit.
    for spin, (cL, cdagL, cR, cdagR) in enumerate([
            (cup, cdagup, cup, cdagup),      # spin = up
            (cdown, cdagdown, cdown, cdagdown),  # spin = down
    ]):
        for i in range(N - 1):
            # global c^d on site i  and  global c on site i+1  (for this spin)
            g_cdag_i = _global_hubbard(cdagL, i, N, P, I4, dtype, device)
            g_c_i = _global_hubbard(cL, i, N, P, I4, dtype, device)
            g_c_ip1 = _global_hubbard(cR, i + 1, N, P, I4, dtype, device)
            g_cdag_ip1 = _global_hubbard(cdagR, i + 1, N, P, I4, dtype, device)
            # c^d_{i} c_{i+1}  +  h.c. = c^d_{i+1} c_{i}
            H = H + (-t) * (g_cdag_i @ g_c_ip1 + g_cdag_ip1 @ g_c_i)

    # --- On-site terms (diagonal, no JW string) ---
    for i in range(N):
        # U (n_up - 1/2)(n_down - 1/2)
        H = H + U * _global_local4(nmh_up @ nmh_down, i, N, dtype, device)
        # -mu (n_up + n_down - 1)
        H = H + (-mu) * _global_local4(ntot_m1, i, N, dtype, device)
        # -h (n_up - n_down)
        H = H + (-h) * _global_local4(nup_m_ndown, i, N, dtype, device)

    return H


def _global_hubbard(local4: tc.Tensor, site: int, N: int,
                    parity4: tc.Tensor, I4: tc.Tensor,
                    dtype, device) -> tc.Tensor:
    """Global site-level Hubbard operator: parity on sites 0..site-1, local4 at site.

    Builds ``P x ... x P x local4 x I x ... x I`` over ``N`` four-level sites
    (standard Hubbard basis per site), with the per-site JW parity
    ``P = F_up x F_down`` on every site left of ``site``, the 4x4 ``local4``
    operator at ``site`` (one of cup/cdagup/cdown/cdagdown from
    ``hubbard_local_operators``), and 4x4 identity elsewhere. This is the
    global fermionic operator for a single (site, spin) mode in the standard
    Hubbard basis; the on-site JW structure (down carries F_up) is already
    inside ``local4``.
    """
    term = None
    for k in range(N):
        if k < site:
            op = parity4
        elif k == site:
            op = local4
        else:
            op = I4
        term = op if term is None else _kron(term, op)
    return term


def _global_local4(op4: tc.Tensor, site: int, N: int,
                   dtype, device) -> tc.Tensor:
    """Embed a 4x4 local Hubbard operator on ``site`` with 4x4 identity elsewhere.

    The 4x4 Hubbard operators (number, double-occ, sz, ...) are diagonal or
    on-site, so they commute with the JW parity string and need no JW string.
    """
    I4 = tc.eye(4, dtype=dtype, device=device)
    term = None
    for k in range(N):
        g = op4 if k == site else I4
        term = g if term is None else _kron(term, g)
    return term
