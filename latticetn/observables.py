"""Stage 2 observable references and MPS observable API.

Conventions (see docs/PHYSICS_SPEC.md, docs/BENCHMARK_SPEC.md):

- Spin operators use ``S = sigma / 2`` (NOT Pauli).
- Heisenberg bond term: ``Sx.Sx + Sy.Sy + Sz.Sz = Sz Sz + (1/2)(S+ S- + S- S+)``.
- Open boundary, site 0 is the most-significant index in the dense state
  vector (consistent with ``operators.heisenberg_dense`` and ``MPS.to_dense``).

Dense functions take a state vector of shape ``(2**N,)`` (assumed normalized for
expectation values). MPS functions are Stage-2 correctness references: they
convert via ``mps.to_dense()`` and dispatch to the dense path. A later Stage 3
may replace them with canonical-MPS contractions.

These helpers are intentionally allowed to use ``.detach()``/numpy for the
``dense_entanglement_entropy`` SVD path, because entanglement entropy is an
*observable* computed from a converged/normalized state, not part of the
differentiable energy path. The expectation-value helpers below operate on the
given (already-normalized) state/mps and do not touch autograd state used by the
training energy path; they keep everything differentiable through ``psi``.
"""

from __future__ import annotations

import numpy as np
import torch as tc

from .operators import spin_operators
from .fermion_operators import fermion_operators, hubbard_local_operators


# ---------------------------------------------------------------------------
# Dense-state observable references
# ---------------------------------------------------------------------------

def _embed_local(op: tc.Tensor, site: int, N: int, dtype, device) -> tc.Tensor:
    """Dense operator acting with `op` on `site` and identity elsewhere.

    Big-endian ordering: site 0 is the leftmost (most significant) factor,
    matching operators.heisenberg_dense and MPS.to_dense.
    """
    dim = op.shape[0]
    term = None
    id2 = tc.eye(dim, dtype=dtype, device=device)
    for k in range(N):
        g = op if k == site else id2
        term = g if term is None else tc.kron(term, g)
    return term


def dense_expect_local(state, op, site, N):
    """<psi| op_site |psi> for a normalized dense state vector.

    Returns a complex scalar (kept complex so callers can inspect the phase;
    Hermitian observables yield a real value).
    """
    op = tc.as_tensor(op)
    dtype = state.dtype if tc.is_complex(state) else op.dtype
    op = op.to(dtype=dtype, device=state.device)
    O = _embed_local(op, site, N, dtype, state.device)
    # <psi|O|psi> = psi^dagger @ O @ psi
    val = state.conj() @ (O @ state)
    return val


def dense_expect_two_site(state, op1, i, op2, j, N):
    """<psi| op1_i op2_j |psi> for a normalized dense state vector.

    Sites i and j must be distinct. Big-endian ordering (site 0 most
    significant).
    """
    assert i != j, "two-site observable requires distinct sites"
    op1 = tc.as_tensor(op1)
    op2 = tc.as_tensor(op2)
    dtype = state.dtype if tc.is_complex(state) else op1.dtype
    op1 = op1.to(dtype=dtype, device=state.device)
    op2 = op2.to(dtype=dtype, device=state.device)
    dim = op1.shape[0]
    id2 = tc.eye(dim, dtype=dtype, device=state.device)
    term = None
    for k in range(N):
        if k == i:
            g = op1
        elif k == j:
            g = op2
        else:
            g = id2
        term = g if term is None else tc.kron(term, g)
    val = state.conj() @ (term @ state)
    return val


def dense_connected_correlation(state, op1, i, op2, j, N):
    """Connected correlation <op1_i op2_j> - <op1_i><op2_j>."""
    two = dense_expect_two_site(state, op1, i, op2, j, N)
    one_i = dense_expect_local(state, op1, i, N)
    one_j = dense_expect_local(state, op2, j, N)
    return two - one_i * one_j


def dense_bond_energy_heisenberg(state, i, N):
    """<S_i . S_{i+1}> for sites (i, i+1) using S = sigma/2.

    Uses Sz Sz + (1/2)(S+ S- + S- S+), matching operators.heisenberg_dense.
    Returns a real scalar (real part).
    """
    ops = spin_operators(dtype=state.dtype if tc.is_complex(state) else tc.complex128,
                         device=state.device)
    Sz, Sp, Sm = ops["Sz"], ops["S+"], ops["S-"]
    e = (
        dense_expect_two_site(state, Sz, i, Sz, i + 1, N)
        + 0.5 * dense_expect_two_site(state, Sp, i, Sm, i + 1, N)
        + 0.5 * dense_expect_two_site(state, Sm, i, Sp, i + 1, N)
    )
    return e.real


def dense_entanglement_entropy(state, cut, N):
    """Bipartite von Neumann entropy across a cut (number of sites = cut).

    cut in {1,...,N-1}: sites [0, cut) vs [cut, N). Open boundary.
    Returns a real scalar (nats, i.e. natural-log entropy).

    Uses an SVD of the (2**cut, 2**(N-cut)) reshaped state. The SVD is a pure
    observable computation on a converged state, so detaching to numpy here does
    not break any differentiable training energy path.
    """
    psi = tc.as_tensor(state)
    if tc.is_complex(psi):
        psi_np = psi.detach().cpu().numpy()
    else:
        psi_np = psi.detach().cpu().numpy().astype(np.complex128)
    left = 2 ** cut
    right = 2 ** (N - cut)
    mat = psi_np.reshape(left, right)
    # Schmidt coefficients = singular values of the bipartition matrix.
    s = np.linalg.svd(mat, compute_uv=False)
    # keep positive singular values, normalize against roundoff if needed
    s = s[s > 1e-15]
    if s.size == 0:
        return tc.tensor(0.0, dtype=tc.float64)
    # probabilities
    p = (s.astype(np.float64) ** 2)
    p = p / p.sum()
    # S = -sum p ln p
    S = float(-np.sum(p * np.log(p)))
    return tc.tensor(S, dtype=tc.float64)


# ---------------------------------------------------------------------------
# MPS observable API (Stage 2: dense-reference quality, not efficient large-N)
# ---------------------------------------------------------------------------

def _mps_dense_state(mps) -> tc.Tensor:
    """Return the normalized dense state vector of an MPS.

    Normalization is applied here as an observable post-processing step (this
    is NOT the training energy path), so the caller gets a proper quantum
    state. We avoid mutating the MPS tensors.
    """
    psi = mps.to_dense()
    nrm = tc.linalg.norm(psi)
    # guard against a (pathological) zero state
    nrm = tc.where(nrm > 0, nrm, tc.ones_like(nrm))
    return psi / nrm


def mps_expect_local(mps, op, site):
    """<Sz_site>-style local observable of an MPS, via dense reference."""
    psi = _mps_dense_state(mps)
    return dense_expect_local(psi, op, site, mps.N)


def mps_expect_two_site(mps, op1, i, op2, j):
    """Two-site observable <op1_i op2_j> of an MPS, via dense reference."""
    psi = _mps_dense_state(mps)
    return dense_expect_two_site(psi, op1, i, op2, j, mps.N)


def mps_connected_correlation(mps, op1, i, op2, j):
    """Connected two-site correlation of an MPS, via dense reference."""
    psi = _mps_dense_state(mps)
    return dense_connected_correlation(psi, op1, i, op2, j, mps.N)


def mps_bond_energy_heisenberg(mps, i):
    """Nearest-neighbor bond energy <S_i . S_{i+1}> of an MPS, via dense ref."""
    psi = _mps_dense_state(mps)
    return dense_bond_energy_heisenberg(psi, i, mps.N)


def mps_entanglement_entropy(mps, cut):
    """Bipartite entanglement entropy of an MPS across a cut, via dense ref."""
    psi = _mps_dense_state(mps)
    return dense_entanglement_entropy(psi, cut, mps.N)


# ---------------------------------------------------------------------------
# Spinless-fermion observables (Stage 7A)
# ---------------------------------------------------------------------------
#
# These are small-N dense-reference observables for the open-boundary 1D
# spinless fermion t-V chain. They are NOT full graded fermionic tensors; they
# match the Stage 11 Jordan-Wigner convention used by
# ``operators.spinless_fermion_dense`` and ``MPO.generate_spinless_fermion``.
#
# - local density <n_i>             : diagonal, no JW string needed.
# - density-density <n_i n_j>       : diagonal, no JW string needed.
# - NN hopping <c^d_i c_{i+1}+h.c.> : adjacent JW strings cancel on all sites
#   left of the bond, so this helper uses the reduced two-site product
#   (c^d on i, c on i+1) + h.c. Nonlocal Green functions would need explicit
#   JW strings and are intentionally outside this nearest-neighbor helper.
# ---------------------------------------------------------------------------


def _fermion_global_two_site(op_i, i, op_i1, i1, N, dtype, device) -> tc.Tensor:
    """Reduced adjacent spinless-fermion two-site operator.

    For nearest-neighbor Hamiltonian terms, the two Jordan-Wigner strings
    cancel on all sites left of the bond. This builds the remaining local
    product ``op_i`` at ``i`` and ``op_i1`` at ``i1`` with identity elsewhere.
    It is not the helper for nonlocal single-particle Green functions.
    """
    ops = fermion_operators(dtype=dtype, device=device)
    I = ops["I"]
    assert i < i1, "use i < i1 (left factor first)"
    term = None
    for k in range(N):
        if k == i:
            g = op_i
        elif k == i1:
            g = op_i1
        else:
            g = I
        term = g if term is None else tc.kron(term, g)
    return term


def dense_fermion_local_density(state, site: int, N: int) -> tc.Tensor:
    """<n_site> for a normalized dense fermion state vector.

    The number operator is diagonal, so no Jordan-Wigner string is needed.
    Returns a real scalar (real part).
    """
    ops = fermion_operators(dtype=state.dtype if tc.is_complex(state)
                            else tc.complex128, device=state.device)
    val = dense_expect_local(state, ops["n"], site, N)
    return val.real


def dense_fermion_density_density(state, i: int, j: int, N: int) -> tc.Tensor:
    """<n_i n_j> for a normalized dense fermion state vector (i != j).

    Diagonal observable; no JW string needed. Returns a real scalar.
    """
    assert i != j, "density-density requires distinct sites"
    ops = fermion_operators(dtype=state.dtype if tc.is_complex(state)
                            else tc.complex128, device=state.device)
    val = dense_expect_two_site(state, ops["n"], i, ops["n"], j, N)
    return val.real


def dense_fermion_nn_hopping(state, i: int, N: int) -> tc.Tensor:
    """<c^d_i c_{i+1} + c^d_{i+1} c_i> for a normalized dense fermion state.

    Nearest-neighbor hopping observable. The adjacent JW strings cancel on all
    sites left of the bond, matching the dense Hamiltonian's reduced hopping
    term. Returns a real scalar.
    """
    assert 0 <= i < N - 1, "NN hopping requires 0 <= i < N-1"
    dt = state.dtype if tc.is_complex(state) else tc.complex128
    dev = state.device
    ops = fermion_operators(dtype=dt, device=dev)
    c = ops["c"]
    cdag = ops["cdag"]
    lo, hi = (i, i + 1)
    # c^d_i c_{i+1} with the adjacent JW strings cancelled.
    op_cdag_c = _fermion_global_two_site(cdag, lo, c, hi, N, dt, dev)
    # c^d_{i+1} c_i: swap the two factors (still nearest-neighbor reduced).
    op_c_cdag = _fermion_global_two_site(c, lo, cdag, hi, N, dt, dev)
    val = state.conj() @ ((op_cdag_c + op_c_cdag) @ state)
    return val.real

def mps_fermion_local_density(mps, site: int) -> tc.Tensor:
    """<n_site> of a fermion MPS, via dense reference."""
    psi = _mps_dense_state(mps)
    return dense_fermion_local_density(psi, site, mps.N)


def mps_fermion_density_density(mps, i: int, j: int) -> tc.Tensor:
    """<n_i n_j> of a fermion MPS, via dense reference."""
    psi = _mps_dense_state(mps)
    return dense_fermion_density_density(psi, i, j, mps.N)


def mps_fermion_nn_hopping(mps, i: int) -> tc.Tensor:
    """<c^d_i c_{i+1} + h.c.> of a fermion MPS, via dense reference."""
    psi = _mps_dense_state(mps)
    return dense_fermion_nn_hopping(psi, i, mps.N)


# ---------------------------------------------------------------------------
# Spinful-Hubbard observables (Stage 7C)
# ---------------------------------------------------------------------------
#
# Small-N dense-reference observables for the open-boundary 1D spinful Hubbard
# chain. Like the spinless-fermion observables they are NOT full graded
# fermionic tensors; they use the Jordan-Wigner construction (per-site parity
# P = F_up x F_down on the left-factor site of each spin-resolved hop),
# matching ``operators.hubbard_dense`` and ``MPO.generate_hubbard``.
#
# Local basis ``|0>, |up>, |down>, |up,down>`` (d=4); global mode ordering
# site-major ``(0_up,0_down,1_up,1_down,...)``.
#
# - local <n_up_i>, <n_down_i>, <n_tot_i> : diagonal -> no JW string needed.
# - double occupancy <n_up_i n_down_i>    : diagonal (on-site product) -> no
#   JW string needed.
# - NN spin-resolved hopping <c^d_{i,s} c_{i+1,s} + h.c.> : the GLOBAL product
#   carries P on sites 0..i-1 from BOTH factors (which cancel) plus one
#   surviving P at site i from the right factor's string -> the two-site
#   operator is (c^d_sigma @ P) on site i, c_sigma on site i+1, plus its h.c.
#   (P @ c_sigma) on site i, c^d_sigma on site i+1 - exactly the form used in
#   ``operators.hubbard_dense``.
# ---------------------------------------------------------------------------


def _hubbard_global_two_site(op_i, i, op_i1, i1, N, dtype, device) -> tc.Tensor:
    """Global two-site Hubbard operator with the per-site JW parity.

    Builds the two-site operator with NO inter-site parity string (the two
    factors' strings cancel on sites 0..i-1) and the given local 4x4 operators
    at sites i and i1, identity elsewhere. The caller is responsible for
    passing the correct (c^d_sigma @ P) / (P @ c_sigma) form for op_i (this
    encapsulates the surviving site-i parity from the right factor's JW
    string); op_i1 is the bare right-factor operator (c_sigma or c^d_sigma).
    """
    hop = hubbard_local_operators(dtype=dtype, device=device)
    I = hop["I"]
    assert i < i1, "use i < i1 (left factor first)"
    term = None
    for k in range(N):
        if k == i:
            g = op_i
        elif k == i1:
            g = op_i1
        else:
            g = I
        term = g if term is None else tc.kron(term, g)
    return term


def dense_hubbard_local_density(state, site: int, N: int,
                                spin: str = "tot") -> tc.Tensor:
    """<n_{site, spin}> for a normalized dense Hubbard state vector.

    spin in {"up", "down", "tot"}. The number operator is diagonal, so no JW
    string is needed. Returns a real scalar (real part).
    """
    dt = state.dtype if tc.is_complex(state) else tc.complex128
    dev = state.device
    ops = hubbard_local_operators(dtype=dt, device=dev)
    op = {"up": ops["nup"], "down": ops["ndown"], "tot": ops["ntot"]}[spin]
    val = dense_expect_local(state, op, site, N)
    return val.real


def dense_hubbard_double_occ(state, site: int, N: int) -> tc.Tensor:
    """<n_up_site n_down_site> (double occupancy) for a dense Hubbard state.

    Diagonal on-site observable; no JW string. Returns a real scalar.
    """
    dt = state.dtype if tc.is_complex(state) else tc.complex128
    dev = state.device
    ops = hubbard_local_operators(dtype=dt, device=dev)
    val = dense_expect_local(state, ops["double_occ"], site, N)
    return val.real


def dense_hubbard_local_sz(state, site: int, N: int) -> tc.Tensor:
    """<S^z_site> = (1/2)(<n_up> - <n_down>) for a dense Hubbard state.

    Diagonal; no JW string. Returns a real scalar.
    """
    dt = state.dtype if tc.is_complex(state) else tc.complex128
    dev = state.device
    ops = hubbard_local_operators(dtype=dt, device=dev)
    val = dense_expect_local(state, ops["sz"], site, N)
    return val.real


def dense_hubbard_nn_hopping(state, i: int, N: int,
                             spin: str = "up") -> tc.Tensor:
    """<c^d_{i,s} c_{i+1,s} + h.c.> for a normalized dense Hubbard state.

    Spin-resolved nearest-neighbor hopping observable. Built with the
    surviving per-site parity P at the left-factor site (matching the dense
    Hamiltonian's hopping term), so it is a genuine fermionic observable.
    spin in {"up", "down"}. Returns a real scalar.
    """
    assert 0 <= i < N - 1, "NN hopping requires 0 <= i < N-1"
    dt = state.dtype if tc.is_complex(state) else tc.complex128
    dev = state.device
    ops = hubbard_local_operators(dtype=dt, device=dev)
    P = ops["parity"]
    if spin == "up":
        cdag, c = ops["cdagup"], ops["cup"]
    elif spin == "down":
        cdag, c = ops["cdagdown"], ops["cdown"]
    else:
        raise ValueError(f"spin must be 'up' or 'down', got {spin!r}")
    lo, hi = (i, i + 1)
    # c^d_{i,s} c_{i+1,s} : left factor (cdag @ P) at i, right factor c at i+1
    op_cdag_c = _hubbard_global_two_site(cdag @ P, lo, c, hi, N, dt, dev)
    # h.c. c^d_{i+1,s} c_i : left factor (P @ c) at i, right factor cdag at i+1
    op_c_cdag = _hubbard_global_two_site(P @ c, lo, cdag, hi, N, dt, dev)
    val = state.conj() @ ((op_cdag_c + op_c_cdag) @ state)
    return val.real


def mps_hubbard_local_density(mps, site: int, spin: str = "tot") -> tc.Tensor:
    """<n_{site, spin}> of a Hubbard MPS, via dense reference."""
    psi = _mps_dense_state(mps)
    return dense_hubbard_local_density(psi, site, mps.N, spin=spin)


def mps_hubbard_double_occ(mps, site: int) -> tc.Tensor:
    """<n_up_site n_down_site> of a Hubbard MPS, via dense reference."""
    psi = _mps_dense_state(mps)
    return dense_hubbard_double_occ(psi, site, mps.N)


def mps_hubbard_local_sz(mps, site: int) -> tc.Tensor:
    """<S^z_site> of a Hubbard MPS, via dense reference."""
    psi = _mps_dense_state(mps)
    return dense_hubbard_local_sz(psi, site, mps.N)


def mps_hubbard_nn_hopping(mps, i: int, spin: str = "up") -> tc.Tensor:
    """<c^d_{i,s} c_{i+1,s} + h.c.> of a Hubbard MPS, via dense reference."""
    psi = _mps_dense_state(mps)
    return dense_hubbard_nn_hopping(psi, i, mps.N, spin=spin)
