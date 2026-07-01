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


def mps_bond_energy_heisenberg(mps, i):
    """Nearest-neighbor bond energy <S_i . S_{i+1}> of an MPS, via dense ref."""
    psi = _mps_dense_state(mps)
    return dense_bond_energy_heisenberg(psi, i, mps.N)


def mps_entanglement_entropy(mps, cut):
    """Bipartite entanglement entropy of an MPS across a cut, via dense ref."""
    psi = _mps_dense_state(mps)
    return dense_entanglement_entropy(psi, cut, mps.N)
