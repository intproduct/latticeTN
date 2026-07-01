"""Native MPS / MPO tensor-network contractions (Stage 3B).

These contractions evaluate norms and observables by **contracting the MPS/MPO
tensors directly** (left-to-right sweeps building environment tensors), so they
scale polynomially in N and chi and do NOT call ``MPS.to_dense()``. This is the
scalable path that Stage 2's observables lacked.

Differentiability:
- The energy path (``native_mpo_numerator`` / ``rayleigh_energy_native``) is
  fully differentiable: NO ``.detach()`` / ``.data`` / unnecessary ``.item()``
  / ``torch.no_grad()``. Gradients flow back to the MPS tensors.
- Observable / report helpers (``native_local_expect`` etc.) may be called under
  ``torch.no_grad()`` by callers; they themselves stay differentiable so they
  can also be used inside a loss if desired.

Tensor conventions (see latticetn/mps.py, latticetn/mpo.py):
- MPS site tensor A: (left_bond, phys, right_bond) = (l, s, r), site 0 most
  significant in the dense state.
- MPO site tensor W: (left_bond, right_bond, phys_in, phys_out) = (l, r, si, so).

Open boundary: the left bond of site 0 and the right bond of site N-1 are 1.
"""

from __future__ import annotations

import torch as tc

from .operators import spin_operators  # noqa: F401  (re-exported convenience)


# ---------------------------------------------------------------------------
# environment builders (differentiable)
# ---------------------------------------------------------------------------

def _left_norm_env(tensors: list[tc.Tensor], up_to: int) -> tc.Tensor:
    """Left environment of <psi|psi> at bond `up_to`: shape (l_bra, l_ket).

    Contracts sites [0, up_to): environment = sum over phys of A* (bra) and A
    (ket). Starts from (1,1). Differentiable through `tensors`.
    """
    v = tc.ones((1, 1), dtype=tensors[0].dtype, device=tensors[0].device)
    for i in range(up_to):
        A = tensors[i]
        # v:(lb,lk); A*:(lb,s,rb); A:(lk,s,rk)
        v = tc.einsum("lk,lsb,ksm->bm", v, A.conj(), A)
    return v


def _right_norm_env(tensors: list[tc.Tensor], from_: int) -> tc.Tensor:
    """Right environment of <psi|psi> at bond `from_`: shape (r_bra, r_ket).

    Contracts sites [from_, N) right-to-left.
    """
    v = tc.ones((1, 1), dtype=tensors[-1].dtype, device=tensors[-1].device)
    for i in range(len(tensors) - 1, from_ - 1, -1):
        A = tensors[i]
        # v:(rb,rk); A*:(lb,s,rb); A:(lk,s,rk)  with lb=rb(of v)
        v = tc.einsum("ab,asc,bsd->cd", v, A.conj(), A)
    return v


def native_norm_sq(mps) -> tc.Tensor:
    """<psi|psi> via native contraction (no to_dense). Fully differentiable.

    Left-to-right sweep of the (bra, ket) environment.
    """
    tensors = mps.tensors
    v = tc.ones((1, 1), dtype=tensors[0].dtype, device=tensors[0].device)
    for A in tensors:
        v = tc.einsum("lk,lsb,ksm->bm", v, A.conj(), A)
    return v.reshape(())


def native_norm(mps) -> tc.Tensor:
    """sqrt(<psi|psi>) (nonnegative real). Differentiable."""
    return native_norm_sq(mps).real.clamp_min(0).sqrt()


# ---------------------------------------------------------------------------
# local & two-site expectations (differentiable)
# ---------------------------------------------------------------------------

def native_local_expect(mps, op: tc.Tensor, site: int) -> tc.Tensor:
    """<psi| op_site |psi> via native contraction. Differentiable.

    op: (d, d) acting on the physical index at `site`. Builds the left env up
    to `site`, inserts op, then sweeps the right env to the end.
    """
    tensors = mps.tensors
    op = tc.as_tensor(op).to(dtype=tensors[0].dtype, device=tensors[0].device)
    left = _left_norm_env(tensors, up_to=site)            # (lb, lk)
    A = tensors[site]                                     # (lb, s, rb)
    # insert op between bra and ket physical legs:
    # left:(lb,lk); A*:(lb,s,rb); op:(s,s'); A:(lk,s',rk)
    mid = tc.einsum("lk,lsb,st,ktm->bm", left, A.conj(), op, A)
    return _finish_right(tensors, site, mid)


def _finish_right(tensors, site: int, mid: tc.Tensor) -> tc.Tensor:
    """Contract mid (rb_bra, rb_ket) rightward through sites (site, N).

    v:(rb_bra=a, rb_ket=b); next A*:(lb=a,s,rb=c); A:(lb=b,s,rb=d) -> (c,d).
    """
    v = mid
    for i in range(site + 1, len(tensors)):
        A = tensors[i]
        v = tc.einsum("ab,asc,bsd->cd", v, A.conj(), A)
    return v.reshape(())


def native_two_site_expect(mps, op1: tc.Tensor, i: int,
                           op2: tc.Tensor, j: int) -> tc.Tensor:
    """<psi| op1_i op2_j |psi> via native contraction. Differentiable.

    Sites i, j distinct. Sweeps left env to i, inserts op1, sweeps to j,
    inserts op2, sweeps to the end.
    """
    assert i != j, "two-site contraction requires distinct sites"
    if i > j:
        # canonical order: keep i<j by symmetry of the algorithm? Not necessarily
        # symmetric for non-commuting ops, so respect caller order.
        pass
    tensors = mps.tensors
    dt = tensors[0].dtype
    dev = tensors[0].device
    op1 = tc.as_tensor(op1).to(dtype=dt, device=dev)
    op2 = tc.as_tensor(op2).to(dtype=dt, device=dev)
    lo, hi = (i, j) if i < j else (j, i)
    op_lo, op_hi = (op1, op2) if i < j else (op2, op1)

    # left env to lo, insert op_lo.
    # v:(lb_bra=a, lb_ket=b); A*:(lb=a, s, rb=c); op:(s, t); A:(lb=b, t, rb=d)
    # -> (rb_bra=c, rb_ket=d)
    v = _left_norm_env(tensors, up_to=lo)                 # (lb_bra, lb_ket)
    A = tensors[lo]
    v = tc.einsum("ab,asc,st,btd->cd", v, A.conj(), op_lo, A)
    # sweep to hi (plain norm contractions): v:(rb_bra=a,rb_ket=b); B*:(lb=a,s,rb=c); B:(lb=b,s,rb=d)
    for k in range(lo + 1, hi):
        B = tensors[k]
        v = tc.einsum("ab,asc,bsd->cd", v, B.conj(), B)
    # insert op_hi: same structure as op_lo insertion.
    A = tensors[hi]
    v = tc.einsum("ab,asc,st,btd->cd", v, A.conj(), op_hi, A)
    # finish right
    return _finish_right(tensors, hi, v)


def native_bond_energy_heisenberg(mps, i: int) -> tc.Tensor:
    """<S_i . S_{i+1}> via native two-site contractions. Differentiable.

    S.S = Sz Sz + (1/2)(S+ S- + S- S+), matching operators.heisenberg_dense.
    """
    ops = spin_operators(dtype=mps.tensors[0].dtype, device=mps.tensors[0].device)
    val = (
        native_two_site_expect(mps, ops["Sz"], i, ops["Sz"], i + 1)
        + 0.5 * native_two_site_expect(mps, ops["S+"], i, ops["S-"], i + 1)
        + 0.5 * native_two_site_expect(mps, ops["S-"], i, ops["S+"], i + 1)
    )
    return val.real


def native_correlation(mps, op: tc.Tensor, i: int, j: int) -> tc.Tensor:
    """Native two-point correlation <op_i op_j> (e.g. <Sz_i Sz_j>). Real part for Hermitian op."""
    return native_two_site_expect(mps, op, i, op, j)


# ---------------------------------------------------------------------------
# MPO expectation & Rayleigh energy (differentiable energy path)
# ---------------------------------------------------------------------------

def native_mpo_numerator(mps, mpo) -> tc.Tensor:
    """<psi| MPO |psi> via native (bra, mpo, ket) left sweep. Differentiable.

    Environment shape (l_bra, l_mpo, l_ket). Same contraction structure as
    MPS._expect_MPO but standalone and re-usable.
    """
    tensors = mps.tensors
    Ws = mpo.tensors
    assert mps.N == mpo.length
    v = tc.ones((1, 1, 1), dtype=tensors[0].dtype, device=tensors[0].device)
    for A, W in zip(tensors, Ws):
        # v:(lb,lm,lk); A*:(lb,sb,rb); W:(lm,rm,si,so); A:(lk,si,rk)
        # shared: lb=v&Ab, lm=v&W, lk=v&Ak, sb=Ab&so(so=W's 4th idx), si=W&Ak3
        v = tc.einsum("lmr,lsb,mtys,ryz->btz", v, A.conj(), W, A)
    return v.reshape(())


def native_mpo_expectation(mps, mpo) -> tc.Tensor:
    """<psi|MPO|psi> / <psi|psi> (Rayleigh ratio). Differentiable.

    Note: callers that want the *raw* numerator (e.g. for custom denominators)
    should use native_mpo_numerator. This helper divides by the native norm-sq.
    """
    num = native_mpo_numerator(mps, mpo)
    den = native_norm_sq(mps)
    e = num / den
    return e.real


def rayleigh_energy_native(mps, mpo) -> tc.Tensor:
    """Alias of native_mpo_expectation; the native Rayleigh quotient energy.

    Fully differentiable energy path. Equivalent to MPS.energy_with_MPO but
    reached purely through the standalone contraction module.
    """
    return native_mpo_expectation(mps, mpo)
