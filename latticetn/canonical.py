"""MPS canonicalization and SVD compression (Stage 3A).

This module provides NON-differentiable canonical-form operations on MPS:

- left-canonical / right-canonical / mixed-canonical (with center site)
- canonical-form norm check
- SVD-based bond compression with truncation-error reporting
- canonical-form bipartite entanglement entropy across a cut
- dense-state -> MPS construction (successive SVDs, optional truncation)

Physics conventions are unchanged from Stage 1/2 (S = sigma/2, J = 1.0, open
boundary, complex128). These routines are preprocessing/postprocessing steps:
they detach from the autograd graph (operating under ``torch.no_grad`` on plain
tensors) and DO NOT touch the differentiable ``energy_with_MPO`` path. No
``.detach``/``.data``/unnecessary ``.item`` is added to the energy path.

Tensor convention (see latticetn/mps.py): each site tensor has shape
``(left_bond, phys, right_bond) = (l, d, r)``.

- Left-canonical site A:  reshape ``(l*d, r)``, columns orthonormal
  (A^H A = I_r).
- Right-canonical site B: reshape ``(l, d*r)``, rows orthonormal
  (B B^H = I_l).
- Mixed-canonical with center c: sites < c are left-canonical, sites > c are
  right-canonical, site c carries the entanglement (the Schmidt coefficients
  live on its bonds).

QR (left sweep) and LQ-via-QR-transpose (right sweep) are exact
(rank-preserving); SVD truncation in ``svd_compress`` reduces bond dimension
and reports the discarded weight per bond.
"""

from __future__ import annotations

import math

import torch as tc

from .mps import MPS


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _detach_clone(mps: "MPS") -> list[tc.Tensor]:
    return [t.detach().clone() for t in mps.tensors]


def _wrap(mps: "MPS", tensors: list[tc.Tensor]) -> "MPS":
    return MPS.from_tensors(tensors, dtype=mps.dtype, device=mps.device,
                            requires_grad=False)


def _svd(M: tc.Tensor):
    """Complex-safe reduced SVD. Returns (U, S, Vh) with S real, >=0."""
    return tc.linalg.svd(M, full_matrices=False)


# ---------------------------------------------------------------------------
# orthonormality diagnostics (used by tests)
# ---------------------------------------------------------------------------

def left_orthonormal_error(A: tc.Tensor) -> float:
    """Max |A^H A - I| for tensor (l,d,r) treated as left-canonical.

    A^H A is contraction over (l, s), leaving the right bond: should be I_r.
    """
    l, d, r = A.shape
    M = A.reshape(l * d, r)
    G = M.conj().t() @ M
    eye = tc.eye(r, dtype=A.dtype, device=A.device)
    return float((G - eye).abs().max())


def right_orthonormal_error(B: tc.Tensor) -> float:
    """Max |B B^H - I| for tensor (l,d,r) treated as right-canonical.

    B B^H is contraction over (s, r), leaving the left bond: should be I_l.
    """
    l, d, r = B.shape
    M = B.reshape(l, d * r)
    G = M @ M.conj().t()
    eye = tc.eye(l, dtype=B.dtype, device=B.device)
    return float((G - eye).abs().max())


def left_orthonormal_all(mps: "MPS", upto: int | None = None) -> float:
    """Max left-orthonormality error over sites [0, upto)."""
    n = upto if upto is not None else (mps.N - 1)
    errs = [left_orthonormal_error(mps.tensors[i]) for i in range(n)]
    return max(errs) if errs else 0.0


def right_orthonormal_all(mps: "MPS", from_: int | None = None) -> float:
    """Max right-orthonormality error over sites (from_, N-1]."""
    start = (from_ + 1) if from_ is not None else 1
    errs = [right_orthonormal_error(mps.tensors[i]) for i in range(start, mps.N)]
    return max(errs) if errs else 0.0


# ---------------------------------------------------------------------------
# canonicalization sweeps
# ---------------------------------------------------------------------------

def left_canonical(mps: "MPS") -> "MPS":
    """Return an MPS in left-canonical form (sites 0..N-2 left-orthonormal).

    Exact (rank-preserving QR sweep); the dense state is preserved up to a
    global phase. The last site carries the norm.
    """
    tensors = _detach_clone(mps)
    N = len(tensors)
    with tc.no_grad():
        for i in range(N - 1):
            l, d, r = tensors[i].shape
            Mat = tensors[i].reshape(l * d, r)
            Q, R = tc.linalg.qr(Mat)           # Q:(l*d,k) col-orth, R:(k,r)
            k = Q.shape[1]
            tensors[i] = Q.reshape(l, d, k)
            nxt = tensors[i + 1]               # (r, d, r')
            tensors[i + 1] = tc.einsum("kr,rdc->kdc", R, nxt)
    return _wrap(mps, tensors)


def right_canonical(mps: "MPS") -> "MPS":
    """Return an MPS in right-canonical form (sites 1..N-1 right-orthonormal).

    Exact LQ sweep (via QR on the transposed mat). Dense state preserved up to
    a global phase; the first site carries the norm.
    """
    tensors = _detach_clone(mps)
    N = len(tensors)
    with tc.no_grad():
        for i in range(N - 1, 0, -1):
            l, d, r = tensors[i].shape
            Mat = tensors[i].reshape(l, d * r)        # (l, d*r)
            # LQ: Mat = L B, B row-orthonormal. Via qr(Mat.t()):
            Qt, Rm = tc.linalg.qr(Mat.t())            # Mat.t():(d*r,l); Qt:(d*r,k); Rm:(k,l)
            k = Rm.shape[0]
            B = Qt.t().reshape(k, d, r)               # (k,d,r) row-orth
            tensors[i] = B
            C = Rm.t()                                # (l, k)  residual on left bond
            prev = tensors[i - 1]                     # (l', d, l)
            tensors[i - 1] = tc.einsum("abc,ce->abe", prev, C)
    return _wrap(mps, tensors)


def mixed_canonical(mps: "MPS", center: int) -> "MPS":
    """Return an MPS in mixed-canonical form with center site ``center``.

    Sites < center are left-canonical, sites > center are right-canonical, and
    site ``center`` carries the entanglement. The Schmidt coefficients across
    the cut ``[0, center) | [center, N)`` (i.e. the bond to the LEFT of the
    center) are readable from an SVD of the center tensor reshaped
    ``(l, d*r)``; see ``entanglement_entropy``.
    """
    if not (0 <= center < mps.N):
        raise ValueError(f"center {center} out of range for N={mps.N}")
    tensors = _detach_clone(mps)
    N = len(tensors)
    with tc.no_grad():
        # left sweep: sites 0 .. center-1
        for i in range(0, center):
            l, d, r = tensors[i].shape
            Mat = tensors[i].reshape(l * d, r)
            Q, R = tc.linalg.qr(Mat)
            k = Q.shape[1]
            tensors[i] = Q.reshape(l, d, k)
            nxt = tensors[i + 1]
            tensors[i + 1] = tc.einsum("kr,rdc->kdc", R, nxt)
        # right sweep: sites N-1 .. center+1
        for i in range(N - 1, center, -1):
            l, d, r = tensors[i].shape
            Mat = tensors[i].reshape(l, d * r)
            Qt, Rm = tc.linalg.qr(Mat.t())
            k = Rm.shape[0]
            B = Qt.t().reshape(k, d, r)
            tensors[i] = B
            C = Rm.t()
            prev = tensors[i - 1]
            tensors[i - 1] = tc.einsum("abc,ce->abe", prev, C)
    return _wrap(mps, tensors)


def _left_svd_canonical(mps: "MPS") -> "MPS":
    """Exact non-truncating SVD analogue of :func:`left_canonical`."""
    tensors = _detach_clone(mps)
    with tc.no_grad():
        for i in range(len(tensors) - 1):
            l, d, r = tensors[i].shape
            U, S, Vh = _svd(tensors[i].reshape(l * d, r))
            k = U.shape[1]
            tensors[i] = U.reshape(l, d, k)
            residual = S[:, None].to(Vh.dtype) * Vh
            tensors[i + 1] = tc.einsum("kr,rdc->kdc", residual, tensors[i + 1])
    return _wrap(mps, tensors)


def left_canonicalize(mps: "MPS", method: str = "qr") -> "MPS":
    """Return an exact, non-truncated left-canonical representative."""
    if method == "qr":
        return left_canonical(mps)
    if method == "svd":
        return _left_svd_canonical(mps)
    raise ValueError(f"canonicalization method must be 'qr' or 'svd', got {method!r}")


def right_canonicalize(mps: "MPS", method: str = "qr") -> "MPS":
    """Return an exact, non-truncated right-canonical representative."""
    if method != "qr":
        # An exact SVD right sweep is obtained by reversing the QR/SVD roles;
        # Stage 12A only requires SVD as an exact split diagnostic.
        raise ValueError("right canonicalization currently supports method='qr' only")
    return right_canonical(mps)


def mixed_canonicalize(mps: "MPS", center: int, method: str = "qr") -> "MPS":
    """Return an exact mixed-canonical representative with a chosen center."""
    if method != "qr":
        raise ValueError("mixed canonicalization currently supports method='qr' only")
    return mixed_canonical(mps, center)


def normalize_center(mps: "MPS", center: int | None = None, *, atol: float = 0.0) -> "MPS":
    """Normalize a mixed-canonical MPS by scaling only its center tensor.

    The input is first brought to mixed-canonical form, so this is an exact
    gauge change followed by the single global normalization required to pick
    a unit representative of the same physical ray.
    """
    if center is None:
        center = mps.N - 1
    out = mixed_canonical(mps, center)
    with tc.no_grad():
        norm = out.tensors[center].norm()
        if not bool(tc.isfinite(norm)) or float(norm) <= atol:
            raise ValueError(f"cannot normalize MPS center with norm {float(norm)!r}")
        out.tensors[center].div_(norm)
    return out


def canonical_residual(mps: "MPS", center: int | None = None) -> float:
    """Maximum mixed-canonical isometry residual around ``center``."""
    if center is None:
        center = mps.N - 1
    if not (0 <= center < mps.N):
        raise ValueError(f"center {center} out of range for N={mps.N}")
    left = [left_orthonormal_error(mps.tensors[i]) for i in range(center)]
    right = [right_orthonormal_error(mps.tensors[i]) for i in range(center + 1, mps.N)]
    return max(left + right, default=0.0)


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------

def canonical_norm(mps: "MPS") -> float:
    """sqrt(<psi|psi>) via the (unmodified) overlap path.

    This is the same quantity as the dense norm; canonicalization preserves it
    up to machine precision (the state is preserved up to a global phase).
    """
    return float(mps.overlap(mps).real.sqrt().real)


def center_frob_norm(mps: "MPS", center: int) -> float:
    """Frobenius norm of the (non-orthonormal) center tensor.

    In exact mixed-canonical form the whole state norm equals the Frobenius
    norm of the center tensor. Useful as an independent norm cross-check.
    """
    return float(mps.tensors[center].norm())


# ---------------------------------------------------------------------------
# SVD compression
# ---------------------------------------------------------------------------

def svd_compress(mps: "MPS", chi: int) -> tuple["MPS", dict]:
    """Left-canonical SVD-truncated compression, capping every bond at ``chi``.

    Returns (compressed_mps, info) where info has:
        - truncation_errors: list[float], per-bond discarded weight
              (sum of discarded s^2) / (sum of all s^2), in [0, 1].
        - bond_dims:         list[int], resulting bond dimension per bond.
        - max_bond_dim:      int
        - total_truncation:  float (sum of per-bond discarded weights, NOT a
              rigorous total but a useful aggregate).
    The compressed MPS is left-canonical (sites 0..N-2 left-orthonormal); the
    last site carries the residual norm.
    """
    if chi < 1:
        raise ValueError("chi must be >= 1")
    tensors = _detach_clone(mps)
    N = len(tensors)
    trunc_errors: list[float] = []
    bond_dims: list[int] = []
    with tc.no_grad():
        for i in range(N - 1):
            l, d, r = tensors[i].shape
            Mat = tensors[i].reshape(l * d, r)
            U, S, Vh = _svd(Mat)                 # U:(l*d,k0), S:(k0,), Vh:(k0,r)
            k0 = S.shape[0]
            k = min(chi, k0)
            s2 = (S.real ** 2)
            total = float(s2.sum())
            if total > 0:
                keep = float(s2[:k].sum())
                trunc_err = (total - keep) / total
            else:
                trunc_err = 0.0
            trunc_errors.append(trunc_err)
            bond_dims.append(k)
            U = U[:, :k]
            Sk = S[:k]
            Vhk = Vh[:k, :]
            tensors[i] = U.reshape(l, d, k)
            SVh = Sk.reshape(k, 1) * Vhk         # (k, r)
            nxt = tensors[i + 1]                 # (r, d, r')
            tensors[i + 1] = tc.einsum("kr,rdc->kdc", SVh, nxt)
    info = {
        "truncation_errors": trunc_errors,
        "bond_dims": bond_dims,
        "max_bond_dim": max(bond_dims) if bond_dims else 1,
        "total_truncation": float(sum(trunc_errors)),
        "chi": chi,
    }
    return _wrap(mps, tensors), info


# ---------------------------------------------------------------------------
# entanglement entropy (canonical)
# ---------------------------------------------------------------------------

def entanglement_entropy(mps: "MPS", cut: int) -> float:
    """Bipartite von Neumann entropy across the cut ``[0,cut) | [cut,N)``.

    ``cut in {1, ..., N-1}``. Brings the MPS to mixed-canonical form with
    center = ``cut`` and reads the Schmidt spectrum from an SVD of the center
    tensor (reshaped ``(l, d*r)``). Returns the natural-log entropy (nats).
    """
    if not (1 <= cut <= mps.N - 1):
        raise ValueError(f"cut {cut} out of range (need 1..N-1 for N={mps.N})")
    mc = mixed_canonical(mps, center=cut)
    T = mc.tensors[cut]
    l, d, r = T.shape
    M = T.reshape(l, d * r)
    _, S, _ = _svd(M)
    s = S.real
    s = s[s > 1e-15]
    if s.numel() == 0:
        return 0.0
    p = s ** 2
    p = p / p.sum()
    return float(-(p * p.log()).sum())


# ---------------------------------------------------------------------------
# dense -> MPS construction
# ---------------------------------------------------------------------------

def from_dense(state: tc.Tensor, N: int, dim: int = 2, chi: int | None = None,
               dtype=tc.complex128, device="cpu") -> "MPS":
    """Build an MPS from a dense state vector via successive SVDs.

    Exact when ``chi`` is None or large enough; otherwise the bond dimension is
    capped at ``chi`` (left-canonical truncation). Site 0 is the most
    significant index (consistent with ``MPS.to_dense`` and
    ``operators.heisenberg_dense``).
    """
    psi = tc.as_tensor(state).to(dtype=dtype, device=device).reshape(dim ** N)
    tensors: list[tc.Tensor] = []
    chi_left = 1
    rest = psi.reshape(chi_left * dim, dim ** (N - 1))     # (1*d, dim^{N-1})
    with tc.no_grad():
        for i in range(N - 1):
            U, S, Vh = _svd(rest)                          # U:(chi_left*d,k0); Vh:(k0,R)
            k0 = S.shape[0]
            k = min(chi, k0) if chi is not None else k0
            U = U[:, :k]
            tensors.append(U.reshape(chi_left, dim, k))
            S = S[:k]
            Vh = Vh[:k, :]
            sVh = (S.reshape(k, 1) * Vh)                   # (k, R)
            R_new = dim ** (N - 2 - i)
            rest = sVh.reshape(k, dim, R_new).reshape(k * dim, R_new)
            chi_left = k
        tensors.append(rest.reshape(chi_left, dim, 1))     # last site
    return MPS.from_tensors(tensors, dtype=dtype, device=device,
                            requires_grad=False)
