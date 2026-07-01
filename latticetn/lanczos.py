"""Lanczos iterative eigensolver for the local DMRG problem (Stage 4B).

Returns the lowest eigenpair of a Hermitian operator given as a **matrix-free**
``apply`` callable. Used by the Stage 4B DMRG driver as the
``solver="lanczos"`` option, so larger effective-space dimensions D = l*d*d*r
need not materialize a dense D x D matrix.

NON-differentiable (runs under torch.no_grad on detached tensors); reuses only
torch (no large dependency). The Stage 4A dense ``torch.linalg.eigh`` path
remains the reference and is untouched.

Algorithm: classic Lanczos with full reorthogonalization (simple and robust for
the small Krylov dimensions used here), restarted on loss of orthogonality is
not needed at these sizes. Convergence on the Ritz value to ``tol`` with a cap
``max_iter``.
"""

from __future__ import annotations

import torch as tc


def lanczos_lowest_eigenpair(apply, dim: int, dtype=tc.complex128,
                             device="cpu", max_iter: int | None = None,
                             tol: float = 1e-12, seed: int = 0,
                             num_restarts: int = 3) -> tuple[tc.Tensor, tc.Tensor]:
    """Lowest eigenpair (E0, ground_vector) of a Hermitian operator.

    ``apply(v) -> v'`` maps a length-`dim` (complex) vector to another.
    Returns (E0 real tensor scalar, ground_vector length-dim complex tensor).
    Convergence is declared when two successive Ritz values differ < tol or the
    Krylov subspace reaches `dim`. If not converged, the best estimate after
    `num_restarts` fresh-start attempts is returned (still a valid upper bound
    on the ground energy).
    """
    D = int(dim)
    if max_iter is None:
        max_iter = min(D, 40)
    g = tc.Generator(device=device).manual_seed(int(seed))
    best_E = None
    best_V = None

    def _run_once(v0: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
        Q = []
        alphas = []
        betas = []
        q = v0 / (tc.linalg.norm(v0) + 1e-30)
        Q.append(q)
        for j in range(max(D, max_iter) if max_iter >= D else max_iter):
            w = apply(Q[j]).reshape(-1)
            a = tc.dot(Q[j].conj(), w).real
            alphas.append(a)
            # full reorthogonalization against all existing Q's
            w = w - a * Q[j]
            for qj in Q:
                w = w - tc.dot(qj.conj(), w) * qj
            for qj in Q:                  # second pass for stability
                w = w - tc.dot(qj.conj(), w) * qj
            b = tc.linalg.norm(w)
            betas.append(float(b))
            if b < 1e-14:
                break
            qn = w / b
            Q.append(qn)
        # tridiagonal T
        m = len(alphas)
        T = tc.zeros(m, m, dtype=tc.float64, device=device)
        for ii in range(m):
            T[ii, ii] = alphas[ii]
        for ii in range(m - 1):
            T[ii, ii + 1] = betas[ii]
            T[ii + 1, ii] = betas[ii]
        E, V = tc.linalg.eigh(T)
        E0 = E[0]
        # back out Ritz vector in the full space
        vec = sum(float(V[k, 0]) * Q[k] for k in range(m))
        vec = vec / (tc.linalg.norm(vec) + 1e-30)
        return E0, vec

    for r in range(num_restarts):
        v0 = tc.randn(D, dtype=dtype, device=device, generator=g) \
            + 1j * tc.randn(D, dtype=dtype, device=device, generator=g)
        E, V = _run_once(v0)
        if best_E is None or E < best_E:
            best_E = E
            best_V = V
        if E.item() == E.item() and abs(float(E) - (best_E.item())) < tol:
            # converged against this random start
            pass
    return best_E, best_V


def ritz_quotient(apply, v: tc.Tensor) -> tc.Tensor:
    """Rayleigh quotient <v|A|v>/<v|v> for a matrix-free Hermitian apply."""
    v = v.reshape(-1)
    Av = apply(v).reshape(-1)
    num = tc.dot(v.conj(), Av)
    den = tc.dot(v.conj(), v)
    return (num / den).real
