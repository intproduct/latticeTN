"""Hermitian Lanczos exponential action used by traditional TDVP."""

from __future__ import annotations

from dataclasses import dataclass

import torch as tc


@dataclass(frozen=True)
class KrylovInfo:
    dimension: int
    residual: float


def lanczos_expm_action(
    apply,
    vector: tc.Tensor,
    dt: float,
    *,
    krylov_dim: int = 30,
    tol: float = 1e-13,
    return_info: bool = False,
):
    """Compute ``exp(-1j * dt * H) @ vector`` for Hermitian ``H``.

    ``apply`` is matrix-free.  Full reorthogonalization is intentionally used:
    TDVP local spaces are moderate, and robustness is more valuable than the
    small extra Krylov-basis cost.  Tensor work remains on the input device,
    including CUDA.
    """
    flat = vector.reshape(-1)
    if krylov_dim < 1:
        raise ValueError("krylov_dim must be >= 1")
    norm = tc.linalg.vector_norm(flat)
    if not bool(tc.isfinite(norm)) or float(norm) == 0.0:
        raise ValueError("Lanczos exponential requires a finite nonzero vector")

    max_dim = min(int(krylov_dim), int(flat.numel()))
    basis = [flat / norm]
    alphas: list[tc.Tensor] = []
    betas: list[tc.Tensor] = []
    residual = 0.0

    for j in range(max_dim):
        w = apply(basis[j]).reshape(-1)
        alpha = tc.vdot(basis[j], w).real
        alphas.append(alpha)

        # Two-pass full reorthogonalization protects the Hermitian Krylov
        # projection from loss of orthogonality in complex128 calculations.
        for _ in range(2):
            for q in basis:
                w = w - tc.vdot(q, w) * q

        beta = tc.linalg.vector_norm(w)
        residual = float(beta)
        if j + 1 == max_dim or residual <= tol:
            break
        betas.append(beta.real)
        basis.append(w / beta)

    m = len(alphas)
    real_dtype = flat.real.dtype
    tridiagonal = tc.zeros((m, m), dtype=real_dtype, device=flat.device)
    tridiagonal.diagonal().copy_(tc.stack(alphas).to(real_dtype))
    if m > 1:
        offdiag = tc.stack(betas[: m - 1]).to(real_dtype)
        indices = tc.arange(m - 1, device=flat.device)
        tridiagonal[indices, indices + 1] = offdiag
        tridiagonal[indices + 1, indices] = offdiag

    eigenvalues, eigenvectors = tc.linalg.eigh(tridiagonal)
    phase = tc.exp((-1j * float(dt)) * eigenvalues.to(flat.dtype))
    projected = eigenvectors.to(flat.dtype) @ (
        phase * eigenvectors[0, :].to(flat.dtype) * norm.to(flat.dtype)
    )
    q_matrix = tc.stack(basis[:m], dim=1)
    evolved = (q_matrix @ projected).reshape(vector.shape)
    info = KrylovInfo(dimension=m, residual=residual)
    return (evolved, info) if return_info else evolved


__all__ = ["KrylovInfo", "lanczos_expm_action"]
