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
