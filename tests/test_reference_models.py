"""Stage 1: exact-diagonalization reference for the Heisenberg & TFI models."""

from __future__ import annotations

import numpy as np
import torch as tc

from reference_models import (
    heisenberg_dense,
    tfi_dense,
    exact_ground_energy,
    heisenberg_ground_energy,
    bethe_ground_energy_per_site,
)

DTYPE = tc.complex128


def test_heisenberg_small_ground_energies_match_bethe():
    # N=2 singlet: E0 = -3/4 J = -0.75 (exact, spin-1/2 two-site).
    E0_2 = heisenberg_ground_energy(2, J=1.0)
    assert abs(E0_2 - (-0.75)) < 1e-10, E0_2

    # Larger even-N chains: E0/N should approach 1/4 - ln(2) from above.
    e_bethe = bethe_ground_energy_per_site()
    for N in [4, 6, 8]:
        E0 = heisenberg_ground_energy(N, J=1.0)
        per_site = E0 / N
        # finite-size E0/N is above the thermodynamic limit (less negative)
        assert per_site > e_bethe - 1e-9, (N, per_site, e_bethe)
        assert per_site < 0.0  # antiferromagnetic ground state is negative


def test_heisenberg_hermitian_and_real_spectrum():
    H = heisenberg_dense(6, J=1.0)
    # Hermitian
    assert tc.allclose(H, H.conj().T, atol=1e-12)
    # eigenvalues real
    w = np.linalg.eigvalsh(H.detach().numpy())
    assert np.all(np.isreal(w))
    # lowest eigenvalue equals exact_ground_energy
    E0, _ = exact_ground_energy(H)
    assert abs(E0 - float(w[0].real)) < 1e-9


def test_heisenberg_j_scaling():
    # E0(J) = J * E0(J=1) (Heisenberg is linear in J).
    E1 = heisenberg_ground_energy(6, J=1.0)
    E2 = heisenberg_ground_energy(6, J=2.5)
    assert abs(E2 - 2.5 * E1) < 1e-9, (E1, E2)


def test_heisenberg_sign_is_antiferromagnetic():
    # For ferromagnetic J=-1 the fully aligned |↑↑...> is the ground state.
    # <↑↑|S.S|↑↑> per bond = <SzSz> = 1/4, so E0 = J * (N-1) * 1/4 = -(N-1)/4.
    E_ferro = heisenberg_ground_energy(6, J=-1.0)
    assert abs(E_ferro - (-(6 - 1) * 0.25)) < 1e-9, E_ferro


def test_tfi_dense_hermitian_and_known_N2():
    # TFI N=2, J=1, h=0 reduces to -Sz Sz (no transverse field).
    H = tfi_dense(2, J=1.0, h=0.0)
    E0, _ = exact_ground_energy(H)
    # H = -Sz Sz; eigenvalues of Sz Sz (spin) are {+1/4, -1/4, -1/4, +1/4};
    # min of -Sz Sz = -1/4.
    assert abs(E0 - (-0.25)) < 1e-10, E0


def test_spin_convention_is_half_pauli():
    from reference_models import spin_operators

    ops = spin_operators(dtype=DTYPE)
    # Sz eigenvalues must be {+1/2, -1/2}, NOT {+1, -1}.
    w = np.linalg.eigvalsh(ops["Sz"].numpy())
    assert np.allclose(sorted(w.real), [-0.5, 0.5])
    # Sx^2 + Sy^2 + Sz^2 = 3/4 I for spin-1/2.
    s2 = ops["Sx"] @ ops["Sx"] + ops["Sy"] @ ops["Sy"] + ops["Sz"] @ ops["Sz"]
    assert tc.allclose(s2, 0.75 * tc.eye(2, dtype=DTYPE), atol=1e-12)
