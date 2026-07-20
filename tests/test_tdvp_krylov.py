"""Hermitian Lanczos exponential-action validation."""

import torch as tc

from latticetn.tdvp.krylov import lanczos_expm_action


DTYPE = tc.complex128


def test_lanczos_expm_action_matches_dense_matrix_exponential():
    generator = tc.Generator().manual_seed(1201)
    raw = tc.randn((12, 12), dtype=DTYPE, generator=generator)
    hamiltonian = 0.5 * (raw + raw.conj().transpose(0, 1))
    vector = tc.randn(12, dtype=DTYPE, generator=generator)
    dt = 0.137

    got, info = lanczos_expm_action(
        lambda x: hamiltonian @ x,
        vector,
        dt,
        krylov_dim=12,
        tol=1e-14,
        return_info=True,
    )
    expected = tc.matrix_exp(-1j * dt * hamiltonian) @ vector

    assert tc.allclose(got, expected, atol=2e-12, rtol=2e-12)
    assert abs(float(tc.linalg.vector_norm(got) - tc.linalg.vector_norm(vector))) < 2e-12
    assert 1 <= info.dimension <= 12


def test_lanczos_expm_action_supports_negative_time_for_projector_splitting():
    diagonal = tc.tensor([-0.7, 0.1, 1.4], dtype=tc.float64)
    vector = tc.tensor([1.0 + 0.2j, -0.3j, 0.4], dtype=DTYPE)
    got = lanczos_expm_action(
        lambda x: diagonal.to(DTYPE) * x,
        vector,
        -0.25,
        krylov_dim=3,
    )
    expected = tc.exp(0.25j * diagonal).to(DTYPE) * vector
    assert tc.allclose(got, expected, atol=1e-12, rtol=1e-12)
