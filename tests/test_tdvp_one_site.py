"""Stage 12B-1 one-site TDVP scientific validation."""

import torch as tc

from latticetn import canonical, contractions
from latticetn.initial_states import neel_spin_state
from latticetn.mpo import MPO
from latticetn.operators import heisenberg_dense, spin_operators
from latticetn.tdvp import TDVP, TDVPResult


DTYPE = tc.complex128


def _full_bond_neel(n: int):
    """Embed the product state in the full fixed-bond MPS manifold."""
    product = neel_spin_state(n, dtype=DTYPE)
    chi = 2 ** (n // 2)
    return product, canonical.from_dense(product.to_dense(), n, chi=chi, dtype=DTYPE)


def test_one_site_tdvp_conserves_norm_energy_and_canonical_gauge():
    n = 6
    _, initial = _full_bond_neel(n)
    mpo = MPO.from_bonds(n, 2, dtype=DTYPE).generate_heisenberg()
    result = TDVP(initial, mpo, dt=0.01, method="one_site").evolve(steps=10)

    assert isinstance(result, TDVPResult)
    assert max(abs(value - 1.0) for value in result.norm_history) < 1e-12
    assert max(
        abs(value - result.energy_history[0]) for value in result.energy_history
    ) < 1e-11
    # A full symmetric step ends with the orthogonality center at site zero.
    assert canonical.canonical_residual(result.mps, center=0) < 1e-11


def test_one_site_tdvp_matches_n8_exact_time_evolution_and_local_observable():
    n = 8
    product, initial = _full_bond_neel(n)
    mpo = MPO.from_bonds(n, 2, dtype=DTYPE).generate_heisenberg()
    sz = spin_operators(dtype=DTYPE)["Sz"]
    result = TDVP(
        initial,
        mpo,
        dt=0.01,
        method="one_site",
        krylov_dim=24,
    ).evolve(
        steps=5,
        observables={
            "sz_mid": lambda state: contractions.native_local_expect(state, sz, n // 2)
            / contractions.native_norm_sq(state)
        },
    )

    hamiltonian = heisenberg_dense(n, dtype=DTYPE)
    exact = tc.matrix_exp(-1j * result.times[-1] * hamiltonian) @ product.to_dense()
    got = result.mps.to_dense()
    got = got / tc.linalg.vector_norm(got)
    fidelity = abs(tc.vdot(exact, got)) ** 2
    from latticetn.observables import dense_expect_local

    reference_sz = dense_expect_local(exact, sz, n // 2, n).real
    assert float(fidelity) > 1.0 - 1e-11
    assert abs(result.observables_history["sz_mid"][-1] - float(reference_sz)) < 1e-11
    assert max(abs(value - 1.0) for value in result.norm_history) < 1e-12


def test_one_site_tdvp_keeps_bond_dimensions_fixed():
    n = 6
    _, initial = _full_bond_neel(n)
    before = [tuple(tensor.shape) for tensor in initial.tensors]
    mpo = MPO.from_bonds(n, 2, dtype=DTYPE).generate_heisenberg()
    result = TDVP(initial, mpo, dt=0.02, method="one_site").evolve(steps=2)
    after = [tuple(tensor.shape) for tensor in result.mps.tensors]
    assert after == before
