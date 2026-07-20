"""Stage 12B-2 two-site TDVP and adaptive-chi validation."""

import torch as tc

from latticetn import canonical, contractions
from latticetn.initial_states import neel_spin_state
from latticetn.mpo import MPO
from latticetn.operators import heisenberg_dense, spin_operators
from latticetn.tdvp import TDVP


DTYPE = tc.complex128


def test_two_site_tdvp_expands_product_state_bonds_and_reports_truncation():
    n = 8
    initial = neel_spin_state(n, dtype=DTYPE)
    mpo = MPO.from_bonds(n, 2, dtype=DTYPE).generate_heisenberg()
    result = TDVP(
        initial,
        mpo,
        dt=0.02,
        method="two_site",
        max_bond_dim=8,
        truncation_tol=1e-12,
    ).evolve(steps=2)

    assert max(tensor.shape[2] for tensor in result.mps.tensors[:-1]) > 1
    assert len(result.truncation_history) == 2
    assert result.truncation_history[-1]["max_bond"] <= 8
    assert all(
        0.0 <= update["truncation_error"] <= 1.0
        for step in result.truncation_history
        for update in step["updates"]
    )


def test_two_site_tdvp_full_chi_matches_n8_exact_evolution():
    n = 8
    initial = neel_spin_state(n, dtype=DTYPE)
    mpo = MPO.from_bonds(n, 2, dtype=DTYPE).generate_heisenberg()
    result = TDVP(
        initial,
        mpo,
        dt=0.01,
        method="two_site",
        max_bond_dim=16,
        truncation_tol=0.0,
        krylov_dim=24,
    ).evolve(steps=5)

    exact = tc.matrix_exp(-1j * result.times[-1] * heisenberg_dense(n, dtype=DTYPE)) \
        @ initial.to_dense()
    evolved = result.mps.to_dense()
    evolved = evolved / tc.linalg.vector_norm(evolved)
    fidelity = abs(tc.vdot(exact, evolved)) ** 2

    assert float(fidelity) > 1.0 - 1e-11
    assert max(abs(value - 1.0) for value in result.norm_history) < 1e-12
    assert max(
        abs(value - result.energy_history[0]) for value in result.energy_history
    ) < 1e-11
    assert result.truncation_history[-1]["bond_dims"] == [2, 4, 8, 16, 8, 4, 2]


def test_adaptive_chi_heisenberg_quench_is_physical_and_stable():
    n = 8
    initial = neel_spin_state(n, dtype=DTYPE)
    mpo = MPO.from_bonds(n, 2, dtype=DTYPE).generate_heisenberg()
    sz = spin_operators(dtype=DTYPE)["Sz"]
    result = TDVP(
        initial,
        mpo,
        dt=0.02,
        method="two_site",
        max_bond_dim=4,
        truncation_tol=1e-10,
        krylov_dim=20,
    ).evolve(
        steps=10,
        observables={
            "sz_mid": lambda state: contractions.native_local_expect(state, sz, n // 2)
            / contractions.native_norm_sq(state),
            "entropy_mid": lambda state: canonical.entanglement_entropy(state, n // 2),
        },
    )

    assert max(abs(value - 1.0) for value in result.norm_history) < 1e-12
    assert max(
        abs(value - result.energy_history[0]) for value in result.energy_history
    ) < 1e-8
    assert result.observables_history["entropy_mid"][-1] > 0.05
    assert result.observables_history["sz_mid"][-1] < 0.49
    assert max(step["max_bond"] for step in result.truncation_history) == 4
    assert max(step["max_truncation"] for step in result.truncation_history) <= 1.1e-10
