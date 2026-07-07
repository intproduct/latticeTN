from __future__ import annotations

import torch as tc

from latticetn.benchmarks.exact_reference import (
    dense_model_hamiltonian,
    exact_ground_reference,
    hubbard_sector_indices,
    spinless_sector_indices,
)
from latticetn.config_schema import MethodConfig, ObservableSpec, RuntimeConfig
from latticetn.model_registry import build_model_spec
from latticetn.runner import run_latticetn_job


DTYPE = tc.complex128


def test_fixed_sector_basis_dimensions():
    assert len(spinless_sector_indices(6, 3)) == 20
    assert len(spinless_sector_indices(4, 2)) == 6
    assert len(hubbard_sector_indices(3, 1, 1)) == 9
    assert len(hubbard_sector_indices(4, 2, 2)) == 36


def test_exact_reference_cases_are_hermitian_and_sector_restricted():
    cases = [
        ("heisenberg", 6, {"J": 1.0}, None),
        ("tfi", 6, {"J": 1.0, "h": 0.7}, None),
        ("spinless_tv", 6, {"t": 1.0, "V": 0.5, "mu": 0.0}, {"target_n": 3}),
        ("hubbard", 3, {"t": 1.0, "U": 4.0, "mu": 0.0, "h": 0.0}, {"target_nup": 1, "target_ndown": 1}),
    ]
    for model, N, params, sector in cases:
        H = dense_model_hamiltonian(model, N, params, dtype=DTYPE)
        assert tc.allclose(H, H.conj().T, atol=1e-12), model
        ref = exact_ground_reference(model, N, params, sector, dtype=DTYPE)
        assert ref.dim_full == H.shape[0]
        assert ref.dim_sector <= ref.dim_full
        assert ref.energy_per_site == ref.energy / N
        assert ref.energy == ref.energy


def test_short_ad_variational_bounds_against_small_n_ed():
    cases = [
        ("heisenberg", 4, {"J": 1.0}, None, "none", 4),
        ("tfi", 4, {"J": 1.0, "h": 0.7}, None, "none", 4),
        ("spinless_tv", 4, {"t": 1.0, "V": 0.5, "mu": 0.0}, {"target_n": 2}, "hard", 4),
        ("hubbard", 3, {"t": 1.0, "U": 2.0, "mu": 0.0, "h": 0.0}, {"target_nup": 1, "target_ndown": 1}, "hard", 4),
    ]
    for model, N, params, sector, sector_mode, chi in cases:
        model_spec = build_model_spec(model, N, params, sector=sector)
        method = MethodConfig(
            name="ad_dmrg",
            chi=chi,
            sweeps=1,
            optimizer="adam",
            local_steps=2,
            lr=0.01,
            sector_mode=sector_mode,
        )
        runtime = RuntimeConfig(device="cpu", dtype="complex128", seed=0, no_ed=True)
        result = run_latticetn_job(
            model_spec,
            method,
            runtime,
            ObservableSpec(["energy", "energy_per_site", "sector", "bond_dims", "gradient_norm"]),
        )
        ref = exact_ground_reference(model, N, params, sector, dtype=DTYPE)
        final_energy = float(result["summary"]["final_energy"])
        assert final_energy >= ref.energy - 1e-8, (model, final_energy, ref.energy)
        assert result["diagnostics"]["ad_used"] is True
        assert result["diagnostics"]["ed_used"] is False
        assert result["diagnostics"]["dense_hamiltonian_built"] is False


def test_heisenberg_classical_dmrg_variational_bound_against_ed():
    model_spec = build_model_spec("heisenberg", 4, {"J": 1.0})
    result = run_latticetn_job(
        model_spec,
        MethodConfig(name="dmrg", chi=4, sweeps=1),
        RuntimeConfig(device="cpu", dtype="complex128", seed=0, no_ed=True),
        ObservableSpec(["energy", "energy_per_site", "bond_dims", "runtime"]),
    )
    ref = exact_ground_reference("heisenberg", 4, {"J": 1.0}, dtype=DTYPE)
    final_energy = float(result["summary"]["final_energy"])
    assert final_energy >= ref.energy - 1e-8
    assert result["diagnostics"]["classical_dmrg_used"] is True
    assert result["diagnostics"]["lanczos_used"] is True
    assert result["diagnostics"]["ed_used"] is False
