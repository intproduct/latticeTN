"""Red-line regressions for the Stage 12A-P0 scientific compliance repair."""

from __future__ import annotations

import pytest
import torch as tc

from latticetn import contractions as K
from latticetn.charge_sectors import (
    hubbard_hard_sector_product_mps,
    sector_normalize_center,
)
from latticetn.config_schema import MethodConfig
from latticetn.model_registry import build_model_spec
from latticetn.mps import MPS
from latticetn.numerics import (
    nonnegative_if_roundoff,
    real_if_hermitian,
    truncation_error,
)
from latticetn.operators import spin_operators
from latticetn.runner import _sector_penalty, run_latticetn_job


DTYPE = tc.complex128


def _sector_dims(camps):
    return [
        {charge: dim for charge, dim in zip(bond.charges, bond.dims)}
        for bond in camps.bond_charges
    ]


def test_hard_sector_chi_family_is_nested_connected_and_canonicalizable():
    previous = None
    for chi in (32, 48, 60, 64, 80, 96):
        camps = hubbard_hard_sector_product_mps(
            N=20,
            target_nup=10,
            target_ndown=10,
            chi=chi,
            dtype=DTYPE,
        )
        dims = _sector_dims(camps)
        assert camps.bond_charges[10].bond_dim == chi
        assert max(bond.bond_dim for bond in camps.bond_charges) <= chi
        assert any(dim > 1 for bond in camps.bond_charges for dim in bond.dims)
        normalized = sector_normalize_center(camps)
        assert abs(float(normalized.mps.overlap(normalized.mps).real) - 1.0) < 1e-10
        if previous is not None:
            for old_bond, new_bond in zip(previous, dims):
                for charge, old_dim in old_bond.items():
                    assert new_bond.get(charge, 0) >= old_dim
        previous = dims


def test_soft_sector_penalty_is_expectation_of_operator_square():
    # (|00> + |11>)/sqrt(2): <N>=1 but Var(N)=1.
    root2 = 2.0 ** -0.5
    tensors = [
        tc.tensor([[[root2, 0.0], [0.0, root2]]], dtype=DTYPE),
        tc.tensor([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=DTYPE),
    ]
    mps = MPS.from_tensors(tensors, dtype=DTYPE)
    model = build_model_spec(
        "spinless_tv",
        N=2,
        sector={"mode": "soft", "target_n": 1, "lambda_n": 3.0},
    )
    method = MethodConfig(
        name="ad_global",
        chi=2,
        sweeps=1,
        sector_mode="soft",
    )
    assert tc.allclose(
        _sector_penalty(model, method, mps),
        tc.tensor(3.0, dtype=tc.float64),
    )


def test_global_ad_auto_uses_requested_bond_manifold():
    result = run_latticetn_job(
        build_model_spec("heisenberg", N=4),
        {
            "name": "ad_global",
            "chi": 4,
            "sweeps": 1,
            "local_steps": 1,
            "optimizer": "adam",
            "lr": 0.001,
        },
        {"device": "cpu", "dtype": "complex128", "seed": 7, "no_ed": True},
        {"names": ["energy", "bond_dims"]},
    )
    assert result["diagnostics"]["initialization"] == "random"
    assert result["summary"]["initial_max_bond"] == 4
    assert result["summary"]["initial_bond_dims"] == [2, 4, 2]


def test_mps_geometry_uses_local_dimension_and_rng_is_not_repeated():
    tc.manual_seed(123)
    first = MPS(4, dim=4, chi=64, dtype=DTYPE)
    second = MPS(4, dim=4, chi=64, dtype=DTYPE)
    assert [tensor.shape[2] for tensor in first.tensors[:-1]] == [4, 16, 4]
    assert not tc.equal(first.tensors[0], second.tensors[0])
    tc.manual_seed(123)
    replay = MPS(4, dim=4, chi=64, dtype=DTYPE)
    assert tc.equal(first.tensors[0], replay.tensors[0])


def test_native_expectations_are_scale_invariant():
    tc.manual_seed(4)
    mps = MPS(4, dim=2, chi=4, dtype=DTYPE)
    scaled = MPS.from_tensors(
        [7.0 * mps.tensors[0], *mps.tensors[1:]],
        dtype=DTYPE,
    )
    sz = spin_operators(dtype=DTYPE)["Sz"]
    assert tc.allclose(
        K.native_local_expect(mps, sz, 1),
        K.native_local_expect(scaled, sz, 1),
        atol=1e-12,
        rtol=1e-12,
    )
    assert tc.allclose(
        K.native_two_site_expect(mps, sz, 0, sz, 3),
        K.native_two_site_expect(scaled, sz, 0, sz, 3),
        atol=1e-12,
        rtol=1e-12,
    )


def test_reported_best_energy_and_final_observables_share_best_mps():
    result = run_latticetn_job(
        build_model_spec("heisenberg", N=4),
        {
            "name": "ad_global",
            "chi": 4,
            "sweeps": 2,
            "local_steps": 1,
            "optimizer": "adam",
            "lr": 0.05,
        },
        {"device": "cpu", "dtype": "complex128", "seed": 5, "no_ed": True},
        {"names": ["energy"]},
    )
    summary = result["summary"]
    assert summary["final_state_source"] == "best_mps"
    assert summary["best_energy"] == pytest.approx(summary["final_energy"], abs=1e-12)
    assert result["observables"]["energy"] == pytest.approx(
        summary["best_energy"], abs=1e-12
    )


def test_significant_numeric_failures_are_not_cleaned():
    with pytest.raises(FloatingPointError, match="non-finite"):
        truncation_error(
            tc.tensor([1.0, float("nan")], dtype=tc.float64),
            1,
            name="test",
        )
    with pytest.raises(FloatingPointError, match="significantly negative"):
        nonnegative_if_roundoff(
            tc.tensor(-1e-3, dtype=tc.float64),
            name="test variance",
        )
    with pytest.raises(FloatingPointError, match="imaginary"):
        real_if_hermitian(
            tc.tensor(1.0 + 1e-2j, dtype=DTYPE),
            name="test energy",
        )
