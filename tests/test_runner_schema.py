import json

import pytest
import torch as tc

from latticetn.model_registry import build_model_spec
from latticetn.runner import run_latticetn_job


def test_run_latticetn_job_heisenberg_ad_cpu_json_serializable():
    result = run_latticetn_job(
        build_model_spec("heisenberg", N=4),
        {"name": "ad_dmrg", "chi": 2, "sweeps": 1, "optimizer": "adam", "local_steps": 1, "lr": 0.01},
        {"device": "cpu", "dtype": "complex128", "no_ed": True},
        {"names": ["energy", "energy_per_site", "bond_dims", "runtime"]},
    )
    json.dumps(result)
    assert result["diagnostics"]["ad_used"] is True
    assert result["diagnostics"]["ed_used"] is False
    assert result["diagnostics"]["classical_dmrg_used"] is False
    assert result["diagnostics"]["lanczos_used"] is False
    assert "final_energy" in result["summary"]


def test_run_latticetn_job_spinless_hard_ad_cpu():
    spec = build_model_spec("spinless_tv", N=4, sector={"mode": "hard", "target_n": 2})
    result = run_latticetn_job(
        spec,
        {"name": "ad_dmrg", "chi": 4, "sweeps": 1, "optimizer": "adam", "local_steps": 1, "lr": 0.01, "sector_mode": "hard"},
        {"device": "cpu", "dtype": "complex128", "no_ed": True},
        {"names": ["energy", "sector", "bond_dims"]},
    )
    assert result["observables"]["sector"]["abs_error"] < 1e-10
    assert result["diagnostics"]["max_forbidden_abs"] == 0.0


def test_run_latticetn_job_hubbard_hard_ad_cpu():
    spec = build_model_spec("hubbard", N=4, sector={"mode": "hard", "target_nup": 2, "target_ndown": 2})
    result = run_latticetn_job(
        spec,
        {"name": "ad_dmrg", "chi": 4, "sweeps": 1, "optimizer": "adam", "local_steps": 1, "lr": 0.01, "sector_mode": "hard"},
        {"device": "cpu", "dtype": "complex128", "no_ed": True},
        {"names": ["energy", "sector", "bond_dims"]},
    )
    assert result["observables"]["sector"]["n_up_abs_error"] < 1e-10
    assert result["observables"]["sector"]["n_down_abs_error"] < 1e-10
    assert result["diagnostics"]["max_forbidden_abs"] == 0.0


def test_run_latticetn_job_classical_dmrg_flags():
    result = run_latticetn_job(
        build_model_spec("heisenberg", N=4),
        {"name": "dmrg", "chi": 2, "sweeps": 1},
        {"device": "cpu", "dtype": "complex128", "no_ed": True},
        {"names": ["energy", "bond_dims", "truncation"]},
    )
    assert result["diagnostics"]["classical_dmrg_used"] is True
    assert result["diagnostics"]["lanczos_used"] is True
    assert result["diagnostics"]["ad_used"] is False
    assert result["diagnostics"]["ed_used"] is False


@pytest.mark.skipif(not tc.cuda.is_available(), reason="CUDA not available")
def test_run_latticetn_job_cuda_tiny_clean_skip_otherwise():
    result = run_latticetn_job(
        build_model_spec("spinless_tv", N=4, sector={"mode": "hard", "target_n": 2}),
        {"name": "ad_dmrg", "chi": 4, "sweeps": 1, "optimizer": "adam", "local_steps": 1, "lr": 0.01, "sector_mode": "hard"},
        {"device": "cuda", "dtype": "complex64", "no_ed": True},
        {"names": ["energy", "sector"]},
    )
    assert result["runtime"]["resolved_device"] == "cuda"
    assert result["diagnostics"]["max_forbidden_abs"] == 0.0
