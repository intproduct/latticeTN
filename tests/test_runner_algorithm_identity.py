import math

import pytest
import torch as tc

from latticetn.model_registry import build_model_spec
from latticetn.runner import namespace_from_legacy_ad_args, run_latticetn_job
from scripts.run_ad_model_benchmark import build_parser


def test_ad_two_site_dispatch_calls_two_site_train(monkeypatch):
    import latticetn.runner as runner

    called = {"two_site": 0}

    def fake_train(mps, mpo, **kwargs):
        called["two_site"] += 1
        return {
            "sweeps": [{"sweep": 0, "direction": "right", "energy_after": -1.0, "max_bond": 2, "max_trunc": 0.0}],
            "initial_energy": 0.0,
            "final_energy": -1.0,
            "final_bond_dims": [2, 2, 2],
            "max_bond": 2,
            "energy_history": [0.0, -1.0],
            "best_energy": -1.0,
            "best_step": 1,
            "final_state_source": "best_mps",
            "optimizer_steps": 1,
            "closure_evals": 1,
        }

    monkeypatch.setattr(runner, "train_ad_two_site", fake_train)
    result = run_latticetn_job(
        build_model_spec("heisenberg", N=4),
        {"name": "ad_two_site", "chi": 2, "sweeps": 1, "optimizer": "adam", "local_steps": 1},
        {"device": "cpu", "dtype": "complex128", "no_ed": True},
        {"names": ["energy"]},
    )
    assert called["two_site"] == 1
    assert result["diagnostics"]["algorithm_id"] == "ad_two_site"
    assert result["diagnostics"]["optimizer_path"] == "two_site_ad_local_theta"


def test_ad_global_does_not_call_two_site_train(monkeypatch):
    import latticetn.runner as runner

    def fail_train(*args, **kwargs):
        raise AssertionError("ad_global must not call train_ad_two_site")

    monkeypatch.setattr(runner, "train_ad_two_site", fail_train)
    result = run_latticetn_job(
        build_model_spec("heisenberg", N=4),
        {"name": "ad_global", "chi": 2, "sweeps": 1, "optimizer": "adam", "local_steps": 1, "lr": 0.01},
        {"device": "cpu", "dtype": "complex128", "no_ed": True, "seed": 0},
        {"names": ["energy"]},
    )
    assert result["diagnostics"]["algorithm_id"] == "ad_global"
    assert result["summary"]["optimizer_steps"] == 1


def test_hard_sector_ad_two_site_fails_loudly():
    spec = build_model_spec("spinless_tv", N=4, sector={"mode": "hard", "target_n": 2})
    with pytest.raises(ValueError, match="ad_two_site does not support sector_mode='hard'"):
        run_latticetn_job(
            spec,
            {"name": "ad_two_site", "chi": 2, "sweeps": 1, "sector_mode": "hard"},
            {"device": "cpu", "dtype": "complex128", "no_ed": True},
            {"names": ["energy"]},
        )


def test_hard_sector_dispatch_uses_global_mask_path():
    spec = build_model_spec("spinless_tv", N=4, sector={"mode": "hard", "target_n": 2})
    result = run_latticetn_job(
        spec,
        {"name": "ad_global", "chi": 2, "sweeps": 1, "optimizer": "adam", "local_steps": 1, "sector_mode": "hard"},
        {"device": "cpu", "dtype": "complex128", "no_ed": True, "seed": 0},
        {"names": ["energy", "sector"]},
    )
    assert result["diagnostics"]["optimizer_path"] == "global_ad_hard_charge_mask"
    assert result["diagnostics"]["max_forbidden_abs"] == 0.0
    assert result["observables"]["sector"]["abs_error"] < 1e-10


def test_legacy_cli_defaults_and_options_propagate():
    args = build_parser().parse_args([
        "--model", "heisenberg",
        "--N", "4",
        "--chi", "8",
        "--method", "auto",
        "--init", "random",
        "--stabilization", "none",
        "--grad-clip", "0.5",
        "--lbfgs-tolerance-grad", "1e-12",
        "--lbfgs-tolerance-change", "1e-15",
        "--no-ed",
    ])
    _model, method, _runtime, _obs = namespace_from_legacy_ad_args(args)
    assert method.name == "ad_two_site"
    assert method.initialization == "random"
    assert method.projection == "none"
    assert method.post_step_stabilization == "none"
    assert method.grad_clip == 0.5
    assert method.lbfgs_tolerance_grad == 1e-12
    assert method.lbfgs_tolerance_change == 1e-15


def test_global_adam_optimizer_constructed_once(monkeypatch):
    import latticetn.runner as runner

    real_adam = tc.optim.Adam
    created = {"count": 0}

    class SpyAdam(real_adam):
        def __init__(self, *args, **kwargs):
            created["count"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(runner.tc.optim, "Adam", SpyAdam)
    result = run_latticetn_job(
        build_model_spec("heisenberg", N=4),
        {"name": "ad_global", "chi": 2, "sweeps": 3, "optimizer": "adam", "local_steps": 1, "lr": 0.01},
        {"device": "cpu", "dtype": "complex128", "no_ed": True, "seed": 0},
        {"names": ["energy"]},
    )
    assert created["count"] == 1
    assert result["summary"]["global_steps"] == 3


@pytest.mark.parametrize("projection", ["none", "tensor_norm", "canonical"])
def test_global_ad_honors_projection_and_records_conditioning(monkeypatch, projection):
    import latticetn.runner as runner

    seen = []
    real_project = runner._project

    def spy_project(mps, mode):
        seen.append(mode)
        return real_project(mps, mode)

    monkeypatch.setattr(runner, "_project", spy_project)
    result = run_latticetn_job(
        build_model_spec("heisenberg", N=4),
        {
            "name": "ad_global",
            "chi": 4,
            "sweeps": 1,
            "optimizer": "adam",
            "local_steps": 2,
            "lr": 0.01,
            "projection": projection,
            "initialization": "random",
        },
        {"device": "cpu", "dtype": "complex128", "no_ed": True, "seed": 0},
        {"names": ["energy", "gradient_norm", "bond_dims"]},
    )
    assert seen == [projection, projection]
    assert result["diagnostics"]["algorithm_id"] == "ad_global"
    assert result["diagnostics"]["projection"] == projection
    assert result["summary"]["global_steps"] == 2
    assert result["summary"]["optimizer_steps"] == 2
    assert result["summary"]["closure_evals"] == 2
    assert math.isfinite(result["summary"]["final_energy"])
    assert math.isfinite(result["summary"]["best_energy"])
    assert math.isfinite(result["summary"]["final_gradient_norm"])
    assert result["summary"]["best_energy"] <= result["summary"]["initial_energy"] + 1e-10
    for rec in result["sweep_history"]:
        assert rec["bond_dims"] == result["summary"]["final_bond_dims"]
        assert math.isfinite(rec["gradient_norm"])
        assert math.isfinite(rec["state_norm"])
        assert rec["state_norm"] > 0.0
