import json

import pytest

from latticetn.runner import run_latticetn_job


def _spinless_method(projection, **extra):
    data = {
        "name": "ad_global",
        "chi": 4,
        "sweeps": 1,
        "optimizer": "adam",
        "local_steps": 1,
        "global_steps": 3,
        "lr": 0.01,
        "sector_mode": "hard",
        "initialization": "spinless_cdw",
        "projection": projection,
    }
    data.update(extra)
    return data


def _run(method, output=None):
    return run_latticetn_job(
        {
            "name": "spinless_tv",
            "N": 4,
            "boundary": "obc",
            "parameters": {"t": 1.0, "V": 0.0, "mu": 0.0},
            "sector": {"mode": "hard", "target_n": 2},
        },
        method,
        {"device": "cpu", "dtype": "complex128", "seed": 12,
         "no_ed": True, "output": str(output) if output else None},
        {"names": ["energy", "sector", "local_density_mid"]},
    )


def test_sector_canonical_interval_reset_and_final_metadata(tmp_path):
    path = tmp_path / "result.json"
    result = _run(_spinless_method(
        "sector_canonical",
        canonical_interval=2,
        reset_optimizer_on_canonicalize=True,
        canonicalization_method="qr",
    ), output=path)
    diag = result["diagnostics"]
    summary = result["summary"]
    assert [x["step"] for x in diag["projection_events"]] == [2]
    assert diag["optimizer_reset_events"] == [2]
    assert diag["projection"] == "sector_canonical"
    assert diag["canonical_interval"] == 2
    assert diag["canonicalization_method"] == "qr"
    assert abs(summary["physical_norm_after_projection"] - 1.0) < 1e-12
    assert summary["canonical_residual"] < 1e-12
    assert diag["max_forbidden_abs"] == 0.0
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["summary"]["raw_norm_before_projection"] == summary["raw_norm_before_projection"]
    assert saved["summary"]["physical_norm_after_projection"] == summary["physical_norm_after_projection"]


def test_tensor_norm_and_sector_canonical_are_distinct():
    tensor = _run(_spinless_method("tensor_norm"))
    sector = _run(_spinless_method("sector_canonical", canonical_interval=1))
    assert tensor["diagnostics"]["projection"] == "tensor_norm"
    assert tensor["diagnostics"]["projection_events"] == []
    assert sector["diagnostics"]["projection"] == "sector_canonical"
    assert len(sector["diagnostics"]["projection_events"]) == 3
    # Accepted physical outputs are normalized regardless of internal mode.
    assert abs(tensor["summary"]["physical_norm_after_projection"] - 1.0) < 1e-12
    assert abs(sector["summary"]["physical_norm_after_projection"] - 1.0) < 1e-12


def test_optimizer_reset_can_be_disabled_explicitly():
    result = _run(_spinless_method(
        "sector_canonical",
        canonical_interval=1,
        reset_optimizer_on_canonicalize=False,
    ))
    assert result["diagnostics"]["optimizer_reset"] == "never"
    assert result["diagnostics"]["optimizer_reset_events"] == []
    assert len(result["diagnostics"]["projection_events"]) == 3


def test_dense_canonical_projection_is_rejected_for_hard_sector():
    with pytest.raises(ValueError, match="sector_canonical"):
        _run(_spinless_method("canonical"))
