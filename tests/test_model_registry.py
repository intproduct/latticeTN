import pytest

from latticetn.model_registry import list_model_ids, get_model_schema, build_model_spec


def test_registry_contains_required_presets():
    ids = set(list_model_ids())
    assert {"heisenberg", "tfi", "spinless_tv", "hubbard", "xxz"} <= ids
    schema = get_model_schema("hubbard")
    assert schema["local_basis"] == "hubbard"
    assert "ad_global" in schema["supported_methods"]
    assert "ad_two_site" in schema["supported_methods"]
    assert "hard" in schema["supported_sector_modes"]


def test_build_model_spec_uses_defaults_and_parameters():
    spec = build_model_spec("spinless_tv", N=4, parameters={"V": 0.5}, sector={"mode": "hard", "target_n": 2})
    assert spec.name == "spinless_tv"
    assert spec.parameters["t"] == 1.0
    assert spec.parameters["V"] == 0.5
    assert spec.sector["target_n"] == 2


def test_xxz_registered_but_not_implemented():
    with pytest.raises(NotImplementedError):
        build_model_spec("xxz", N=4)
