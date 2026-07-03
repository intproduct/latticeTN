import pytest

from latticetn.model_spec import ModelSpec, TermSpec, OperatorRef


def test_model_spec_roundtrip_from_dict_to_dict():
    data = {
        "name": "heisenberg",
        "N": 4,
        "local_basis": "spin_half",
        "boundary": "obc",
        "parameters": {"J": 1.0},
        "terms": [
            {
                "coefficient": "J",
                "operators": [{"op": "Sx", "site": "i"}, {"op": "Sx", "site": "i+1"}],
                "pattern": "nearest_neighbor",
                "plus_hc": False,
            }
        ],
        "sector": {"mode": "none"},
    }
    spec = ModelSpec.from_dict(data)
    assert spec.to_dict()["name"] == "heisenberg"
    assert spec.to_dict()["terms"][0]["operators"][0]["op"] == "Sx"
    assert ModelSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_invalid_term_pattern_rejected():
    with pytest.raises(ValueError, match="unsupported term pattern"):
        ModelSpec(
            name="custom",
            N=4,
            local_basis="spin_half",
            terms=[TermSpec(1.0, [OperatorRef("Sx", "i")], "long_range_magic")],
        ).to_dict() | {}
        ModelSpec.from_dict({
            "name": "custom",
            "N": 4,
            "local_basis": "spin_half",
            "terms": [{"coefficient": 1.0, "operators": [], "pattern": "long_range_magic"}],
        })
