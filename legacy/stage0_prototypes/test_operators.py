import pytest
import torch as tc
from operators import Operators

class TestOperators:
    
    @pytest.fixture
    def setup(self):
        device = tc.device("cpu")
        tensor = tc.tensor([1.0], device=device)
        return device, tensor
    
    def test_get_single_ope_Sz(self, setup):
        device, tensor = setup
        op = Operators("Sz", device, tensor)
        result = op.get_single_ope()
        expected = tc.tensor([[1, 0], [0, -1]], device=device, dtype=tensor.dtype) / 2
        assert tc.allclose(result, expected)
    
    def test_get_single_ope_invalid_name(self, setup):
        device, tensor = setup
        op = Operators("Invalid", device, tensor)
        with pytest.raises(NotImplementedError):
            op.get_single_ope()