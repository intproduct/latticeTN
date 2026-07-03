import torch as tc

from latticetn.charge_sectors import (
    spinless_tensor_charge_mask,
    hubbard_tensor_charge_mask,
    zero_forbidden_gradients_,
    apply_charge_masks_,
    max_forbidden_abs,
)


def test_spinless_tensor_charge_mask_entries_and_device():
    mask = spinless_tensor_charge_mask([0, 1], [1, 2], device="cpu")
    assert mask.dtype == tc.bool
    assert mask.device.type == "cpu"
    assert mask.shape == (2, 2, 2)
    assert bool(mask[0, 1, 0])
    assert bool(mask[1, 1, 1])
    assert not bool(mask[0, 0, 0])
    assert not bool(mask[1, 0, 1])


def test_hubbard_tensor_charge_mask_entries():
    mask = hubbard_tensor_charge_mask([(0, 0), (1, 0)], [(1, 0), (1, 1)])
    assert mask.shape == (2, 4, 2)
    assert bool(mask[0, 1, 0])  # |up>: (0,0) -> (1,0)
    assert bool(mask[0, 3, 1])  # |updown>: (0,0) -> (1,1)
    assert bool(mask[1, 2, 1])  # |down>: (1,0) -> (1,1)
    assert not bool(mask[1, 1, 1])


def test_apply_masks_and_zero_forbidden_gradients():
    mask = spinless_tensor_charge_mask([0], [0, 1])
    tensor = tc.nn.Parameter(tc.ones(mask.shape, dtype=tc.complex128))
    apply_charge_masks_([tensor], [mask])
    assert max_forbidden_abs([tensor], [mask]) == 0.0
    tensor.grad = tc.ones_like(tensor)
    max_grad = zero_forbidden_gradients_([tensor], [mask])
    assert max_grad == 1.0
    assert tc.all(tensor.grad[~mask] == 0)
