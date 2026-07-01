# Review checklist for latticeTN autonomous run

Use this checklist before accepting Claude Code's final changes.

## Physics

- [ ] Heisenberg convention is `S = sigma / 2`, not Pauli-only.
- [ ] Open boundary condition is used in the default validation.
- [ ] N=2 Heisenberg eigenvalues are `[-0.75, 0.25, 0.25, 0.25]`.
- [ ] TFI convention is documented and consistent between dense and MPO forms.
- [ ] No sign flip is hidden in the MPO construction.

## Tensor conventions

- [ ] MPO tensor order is documented, preferably `[left_bond, right_bond, physical_in, physical_out]`.
- [ ] MPS tensor order is documented, preferably `[left_bond, physical, right_bond]`.
- [ ] `mpo_to_dense_matrix` is tested on N=2,3,4.
- [ ] `mps_to_dense_state` is tested on random small MPS.

## Autograd

- [ ] Differentiable energy path uses Rayleigh quotient.
- [ ] No `.detach()`, `.data`, `.item()`, or `torch.no_grad()` is inside differentiable energy computation.
- [ ] `energy.real.backward()` gives gradients on all trainable MPS tensors.

## Numerical validation

- [ ] Random MPS energy matches dense-state energy.
- [ ] Heisenberg variational energy decreases.
- [ ] Variational energy is not below exact ground energy beyond tolerance.
- [ ] Numerical report includes commands, values, errors, and limitations.

## Engineering

- [ ] Tests are real pytest tests with assertions.
- [ ] Fast validation is CPU-only.
- [ ] No long training is hidden inside pytest.
- [ ] Dependencies are minimal and listed.
- [ ] Legacy demos are not broken without documentation.
