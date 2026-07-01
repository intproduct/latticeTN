# Validation protocol

This file defines the autonomous validation target for Claude Code.

## Stage 1: exact references

Create exact dense reference utilities for small finite systems:

- `dense_heisenberg_hamiltonian(N, J=1.0, boundary="open", convention="spin")`
- `dense_tfi_hamiltonian(N, J=1.0, h=1.0, boundary="open")`
- `exact_ground_energy(H)`

Required tests:

```bash
pytest -q tests/test_reference_models.py
```

Pass criteria:

- dense Hamiltonians have correct shape;
- Hamiltonians are Hermitian;
- N=2 Heisenberg eigenvalues are `[-0.75, 0.25, 0.25, 0.25]` under spin convention.

## Stage 2: tensor-to-dense bridges

Implement small-system debugging utilities:

- `mps_to_dense_state(mps)`
- `mpo_to_dense_matrix(mpo)`

These are for tests and debugging only. They may scale exponentially and must not be used for large systems.

Required tests:

```bash
pytest -q tests/test_mps_dense.py tests/test_mpo_dense.py
```

Pass criteria:

- dense state has shape `(2**N,)`;
- MPO dense matrix has shape `(2**N, 2**N)`;
- utilities preserve dtype/device as far as practical;
- tests use N <= 6.

## Stage 3: existing TFI consistency

Use existing TFI code as a regression model.

Required tests:

```bash
pytest -q tests/test_tfi_mpo_dense.py
```

Pass criteria:

- for N=2,3,4 open chains, `mpo_to_dense_matrix(generate_TFI_MPO(...))` matches `dense_tfi_hamiltonian(...)` with max absolute difference <= 1e-10.

## Stage 4: Rayleigh quotient energy path

The differentiable MPS-MPO energy must compute:

```text
E = <psi|H|psi> / <psi|psi>
```

Required tests:

```bash
pytest -q tests/test_energy_rayleigh.py
```

Pass criteria:

- random MPS energy from MPO equals dense-state Rayleigh quotient within 1e-8 for N=2,3,4;
- energy is a scalar tensor;
- `energy.real.backward()` gives non-None gradients for all trainable MPS tensors;
- no `.detach()`, `.item()`, `.data`, or `torch.no_grad()` appears inside differentiable energy calculation code.

## Stage 5: Heisenberg MPO

Implement open-boundary Heisenberg MPO under the spin convention in `docs/PHYSICS_SPEC.md`.

Required tests:

```bash
pytest -q tests/test_heisenberg_mpo_dense.py tests/test_heisenberg_energy_dense_compare.py
```

Pass criteria:

- for N=2,3,4,5, Heisenberg MPO dense matrix matches exact dense Hamiltonian within 1e-10;
- random MPS energy matches dense-state Rayleigh quotient within 1e-8.

## Stage 6: short variational solve

Add a short reproducible script:

```bash
python scripts/run_heisenberg_small.py --N 6 --chi 8 --steps 300 --lr 1e-2 --seed 0 --device cpu
```

Pass criteria:

- output includes exact energy, initial energy, final energy, absolute error, and command-line args;
- final energy is lower than initial energy;
- final energy is not below exact energy by more than 1e-8;
- final energy is within 1e-3 of exact energy for the default smoke target, or the report explains why a stricter threshold is not yet stable.

## Final artifact

Create or update:

```text
docs/NUMERICAL_REPORT.md
```

It must contain:

- physics convention;
- exact diagonalization results;
- variational MPS results;
- commands run;
- pass/fail table;
- known limitations;
- recommended next experiments.
