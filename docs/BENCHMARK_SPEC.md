# Stage 2 Benchmark Specification

## Goal

Upgrade latticeTN from a small-system validation prototype into a reproducible MPS benchmark tool for the finite open-boundary 1D spin-1/2 antiferromagnetic Heisenberg chain.

Stage 1 proved that MPS + MPO + PyTorch autograd can solve a small Heisenberg chain and agree with exact diagonalization. Stage 2 should prove that the library can produce standard benchmark observables and convergence evidence.

## Hamiltonian convention

Use the same convention as Stage 1:

```text
H = J * sum_{i=1}^{N-1} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
S = sigma / 2
J = 1.0
boundary = open
```

Do **not** switch to Pauli convention. Do **not** silently change the sign of J or the boundary condition.

## Reference values

Small systems use exact diagonalization as the golden reference.

For large-system trend checking only, the infinite-chain energy density is:

```text
e0 = 1/4 - ln(2) ~= -0.4431471805599453
```

For open finite chains, prefer reporting `E / (N - 1)` as energy per bond. The thermodynamic value is a trend reference, not a strict finite-N target.

## Required observables

Implement or expose these APIs in a way that can be tested on small systems:

1. total energy
2. energy per bond
3. local magnetization `<Sz_i>`
4. nearest-neighbor bond energy `<S_i · S_{i+1}>`
5. two-point correlation `<Sz_i Sz_j>`
6. bipartite entanglement entropy across a cut

Suggested module:

```text
latticetn/observables.py
```

Suggested functions:

```python
dense_expect_local(state, op, site, N)
dense_expect_two_site(state, op1, i, op2, j, N)
dense_bond_energy_heisenberg(state, i, N)
dense_entanglement_entropy(state, cut, N)

mps_expect_local(mps, op, site)
mps_expect_two_site(mps, op1, i, op2, j)
mps_bond_energy_heisenberg(mps, i)
mps_entanglement_entropy(mps, cut)
```

It is acceptable in Stage 2 for MPS observables to use `mps.to_dense()` internally for small-system validation. A later Stage 3 can replace these with canonical-MPS contractions.

## Accuracy targets

Small exact systems:

- `N = 4, 6` must be tested.
- `N = 8` should be included where CPU runtime is acceptable.
- MPS observable values must match dense-state observable values within `1e-8` to `1e-6`.
- Variational energy must not go below exact ground energy beyond tolerance.
- Energy error should generally improve or not materially worsen with larger chi under the same seed/settings.

## Runtime policy

Fast benchmark must be CPU-only and suitable for frequent local validation.

Do not run GPU jobs in pytest.

Do not run long training inside pytest.

Full benchmark may be optional and slower, but it must be explicit.
