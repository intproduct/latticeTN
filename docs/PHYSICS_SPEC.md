# Physics specification

## Primary validation model

The primary target is the finite open-boundary spin-1/2 antiferromagnetic Heisenberg chain:

```text
H = J * sum_{i=1}^{N-1} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
```

Default convention:

```text
Sx = sigma_x / 2
Sy = sigma_y / 2
Sz = sigma_z / 2
J = 1.0
boundary = open
dtype = torch.complex128
device = cpu for tests
```

Important: this is the spin-operator convention, not the Pauli convention. If the Pauli convention is used, energies differ by a factor of 4.

## Required exact checks

For N=2 open Heisenberg chain:

```text
eigenvalues = [-3/4, 1/4, 1/4, 1/4]
ground_energy = -0.75
```

For all dense Hamiltonians used as references:

```text
shape = (2**N, 2**N)
H == H.conj().T within tolerance
lowest eigenvalue is real within tolerance
```

## Secondary model

The transverse-field Ising model may be used as a regression target because the repo already contains TFI-related code.

Default convention for tests should be explicitly documented in code comments. Do not mix these conventions silently:

```text
H = -J * sum sigma_z_i sigma_z_{i+1} - h * sum sigma_x_i
```

## Numerical tolerances

Default deterministic comparison tolerances:

```text
MPO dense matrix max_abs_diff <= 1e-10
MPS dense energy abs_diff <= 1e-8
Hermiticity max_abs_diff <= 1e-12
N=2 exact Heisenberg eigenvalues abs_diff <= 1e-12
```

Variational optimization tolerance is less strict because it depends on initialization and optimizer:

```text
N=4, chi>=4: final_energy - exact_energy <= 1e-4 preferred
N=6, chi>=8: final_energy - exact_energy <= 1e-3 acceptable for smoke validation
variational energy must not be below exact_energy by more than 1e-8
```
