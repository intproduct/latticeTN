# Physics conventions

This document fixes the operator and basis conventions used by the Stage 11
physics audit. Tests must compare against independent dense references with the
same conventions, not against another call to the same MPO builder.

## Spin-1/2 chains

Local basis:

| index | state |
|---|---|
| 0 | spin up |
| 1 | spin down |

Spin operators use the spin convention:

```text
Sx = sigma_x / 2
Sy = sigma_y / 2
Sz = sigma_z / 2
S+ = Sx + i Sy
S- = Sx - i Sy
```

The default Heisenberg model is the open-boundary antiferromagnetic chain:

```text
H = J * sum_{i=0}^{N-2} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
J = 1.0
boundary = open
```

The transverse-field Ising model currently implemented by `MPO.generate_tfi`
and `build_mpo(ModelSpec(name="tfi", ...))` is also in the spin convention:

```text
H = -J * sum_{i=0}^{N-2} Sz_i Sz_{i+1} - h * sum_{i=0}^{N-1} Sx_i
boundary = open
```

This is not the Pauli convention. Energies and critical-field values differ by
normalization factors from a `sigma_z sigma_z + sigma_x` convention.

## Spinless fermion t-V chain

Local basis:

| index | state |
|---|---|
| 0 | empty, `|0>` |
| 1 | occupied, `|1>` |

Operators:

```text
c |1> = |0>
cdag |0> = |1>
n = cdag c
F = (-1)^n
```

The implemented open-boundary Hamiltonian is:

```text
H = -t * sum_{i=0}^{N-2} (cdag_i c_{i+1} + cdag_{i+1} c_i)
    + V * sum_{i=0}^{N-2} (n_i - 1/2)(n_{i+1} - 1/2)
    - mu * sum_{i=0}^{N-1} (n_i - 1/2)
```

Global fermion operators use the Jordan-Wigner string:

```text
c_i    = F_0 ... F_{i-1} c_i
cdag_i = F_0 ... F_{i-1} cdag_i
```

The density and chemical-potential terms are diagonal and carry no explicit JW
string.

## Hubbard chain

Local basis:

| index | state |
|---|---|
| 0 | `|0>` |
| 1 | `|up>` |
| 2 | `|down>` |
| 3 | `|up down>` |

The on-site mode order is up then down. The global mode ordering is site-major:

```text
(0_up, 0_down, 1_up, 1_down, ..., (N-1)_up, (N-1)_down)
```

The down operator includes the on-site up-mode Jordan-Wigner parity, so:

```text
c_up |up> = |0>
c_up |up down> = |down>
c_down |down> = |0>
c_down |up down> = -|up>
```

The implemented open-boundary Hamiltonian is:

```text
H = -t * sum_{i=0}^{N-2},sigma (cdag_{i,sigma} c_{i+1,sigma}
                                + cdag_{i+1,sigma} c_{i,sigma})
    + U * sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2)
    - mu * sum_i (n_{i,up} + n_{i,down} - 1)
    - h * sum_i (n_{i,up} - n_{i,down})
```

The sign of `mu` is therefore negative in the Hamiltonian, and the magnetic
field couples to `n_up - n_down` with a negative sign.

## Validation scope

Small-system dense references may use explicit Kronecker products, explicit
Jordan-Wigner operators, or exact diagonalization. Large-system benchmarks must
not construct dense Hamiltonians or call ED, classical DMRG, or Lanczos from the
large-N AD runner.
