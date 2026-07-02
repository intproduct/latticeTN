# 01 — Quickstart: Heisenberg MPO + energy

This tutorial gets you from a clean checkout to a verified differentiable
Heisenberg energy in under five minutes. It is the smallest end-to-end
computation in the library: build an MPS + MPO, evaluate the differentiable
Rayleigh quotient, and compare it to exact diagonalization.

## Goal

- Build a random open-boundary MPS and the Heisenberg MPO.
- Evaluate the differentiable Rayleigh energy `E = <ψ|H|ψ>/<ψ|ψ>`.
- Compare it to the exact ground energy (exact diagonalization).
- Understand, in one sentence, what the **AD mainline** is and what is
  **not** the mainline.

## The one-sentence mainline

> **AD mainline:** `MPS parameters (trainable nn.Parameter) → differentiable
> Rayleigh quotient → loss.backward() → torch optimizer step`.
> **SVD/QR/canonicalization** are post-step stabilization/projection/compression
> — **not the solver**. **DMRG/Lanczos/`eigh`** are classical reference
> baselines — **not the AD mainline**.

This tutorial only *evaluates* the differentiable energy (no training yet);
training comes in tutorials 02–04.

## Prerequisites

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

All default runs are **CPU-only**, `torch.complex128`. See
`docs/USER_GUIDE.md` §1 if `import torch` fails (use a real interpreter, not
the Windows MS Store stub).

## Minimal code

Save as `quickstart.py` and run it:

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.contractions import rayleigh_energy_native
from latticetn.operators import heisenberg_dense, exact_ground_energy

tc.manual_seed(0)
N, chi, dtype = 6, 8, tc.complex128

mps = MPS(N, 2, chi, dtype=dtype)                                   # random MPS
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)

E = rayleigh_energy_native(mps, mpo)                                # differentiable
E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))

print("Rayleigh E =", float(E.real))
print("exact  E0  =", E0)
print("|E - E0|   =", abs(float(E.real) - E0))
```

## Run command

```bash
python quickstart.py
```

## Expected output

The random MPS is **not** the ground state, so `E` will be *above* `E0`
(a variational upper bound). Concretely (seed 0, N=6, chi=8):

```text
Rayleigh E = -0.8...   # a random-MPS energy, well above the ground state
exact  E0  = -2.4935771339...
|E - E0|   = 1.6...
```

The exact ground energy for N=6 open Heisenberg (J=1, S=σ/2) is
`-2.4935771339...`. The important checks:

- `E` is **finite** and **real**;
- `E >= E0 - 1e-8` (a variational energy must never undershoot the exact
  ground beyond tolerance — if it does, something is physically wrong);
- `E` has `requires_grad=True` (it is differentiable — autograd can flow to
  the MPS tensors).

## Common errors

- **`ModuleNotFoundError: No module named 'latticetn'`** — you forgot
  `pip install -e .` from the repo root.
- **`RuntimeError: Expected all tensors to be on the same device`** — you mixed
  a CPU MPS with a GPU MPO (or vice versa). Keep `device` consistent across the
  MPS, MPO, and operators. The quickstart is CPU-only, so leave `device="cpu"`.
- **`E` is complex, not real** — `rayleigh_energy_native` already returns
  `.real`; if you compute your own `<ψ|H|ψ>/<ψ|ψ>`, take the real part (the
  Hamiltonian is Hermitian, so the imaginary part is numerical noise).
- **Energies differ by a factor of 4** — you used Pauli matrices `σ` where the
  convention is spin operators `S = σ/2`. Use `spin_operators(...)`, not
  `pauli_matrices(...)`. See `docs/PHYSICS_SPEC.md`.
- **`E` below `E0`** — a variational energy below the exact ground by more than
  `1e-8` is a **bug, not a win**. Re-check your operator convention and dtype
  (`complex128`).

## Where next

- Train the MPS to the ground state: tutorial **02 — Global AD-MPS**.
- Local sweeps: tutorials **03 (one-site)** and **04 (two-site)**.
- API detail: `docs/API_OVERVIEW.md` (`latticetn.contractions`).
