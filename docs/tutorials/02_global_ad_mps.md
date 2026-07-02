# 02 — Global AD-MPS training

Train **all** MPS site tensors simultaneously on the differentiable Rayleigh
quotient with PyTorch autograd + Adam. This is the Stage 4R AD mainline in its
simplest form: one global loss, one optimizer, every tensor updated each step.

## Goal

- Take the quickstart MPS and actually **minimize** its energy.
- Reach the exact Heisenberg ground energy within tolerance (small N).
- Read the training history (`energy_history`, `grad_norm_history`).
- See why "global" means "all tensors at once" and where it is/isn't efficient.

## The mainline (recap)

```
MPS parameters (trainable nn.Parameter)
   -> differentiable Rayleigh quotient  E = <ψ|H|ψ>/<ψ|ψ>
   -> loss.backward()   (autograd; grads flow to every site tensor)
   -> torch optimizer step (Adam)
   -> [post-step: per-tensor L2 renormalization, OUTSIDE the loss graph]
```

The per-tensor L2 renormalization is **stabilization, not the solver** — it
runs under `no_grad` mutating `.data`, outside the differentiable energy. The
Rayleigh quotient is scale-invariant, so it does not change the physics.

## Minimal code

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps
from latticetn.operators import heisenberg_dense, exact_ground_energy

tc.manual_seed(0)
N, chi, dtype = 6, 8, tc.complex128

mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)
ad = ADVariationalMPS(mps, mpo)                       # all site tensors trainable

res = train_ad_mps(ad, num_steps=300, lr=1e-2, optimizer="adam")

E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))
print("initial E =", res["initial_energy"])
print("final   E =", res["final_energy"])
print("exact  E0 =", E0)
print("abs err   =", abs(res["final_energy"] - E0))
print("below gnd =", res["final_energy"] < E0 - 1e-8)
```

## Run command

```bash
python -c "$(cat above)"          # or save to a file and: python file.py
# The repo also has a ready runner with a JSON/markdown report:
python scripts/run_ad_mps_heisenberg.py --N 6 --chi 8 --steps 300 --print
```

## Expected output

With `N=6, chi=8, steps=300, lr=1e-2, Adam, seed=0`:

```text
initial E = -1.2...                # random-MPS start
final   E = -2.493316...           # close to exact
exact  E0 = -2.4935771339
abs err   = 2.6e-04                # within AD_TOL[6] = 1e-3
below gnd = False
```

For `N=4, chi=8, steps=200`: `final E ≈ -1.6160253893`, `abs err ≈ 1.5e-8`
(tolerance `1e-6`). The energy monotonically decreases toward the exact value
and never crosses below it.

## Common errors

- **Energy not decreasing** — check `optimizer="adam"` and `lr=1e-2` (Adam).
  LBFGS is also supported but the smoke default is Adam; a too-small `lr` or
  too-few `num_steps` leaves you far from the minimum.
- **`abs err` too large at N=6** — global AD-MPS is first-order Adam on **all**
  tensors at once; full chi=8 convergence at N=6 needs more steps (300 is the
  smoke count; 1e-3 is the smoke tolerance `AD_TOL[6]`). For machine precision
  use the **local** solvers (tutorials 03/04), which sweep one tensor at a time
  with LBFGS.
- **`below gnd = True`** — physically impossible; a variational energy below
  the exact ground by more than `1e-8` is a bug. Check `S = σ/2` and
  `complex128`.
- **`UserWarning: Converting a tensor with requires_grad=True to a scalar`** —
  this comes from the library's *report path* (`float(admps.energy())` to log
  history), **not** the loss path. It is expected and harmless; the loss path
  itself stays autograd-clean.

## When to use global AD-MPS

- You want the simplest "all tensors, one optimizer" baseline.
- Small systems, or as a reference for the local solvers.
- For faster per-bond convergence at larger N, switch to **one-site** (03) or
  **two-site** (04) AD local optimization.
