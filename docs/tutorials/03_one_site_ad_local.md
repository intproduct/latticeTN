# 03 — One-site AD local optimization

Optimize **one center tensor at a time** on the differentiable Rayleigh
quotient, sweeping the orthogonality center across the chain by QR. This is
the Stage 5A AD mainline: the autograd analogue of one-site DMRG, with
**gradient descent replacing the local eigensolver**.

## Goal

- Understand *why* local sweeps beat global Adam for per-bond convergence.
- Run `train_ad_local` and reach machine precision on small N.
- See that QR is **center movement, not the solver**.

## Why local? (design motivation)

Global AD-MPS (tutorial 02) updates every tensor through one shared loss; the
optimizer must disentangle N coupled gradients, so first-order Adam converges
slowly at larger N. The local idea: **freeze all tensors except one center
tensor**, minimize the (now low-dimensional) Rayleigh quotient on just that
tensor with a few LBFGS steps, then **move** the center to the next site and
repeat — a sweep. Because the chain is orthonormal around the center, the
local quotient **equals** the global Rayleigh quotient (the standard
variational principle), so minimizing it lowers the global energy.

Crucially, "move the center" is done by **QR** — and that QR is **gauge
fixing, not the optimizer**. It runs under `no_grad` mutating `.data`,
outside the loss graph. The solver is always `loss.backward()` + a torch
optimizer step on the center tensor.

## The mainline (recap, local form)

```
mixed-canonical MPS, orthogonality center at site c
   -> only the center tensor is trainable (nn.Parameter)
   -> loss = <ψ|H|ψ>/<ψ|ψ>  (differentiable; equals the global quotient)
   -> loss.backward() + optimizer step (LBFGS) on the center
   -> move center c -> c+1 by QR  (gauge fixing, OUTSIDE the loss graph)
   -> sweep left-to-right then right-to-left
```

## Minimal code

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_local import train_ad_local
from latticetn.operators import heisenberg_dense, exact_ground_energy

tc.manual_seed(0)
N, chi, dtype = 6, 8, tc.complex128

mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)

res = train_ad_local(mps, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                     optimizer="lbfgs", stabilization="qr")

E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))
print("final   E =", res["final_energy"])
print("exact  E0 =", E0)
print("abs err   =", abs(res["final_energy"] - E0))
```

## Run command

```bash
python scripts/run_ad_local_opt.py --N 6 --chi 8 --num-sweeps 4 --local-steps 20 --print
```

## Expected output

With `N=6, chi=8, 4 sweeps, 20 local_steps, lr=1.0, LBFGS, stabilization='qr',
seed=0`:

```text
final   E = -2.4935771330
exact  E0 = -2.4935771339
abs err   = 8.9e-10            # within AD_LOCAL_TOL[6] = 1e-5 (machine precision)
```

For `N=4`: `final E ≈ -1.6160254037`, `abs err ≈ 4.2e-11` (tolerance `1e-8`).
Compare to global AD-MPS at N=6 (`abs err ≈ 2.6e-4`): the local sweep reaches
**machine precision** in about a second — the per-bond LBFGS conditioning is
far better than first-order Adam on all tensors.

## Common errors

- **`stabilization="qr"` vs `"none"`** — `qr` re-canonicalizes the chain as the
  center moves, keeping the sweep well-conditioned. `none` skips this and can
  drift on longer chains. The smoke default is `"qr"`.
- **Energy increases between sweeps** — you likely passed `optimizer="adam"`
  with a bad `lr` for the local problem; LBFGS is the default/smoke choice
  because the local Rayleigh problem is near-quadratic in the center tensor.
- **`RuntimeError` from QR** — should not happen on CPU `complex128`; if you
  see it on GPU, see tutorial 05 (GPU QR/SVD are supported but small systems
  are overhead-dominated).
- **Confusing QR with the solver** — QR is **center movement**, under `no_grad`.
  The solver is `backward()` + `optimizer.step()`. If you find yourself calling
  QR/SVD inside a loss path, stop — that violates the autograd rule.

## When to use one-site AD local

- You want machine-precision ground states at small-moderate N with a **fixed**
  bond dimension.
- You do not need bond growth (one-site keeps χ fixed).
- For bond growth / truncation, see tutorial **04 — Two-site AD local**.
