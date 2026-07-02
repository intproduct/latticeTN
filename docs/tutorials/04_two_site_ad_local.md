# 04 — Two-site AD local optimization

Optimize **one two-site block `Θ` at a time** on the differentiable local
Rayleigh quotient `E(Θ)=<Θ|H_eff|Θ>/<Θ|Θ>`, sweep the active bond, and
**optionally grow or truncate the bond** at the SVD split. This is the Stage
5B AD mainline: the autograd analogue of two-site DMRG — **gradient descent
on `Θ`, not a local eigensolver**.

## Goal

- Understand what makes two-site "two-site" (a trainable 2-site block `Θ`).
- Run `train_ad_two_site` and reach machine precision.
- See that the SVD split is **compression, not the solver**.
- Optionally grow the bond (`max_bond_dim`) toward the exact entanglement.

## Why two-site? (design motivation)

One-site AD (tutorial 03) keeps the bond dimension **fixed** — you cannot
adapt χ to the entanglement structure. Two-site AD contracts two adjacent site
tensors into a single trainable block `Θ(l, s_i, s_{i+1}, r)`, optimizes it,
then **splits** `Θ` back into two site tensors by SVD with an optional
`max_bond_dim` / `cutoff`. The split can **grow or truncate** the bond, so the
ansatz adapts to the entanglement as the sweep proceeds — exactly like
two-site DMRG, but the local update is gradient descent on `Θ`, not a local
`eigh`.

The SVD split is **post-step compression**, under `no_grad` on detached data,
**outside the loss graph**. It is **not the solver**. The solver is always
`loss.backward()` + a torch optimizer step on `Θ`.

## The mainline (recap, two-site form)

```
two-site mixed-canonical MPS at bond (i, i+1)
   -> frozen left/right MPO environments L, R  (constants, under no_grad)
   -> Θ = A_i * A_{i+1}                         (single trainable leaf)
   -> loss = <Θ|H_eff|Θ>/<Θ|Θ>                  (differentiable einsum)
   -> loss.backward() + optimizer step (LBFGS) on Θ
   -> [post-step split: SVD Θ -> A_i, A_{i+1} with optional max_bond_dim/cutoff]
   -> re-canonicalize at next bond (QR gauge fixing, under no_grad)
   -> sweep left-to-right then right-to-left
```

Because the rest of the chain is orthonormal, `E(Θ)` **equals** the global
Rayleigh quotient — minimizing it lowers the global energy.

## Minimal code

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_two_site import train_ad_two_site
from latticetn.operators import heisenberg_dense, exact_ground_energy

tc.manual_seed(0)
N, chi, dtype = 6, 8, tc.complex128

mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)

res = train_ad_two_site(mps, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                        optimizer="lbfgs", max_bond_dim=8, cutoff=None)

E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))
print("final   E =", res["final_energy"])
print("exact  E0 =", E0)
print("abs err   =", abs(res["final_energy"] - E0))
print("final bond dims =", res["final_bond_dims"])
```

## Run command

```bash
python scripts/run_ad_two_site.py --N 6 --chi 8 --num-sweeps 4 --local-steps 20 --print
```

## Expected output

With `N=6, chi=8, 4 sweeps, 20 local_steps, lr=1.0, LBFGS, max_bond_dim=8,
cutoff=None, seed=0`:

```text
final   E = -2.4935771330
exact  E0 = -2.4935771339
abs err   = 8.4e-10            # within AD_TWO_SITE_TOL[6] = 1e-5 (machine precision)
final bond dims = [2, 4, 8, 4, 2]
```

For `N=4`: `final E ≈ -1.6160254036`, `abs err ≈ 1.8e-10`. The bond grows to
`[2, 4, 2]` (max 4) — the entanglement structure of the exact ground state,
recovered purely from gradient descent on `Θ` plus the SVD split.

## Common errors

- **Confusing the SVD split with the solver** — the SVD in `split()` is
  **compression**, under `no_grad`. The solver is `backward()` + optimizer step
  on `Θ`. If you put SVD/`eigh` in the loss path, you break the autograd rule.
- **`max_bond_dim` too small** — truncates the bond and raises the energy error.
  For N=6 the exact chain needs χ≥8; capping at 4 will not reach machine
  precision. Set `max_bond_dim` ≥ the entanglement you expect, or `None` for
  full growth.
- **Truncation errors are all 0** — expected when the bond is already saturated
  below `max_bond_dim` (nothing to truncate). Non-zero truncation appears when
  you cap χ below the natural bond.
- **`stabilization` only accepts `none|tensor_norm`** — two-site has a narrower
  choice than one-site (no `qr`/`canonical`); the SVD split already handles
  gauge, so `qr` is not needed.

## When to use two-site AD local

- You want **bond growth / adaptivity** (the ansatz chooses its own χ via the
  SVD split + `max_bond_dim`/`cutoff`).
- You want machine-precision ground states and a DMRG-like workflow, but with
  autograd as the local update.
- Compare against the DMRG reference (`dmrg.run_dmrg`) — they should reach the
  same variational minimum (DMRG is a **reference baseline**, never the AD
  mainline).
