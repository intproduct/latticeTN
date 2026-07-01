# Stage 5A — AD Local Tensor Optimization Spec

> Status: implemented by `latticetn/ad_local.py`. This is the AD mainline.
> SVD / QR / canonicalization are **optional stabilization**, never the solver.

## 1. Goal

Implement an **automatic-differentiation local-tensor optimization** prototype
on the AD mainline. Fix all MPS tensors except one *local / center tensor*; make
that single tensor a trainable `nn.Parameter` and optimize the differentiable
Rayleigh quotient with PyTorch autograd + a torch optimizer. Sweep the active
site across the chain to optimize the whole state.

This is the autograd analogue of DMRG's local update, but the local update is
**gradient descent, not a local eigensolver**.

## 2. Main solver path (AD mainline)

```
mixed-canonical MPS, all tensors frozen except center tensor c
   -> train center tensor c as nn.Parameter
   -> loss = rayleigh_energy_native(mps, mpo)   (differentiable einsum sweep)
   -> loss.backward()   (autograd; only c receives a gradient)
   -> torch optimizer step (Adam / LBFGS) on c
   -> [optional post-step stabilization: none|tensor_norm|qr|canonical]
   -> move orthogonality center to next site (QR / SVD center movement)
   -> repeat for the next site; sweep back and forth
```

- The loss and the `backward()` + optimizer step are the **only** optimization
  mechanism.
- `SVD` / `QR` / `canonicalization` / `compression` are permitted **only** as
  post-step stabilization / projection / center movement / compression, and run
  under `torch.no_grad()` mutating `.data`, **outside the loss graph**.
- `dmrg.py`, `lanczos.py`, and any classical / dense local eigensolver
  (`eigh`) are **reference baselines / oracles only** and are never imported or
  called inside this module.

## 3. Public API (`latticetn/ad_local.py`)

- `ADLocalOptimizer(mps, mpo, center)` — wraps an MPS already brought to (or to
  be brought to) mixed-canonical form with center `center`. Freezes every tensor
  except the center tensor, which is the only trainable parameter. Provides:
  - `energy()` / `loss()` — differentiable Rayleigh quotient (delegates to
    `contractions.rayleigh_energy_native`).
  - `parameters()` — the single center tensor (trainable).
  - `move_center(new_center, mode="qr")` — shift the orthogonality center by a
    `no_grad` QR/SVD sweep (center movement, NOT the optimizer). Re-freezes all
    but the new center.
- `train_ad_local(...)` — sweep driver: alternates right/left sweeps of
  per-site local Adam optimization, with optional post-step `stabilization`.
  Returns the standard history dict (energy / grad-norm / state-norm /
  canonical-error / bond history) plus per-sweep and per-site records.

## 4. Stabilization options (post-step, OUTSIDE the loss graph)

| value | meaning |
|---|---|
| `none` | no projection |
| `tensor_norm` | rescale the (active) center tensor to unit Frobenius norm under `no_grad` |
| `qr` | left-canonical QR projection of the active region written back onto `.data` under `no_grad` |
| `canonical` | full left-canonical QR sweep (Stage 3A) written back onto `.data` under `no_grad` |

All of these are scale/gauge projections; because the Rayleigh quotient is
gauge- and scale-invariant they do not change the physics. They are stability/
conditioning aids, **not** the optimizer.

## 5. Autograd rule (hard, per `docs/AD_MAINLINE_POLICY.md`)

The main loss path (`ADLocalOptimizer.energy()` / `loss()` and the underlying
`rayleigh_energy_native`) contains NO `detach()` / `.data` /
`torch.no_grad()` / unnecessary `.item()`, NO `eigh`/`svd`/`qr`, NO call into
`dmrg` / `lanczos`. Every `no_grad` / `.data` / `.detach()` lives in
explicitly-marked post-step / center-movement helpers.

Enforced by AST inspection in `tests/test_ad_local_opt_policy.py`.

## 6. Physics conventions

Unchanged: `H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open boundary,
torch.complex128, CPU-only`. No silent switch between `S` and `sigma`.

## 7. Comparisons (reference only)

- exact diagonalization (small N) — golden reference.
- global AD-MPS (`ADVariationalMPS` / `train_ad_mps`) — same mainline, all
  tensors trained simultaneously; AD-local is a different optimization
  strategy on the same differentiable loss.
- classical DMRG (`dmrg.run_dmrg`) — classical reference baseline, never in the
  AD path.
