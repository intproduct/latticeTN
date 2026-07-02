# Stage 5B — Two-Site AD Local Tensor Optimization Spec

> Status: implemented by `latticetn/ad_two_site.py`. Two-site AD is the AD
> mainline for this stage. SVD / QR / canonicalization are **optional split /
> compression / stabilization**, NEVER the solver.

## 1. Goal

Implement an **automatic-differentiation two-site local-tensor optimization**
on the AD mainline. At each bond (i, i+1) contract the two adjacent MPS site
tensors into a single two-site center tensor `Θ(l, s_i, s_{i+1}, r)`, make it
the only trainable `nn.Parameter`, and optimize the differentiable local
Rayleigh quotient `E(Θ) = <Θ|H_eff|Θ>/<Θ|Θ>` with PyTorch autograd + a torch
optimizer. Split `Θ` back into two site tensors by SVD with optional
`max_bond_dim` / `cutoff` truncation, then sweep the active bond across the
chain (left-to-right then right-to-left).

This is the autograd analogue of two-site DMRG's local update, but the local
update is **gradient descent on the two-site Rayleigh quotient, NOT a local
eigensolver (`eigh`/Lanczos)**. Optional bond growth / truncation is a
consequence of the SVD *split*, never of a solver.

## 2. Main solver path (AD mainline)

```text
two-site mixed-canonical MPS at bond (i, i+1)
   -> build frozen left/right MPO environments L, R  (constants, under no_grad)
   -> Theta = A_i * A_{i+1}                          (single trainable leaf)
   -> loss = <Theta|H_eff|Theta> / <Theta|Theta>     (differentiable einsum)
   -> loss.backward()                                (autograd; only Theta grads)
   -> torch optimizer step (Adam / LBFGS) on Theta
   -> [post-step split: SVD Theta -> A_i, A_{i+1} with optional chi/cutoff]
   -> re-canonicalize at next bond (QR gauge fixing, under no_grad)
   -> repeat; sweep left-to-right then right-to-left
```

- The loss and the `backward()` + optimizer step are the **only** optimization
  mechanism.
- `H_eff` is built from the **frozen, detached** left/right MPO environments
  and the two MPO tensors `W_i`, `W_{i+1}`; it is a constant w.r.t. `Θ`. Because
  the rest of the chain is orthonormal (mixed-canonical), the local quotient
  `E(Θ)` EQUALS the global Rayleigh quotient — minimizing it lowers the global
  energy (the standard two-site variational principle).
- `SVD` / `QR` / `canonicalization` / `compression` are permitted **only** as
  the post-step split and the inter-bond re-canonicalization, run under
  `torch.no_grad()` on detached data, **outside the loss graph**.
- `dmrg.py`, `lanczos.py`, and any classical / dense local eigensolver
  (`eigh`) are **reference baselines / oracles only** and are never imported or
  called inside this module.

## 3. Public API (`latticetn/ad_two_site.py`)

- `ADTwoSiteOptimizer(mps, mpo, bond)` — wraps an MPS, brings it to two-site
  mixed-canonical form at `bond`, contracts the two block tensors into a
  trainable `Θ`, and builds the frozen constant environments. Provides:
  - `energy()` / `loss()` — differentiable local Rayleigh quotient
    `<Θ|H_eff|Θ>/<Θ|Θ>` (pure einsum on `Θ` + frozen constants).
  - `parameters()` — the single `Θ` tensor (trainable).
  - `reset_bond(bond)` — re-canonicalize at a new bond and rebuild `Θ` +
    environments (gauge fixing + constant building, NOT the optimizer).
  - `split(max_bond_dim, cutoff, direction)` — SVD split of `Θ` back into two
    site tensors with optional truncation (compression/stabilization, NOT the
    optimizer). Returns `(truncation_error, kept_bond)`.
  - `global_energy()` / `norm()` / `max_bond_dim()` / `bond_dims()` —
    report-path diagnostics.
- `train_ad_two_site(...)` — sweep driver: alternates right (left-to-right) and
  left (right-to-left) sweeps of per-bond two-site AD optimization with
  optional `max_bond_dim` / `cutoff`. Returns the standard history dict
  (energy / grad-norm / bond-dim / truncation-error / sweep-direction) plus
  per-sweep and per-bond records.

## 4. Post-step split / compression (OUTSIDE the loss graph)

`_split_theta(Theta, max_bond_dim, cutoff, direction)`:
- SVD of `Theta` reshaped `(l*s_i, s_{i+1}*r)`.
- Keep `k = max(1, min(max_bond_dim, #singular values with s^2 >= cutoff))`.
- `direction='right'`: `A_i = U` (left-canonical), `A_{i+1} = S Vh` (carries norm).
- `direction='left'`: `A_i = U S` (carries norm), `A_{i+1} = Vh` (right-canonical).
- `truncation_error = sum(discarded s^2) / sum(all s^2)` in `[0, 1]`.
- Runs under `torch.no_grad()` on a detached `Theta` — compression /
  stabilization, **not** the optimizer.

Because the Rayleigh quotient is gauge- and scale-invariant, the split and the
inter-bond re-canonicalization (`_two_site_mixed_canonical`, QR) do not change
the physics; they are stability / conditioning / compression aids.

## 5. Autograd rule (hard, per `docs/AD_MAINLINE_POLICY.md`)

The main loss path (`ADTwoSiteOptimizer.energy` / `loss`) contains NO
`detach()` / `.data` / `torch.no_grad()` / unnecessary `.item()`, NO
`eigh`/`svd`/`qr`, NO call into `dmrg` / `lanczos` or into any split /
canonicalization helper. Every `no_grad` / `.data` / `.detach()` / `svd` / `qr`
lives in explicitly-marked preprocessing / post-step helpers
(`_two_site_mixed_canonical`, `_left_mpo_env`, `_right_mpo_env`,
`_split_theta`, `ADTwoSiteOptimizer.reset_bond`, `split`,
`_stabilize_tensor_norm`).

Enforced by AST inspection in `tests/test_ad_two_site_policy.py`.

## 6. Physics conventions

Unchanged: `H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open boundary,
torch.complex128, CPU-only`. No silent switch between `S` and `sigma`.

## 7. Comparisons (reference only)

- exact diagonalization (small N) — golden reference.
- one-site AD local (`latticetn/ad_local.py`) — same AD mainline, single-site
  center sweep; two-site AD additionally allows bond growth / truncation.
- global AD-MPS (`ADVariationalMPS` / `train_ad_mps`) — same mainline, all
  tensors trained simultaneously.
- classical DMRG (`dmrg.run_dmrg`) — classical reference baseline, never in the
  AD path.
