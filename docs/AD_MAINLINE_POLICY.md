# AD Mainline Policy (latticeTN)

This document is the **standing policy** for what counts as the
automatic-differentiation (AD) mainline of latticeTN and what is allowed to sit
alongside it. It exists so the project does not silently drift from its stated
goal (`CLAUDE.md`: "an automatic-differentiation tensor network project built
with PyTorch" whose solver is "MPS + MPO + PyTorch autograd").

## 1. The AD mainline (the one true solver)

The primary solve path is:

```
MPS parameters (trainable nn.Parameter)
   -> differentiable Rayleigh quotient  E = <psi|H|psi>/<psi|psi>
   -> loss.backward()   (autograd)
   -> torch optimizer step (Adam / LBFGS)
```

- The loss is `latticetn.ad_variational.ADVariationalMPS.energy()` which calls
  `latticetn.contractions.rayleigh_energy_native(mps, mpo)`.
- That energy path is a plain einsum sweep; it must contain NO `torch.no_grad()`,
  `.detach()`, `.data`, unnecessary `.item()`, NO `eigh`/`svd`/`qr`, and NO call
  into `dmrg` / `lanczos`.

## 2. Allowed, non-mainline (reference baselines & oracles)

These modules exist ONLY as reference baselines / sanity oracles and MUST NOT be
imported or called inside any AD mainline loss/optimization path:

- `latticetn/dmrg.py` — classical two-site DMRG (Stage 4A/4B).
- `latticetn/lanczos.py` — Krylov local eigensolver (Stage 4B).
- exact diagonalization (`operators.exact_ground_energy`) — golden reference.

They live behind their own opt-in score scripts (`dmrg_score.py`,
`dmrg_benchmark_score.py`) and are never in `validation_score.py` /
`benchmark_score.py` / `ad_variational_score.py` default test lists.

## 3. Allowed uses of SVD / QR / canonicalization / compression

These are NON-differentiable linear-algebra tools. Permitted roles ONLY:

- **Gauge fixing / projection** — applied AFTER `optimizer.step()` and OUTSIDE
  the loss graph, under `torch.no_grad()` mutating `.data`
  (e.g. `_project(..., 'canonical')`).
- **Stabilization** — per-tensor L2 renormalization (`_renormalize`), same
  post-step, outside-the-graph contract.
- **Compression** — bond truncation (`canonical.svd_compress`) as a reporting /
  preprocessing / post-step projection, never as the optimizer.
- **Diagnostics** — canonical-error / norm reads.
- **Reference paths** — Stage 3A canonicalization / compression as correctness
  references, and dense observables for small-system validation.

SVD/QR/canonicalization/compression MUST NOT be used as the main optimization
algorithm. They never appear inside `rayleigh_energy_native` /
`native_mpo_numerator` / `native_norm_sq` (the loss path).

## 4. Autograd rule (hard)

Any AD-mainline feature's main loss path:
- NO `detach()`, `.data`, `torch.no_grad()`, or unnecessary `.item()`.
- `backward()` is called on the loss only.
- All `.data`-mutation / `no_grad` in an AD feature must be in explicitly-marked
  post-step projection / stabilization helpers, never in the loss.

This is enforced structurally by `tests/test_ad_variational_loss.py` and
`tests/test_ad_gauge_loss_integrity.py` (AST inspection of the loss path).

## 5. Scope discipline

- Stage 4B classical DMRG/Lanczos is a *completed reference baseline*. It is not
  to be expanded into the mainline.
- Subsequent stages (5A+) develop **AD local-tensor optimization** (e.g.
  gauge-stabilized AD, canonical-projection AD, optional bond-growing AD), NOT
  more traditional DMRG/Lanczos.
- Every new stage adds its own *_score.py and report; default
  `validation_score.py` / `benchmark_score.py` lists are not modified to depend
  on GPU or classical-solver tests.

## 6. How to check compliance

Run `python scripts/ad_variational_score.py --fast` (and the suite in
`docs/AD_MAINLINE_AUDIT.md`). If a change would let `dmrg`/`lanczos`/`eigh`/
`svd`/`qr`/`no_grad`/`detach`/`.data` reach the loss path, it violates this
policy and must be redirected.
