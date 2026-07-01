# Contributing

Thanks for interest in `latticeTN`. This is a research/educational codebase with
a strict **AD mainline** policy — please read
[`docs/AD_MAINLINE_POLICY.md`](docs/AD_MAINLINE_POLICY.md) before contributing.

## The one rule that governs everything

The primary solver is **automatic differentiation** of an MPS variational ansatz:

```
MPS parameters -> differentiable Rayleigh quotient -> loss.backward() -> torch optimizer step
```

- The loss path (`contractions.rayleigh_energy_native`) must stay autograd-clean:
  no `detach()`, `.data`, `torch.no_grad()`, unnecessary `.item()`, no
  `eigh`/`svd`/`qr`, no `dmrg`/`lanczos`. This is AST-enforced by tests.
- SVD/QR/canonicalization/compression are **optional post-step stabilization /
  projection / compression only** — never the optimizer.
- `dmrg` / `lanczos` / dense `eigh` are **reference baselines**, never imported
  by the AD modules.

## Before you start

- `pip install -r requirements.txt && pip install -r requirements-dev.txt && pip install -e .`
- Default tests are CPU-only; `torch.complex128`; open boundary; `S = sigma/2`.

## Making changes

1. Make small, testable changes.
2. For every meaningful change: run the smallest relevant pytest target, then
   the relevant `scripts/*_score.py --fast`.
3. Keep loss paths autograd-clean; put any `no_grad`/`.data`/`.detach()` in
   explicitly-marked post-step / diagnostics helpers.
4. A new stage adds its own `*_score.py` and a spec/protocol/report; do **not**
   modify default `validation_score`/`benchmark_score` lists to depend on GPU
   or classical-solver tests.

## Pause and ask (do not push past these)

- A change needs long GPU training.
- The validation target looks physically inconsistent.
- A dependency beyond torch/numpy/scipy/pytest/tqdm/matplotlib seems necessary.
- Passing tests would require weakening physics thresholds without justification.
- The variational energy falls below exact ground by more than tolerance.

## Legacy

`legacy/stage0_prototypes/` is archived history — do not import from it, do not
extend it. Reimplement ideas inside `latticetn/` with current conventions.

## Committing

Do not commit caches (they are gitignored), machine-private config
(`.claude/settings.local.json`), or temporary scratch/log files. Do commit
formal reports under `docs/`. Do not push to a remote unless you own it and it
exists.
