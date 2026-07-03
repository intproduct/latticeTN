# /goal prompt for Codex: AD-mainline benchmark suite

You are continuing the latticeTN project. Read `AGENTS.md`, `CLAUDE.md`, `docs/AD_MAINLINE_POLICY.md`, `docs/AD_TWO_SITE_SPEC.md`, and this file before editing.

## Goal

Promote the current two-site AD local-tensor optimizer from a Heisenberg-only demo into a reusable **AD-mainline benchmark suite** for 1D open-boundary MPO models.

The mainline algorithm is **not classical DMRG**. It is:

```text
MPS + MPO
-> two-site mixed-canonical center Theta
-> differentiable local Rayleigh quotient E(Theta)
-> loss.backward() + torch optimizer step
-> post-step SVD split/compression only
-> finite-chain sweeps
```

## Required deliverables

1. Add or refine `scripts/run_ad_model_benchmark.py`.
2. Ensure it supports:
   - `--model heisenberg`
   - `--model tfi`
   - `--model spinless_tv`
   - `--model hubbard`
3. Ensure it supports:
   - `--N`, `--chi`, `--sweeps`
   - `--device cpu|cuda`
   - `--dtype complex64|complex128|float32|float64`
   - `--optimizer adam|lbfgs`
   - `--local-steps`, `--lbfgs-iters`, `--lr`
   - `--init auto|neel|all_up|all_down|cdw|hubbard_neel|empty|random`
   - model parameters `--J`, `--h`, `--t`, `--V`, `--U`, `--mu`, `--field`
   - `--output` JSON path
4. Add tests in `tests/test_ad_model_benchmark_helpers.py` or equivalent. Tests must be CPU-only and short.
5. Add/update docs with a benchmark protocol and commands.

## Hard constraints

- The AD benchmark runner must not import `latticetn.dmrg`.
- The AD benchmark runner must not import `latticetn.lanczos`.
- The AD benchmark runner must not call exact diagonalization for large N.
- SVD/QR/canonicalization may appear only as preprocessing/post-step gauge/split/compression, not as the optimizer.
- Local optimization must be performed through PyTorch autograd: `loss.backward()` and a torch optimizer.
- Default pytest must not require a GPU.

## Validation commands

Run at least:

```bash
python -m pip install -e .
pytest -q tests/test_ad_two_site_loss.py tests/test_ad_two_site_gradients.py tests/test_ad_two_site_sweep_smoke.py
pytest -q tests/test_ad_model_benchmark_helpers.py
python scripts/run_ad_model_benchmark.py --model heisenberg --N 8 --chi 8 --sweeps 2 --device cpu --dtype complex128 --optimizer lbfgs --local-steps 1 --lbfgs-iters 4 --output docs/tmp_ad_heisenberg_n8.json
python scripts/run_ad_model_benchmark.py --model hubbard --N 4 --chi 8 --sweeps 1 --device cpu --dtype complex128 --optimizer lbfgs --local-steps 1 --lbfgs-iters 3 --output docs/tmp_ad_hubbard_n4.json
```

If GPU is available and explicitly approved, run:

```bash
python scripts/run_ad_model_benchmark.py --model heisenberg --N 80 --chi 64 --sweeps 6 --device cuda --dtype complex64 --init auto --optimizer lbfgs --local-steps 1 --lbfgs-iters 10 --lr 1.0 --stabilization tensor_norm --output docs/ad_heisenberg_N80_chi64.json
```

Expected ballpark for the N=80 Heisenberg CUDA run, based on a previous V100 result:

```text
E/site around -0.4408 for chi=64, 6 directional sweeps, OBC.
ED skipped.
DMRG/Lanczos not used.
```

Do not require exact equality to this number; use it as a sanity check.

## Documentation expectation

Create or update `docs/AD_REAL_MODEL_BENCHMARK_SPEC.md` with:

- algorithm statement;
- model list and conventions;
- commands for PowerShell and bash;
- output schema;
- acceptance criteria;
- known limitations.

Also update an appropriate progress file with commands run and results.
