# Codex handoff: latticeTN AD-mainline benchmark stage

## Current project state

This repository is an automatic-differentiation tensor-network project using PyTorch. The mainline algorithm is **two-site AD local-tensor optimization**, not classical Lanczos/DMRG:

```text
MPS + MPO
-> choose a two-site center tensor Theta
-> minimize differentiable local Rayleigh quotient E(Theta) by loss.backward() + torch optimizer
-> SVD split/compression only after the AD optimizer step
-> sweep through the finite open chain
```

The classical DMRG/Lanczos code exists and is useful as a reference baseline, but it must not be used inside the AD-mainline benchmark runner.

## Recent manual validation already completed

A standalone AD-only N=80 Heisenberg run was executed on Windows PowerShell with a Tesla V100-SXM2-16GB:

```powershell
python .\run_latticetn_n80_ad_two_site.py `
  --N 80 `
  --chi 64 `
  --sweeps 6 `
  --device cuda `
  --dtype complex64 `
  --init neel `
  --optimizer lbfgs `
  --local-steps 1 `
  --lbfgs-iters 10 `
  --lr 1.0 `
  --stabilization tensor_norm
```

Observed result:

```text
FINAL E                 = -35.2651252746582031
FINAL E / site          = -0.4408140659332275
FINAL E / bond          = -0.4463939908184583
E/site - Bethe e_inf    = 0.0023331146267178
final max trunc         = 0.000e+00
final max grad norm     = 1.144e-02
final max bond          = 64
total runtime           = 50.623 s
ED status               = skipped by design
classical DMRG/Lanczos  = not used
```

This means the AD-mainline path is no longer just a small-N toy. The next stage is to turn this into a reproducible benchmark suite over models, sizes, and bond dimensions.

## Main objective for Codex

Create a reusable AD benchmark path that supports at least:

- `--model heisenberg`
- `--model tfi`
- `--model spinless_tv`
- `--model hubbard`

The runner must:

- use `latticetn.ad_two_site.ADTwoSiteOptimizer`;
- use `loss.backward()` and a torch optimizer as the only local optimization mechanism;
- skip exact diagonalization for large N by design;
- not import or call `latticetn.dmrg` or `latticetn.lanczos`;
- support GPU and CPU;
- write machine-readable JSON results;
- print enough diagnostics to judge convergence: energy, E/N, delta energy, max grad norm, truncation, max bond, runtime, GPU memory.

## Files in this handoff bundle

Add these files to the repository root, preserving paths:

```text
CODEX_GOAL_AD_BENCHMARK.md
CODEX_HANDOFF.md
docs/AD_REAL_MODEL_BENCHMARK_SPEC.md
scripts/run_ad_model_benchmark.py
tests/test_ad_model_benchmark_helpers.py
```

Then ask Codex to use `CODEX_GOAL_AD_BENCHMARK.md` as the goal prompt.

## Suggested first commands after adding files

```bash
python -m pip install -e .
python scripts/run_ad_model_benchmark.py --model heisenberg --N 8 --chi 8 --sweeps 2 --device cpu --dtype complex128 --optimizer lbfgs --local-steps 1 --lbfgs-iters 4 --output docs/tmp_ad_heisenberg_n8.json
pytest -q tests/test_ad_model_benchmark_helpers.py
```

On the Windows V100 machine, after smoke tests:

```powershell
$env:PYTHONPATH="."
python .\scripts\run_ad_model_benchmark.py `
  --model heisenberg `
  --N 80 `
  --chi 64 `
  --sweeps 6 `
  --device cuda `
  --dtype complex64 `
  --init auto `
  --optimizer lbfgs `
  --local-steps 1 `
  --lbfgs-iters 10 `
  --lr 1.0 `
  --stabilization tensor_norm `
  --output docs\ad_heisenberg_N80_chi64.json
```

Then try Hubbard:

```powershell
$env:PYTHONPATH="."
python .\scripts\run_ad_model_benchmark.py `
  --model hubbard `
  --N 20 `
  --chi 64 `
  --sweeps 6 `
  --device cuda `
  --dtype complex64 `
  --init auto `
  --t 1.0 `
  --U 4.0 `
  --mu 0.0 `
  --field 0.0 `
  --optimizer lbfgs `
  --local-steps 1 `
  --lbfgs-iters 10 `
  --lr 1.0 `
  --stabilization tensor_norm `
  --output docs\ad_hubbard_N20_U4_chi64.json
```

## Acceptance criteria

1. Existing fast validation still passes or any failure is documented with a narrow cause.
2. `pytest -q tests/test_ad_model_benchmark_helpers.py` passes on CPU.
3. The benchmark runner can run Heisenberg N=80, chi=64, CUDA, no ED, no Lanczos/DMRG.
4. The benchmark runner can run a small Hubbard smoke case without ED.
5. Documentation is updated with the exact commands and the result table.

## Do not do yet

- Do not claim real-material simulation support.
- Do not add PBC/cylinder/2D features in this stage.
- Do not weaken AD-mainline policy by using Lanczos/eigh as the local solver.
- Do not put GPU tests into the default pytest suite.
- Do not require ED for N>12 spin systems or for Hubbard beyond tiny smoke cases.
