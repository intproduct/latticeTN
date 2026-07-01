---
name: latticetn-autovalidate
description: Run an eval-driven autonomous implementation loop for the latticeTN automatic-differentiation tensor-network validation project. Use this when the user asks Claude Code to make latticeTN solve the finite open-boundary 1D spin-1/2 Heisenberg chain with MPS/MPO/autograd and exact diagonalization checks.
---

# latticeTN autonomous validation skill

You are working in the latticeTN repository. Run this as an eval-driven implementation loop, not as a one-off code edit.

## Objective

Make this automatic-differentiation tensor-network project pass the validation protocol for solving the finite open-boundary 1D spin-1/2 Heisenberg chain with MPS + MPO + PyTorch autograd, verified against exact diagonalization.

## Required reading before changing code

Read these files first:

- `CLAUDE.md`
- `docs/PHYSICS_SPEC.md`
- `docs/VALIDATION_PROTOCOL.md`
- `docs/AUTOMATION_EXECUTION_PLAN.md`
- `docs/REVIEW_CHECKLIST.md`

If any required file is missing, create or restore it before implementing physics code.

## Stopping condition

Stop only when both are true:

1. `python scripts/validation_score.py --fast` exits with code 0.
2. `docs/NUMERICAL_REPORT.md` contains a pass/fail validation table with exact energies, variational energies, errors, commands, and limitations.

If `python scripts/validation_score.py --full` is cheap on CPU, run it and include the result. Do not run long GPU jobs.

## Autonomous loop

Repeat until the stopping condition or a pause condition is reached:

1. Inspect the repo and identify the next failing validation stage.
2. Make one focused implementation or test change.
3. Run the smallest relevant test.
4. Run `python scripts/validation_score.py --fast` at checkpoints.
5. Append a compact checkpoint entry to `docs/CLAUDE_PROGRESS.md`.
6. Continue to the next failing stage.

## Stage order

Do not jump directly to Heisenberg optimization. Follow this order:

1. Exact dense reference Hamiltonians.
2. MPS-to-dense state conversion for small N.
3. MPO-to-dense matrix conversion for small N.
4. TFI MPO dense consistency.
5. Differentiable Rayleigh quotient energy path.
6. Heisenberg MPO dense consistency.
7. Random MPS Heisenberg energy compared with dense-state energy.
8. Short CPU-only Heisenberg variational solve.
9. Numerical report.

## Required artifacts

Final work should include or update:

- `tests/test_reference_models.py`
- `tests/test_mps_dense.py`
- `tests/test_mpo_dense.py`
- `tests/test_tfi_mpo_dense.py`
- `tests/test_energy_rayleigh.py`
- `tests/test_heisenberg_mpo_dense.py`
- `tests/test_heisenberg_energy_dense_compare.py`
- `tests/test_heisenberg_variational_smoke.py`
- `scripts/run_heisenberg_small.py`
- `scripts/validation_score.py`
- `docs/CLAUDE_PROGRESS.md`
- `docs/NUMERICAL_REPORT.md`

## Constraints

- Do not run long training jobs.
- Do not use GPU in tests.
- Keep tests CPU-only and small-system only.
- Preserve autograd in differentiable energy paths.
- Do not silently change physics conventions.
- Do not weaken tolerances just to pass tests.
- Do not treat print-only scripts as tests.
- Pause and report if the variational energy goes below exact energy by more than tolerance.

## Preferred command sequence

Use the smallest relevant command first, then checkpoint with the score script:

```bash
pytest -q tests/test_reference_models.py
pytest -q tests/test_mpo_dense.py
pytest -q tests/test_energy_rayleigh.py
python scripts/validation_score.py --fast
```

Only run broader tests when local targets pass.
