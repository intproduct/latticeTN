# AGENTS.md

## Project goal

This repository is an automatic-differentiation tensor network project built with PyTorch. The scientific validation target is:

> Use MPS + MPO + PyTorch autograd to solve the finite open-boundary 1D spin-1/2 Heisenberg chain, verified against exact diagonalization.

The project is successful only when `python scripts/validation_score.py --fast` passes and `docs/NUMERICAL_REPORT.md` contains a validation table.

## First files to read

Before implementation, read:

1. `docs/PHYSICS_SPEC.md`
2. `docs/VALIDATION_PROTOCOL.md`
3. `docs/AUTOMATION_EXECUTION_PLAN.md`
4. `docs/REVIEW_CHECKLIST.md`

For the full autonomous procedure, use the project skill:

```text
/latticetn-autovalidate
```

## Physics conventions

Default Heisenberg model:

```text
H = J * sum_{i=1}^{N-1} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
Sx = sigma_x / 2
Sy = sigma_y / 2
Sz = sigma_z / 2
J = 1.0
boundary = open
```

Do not silently switch between spin operators `S` and Pauli operators `sigma`.

## Engineering rules

- Make small, testable changes.
- Prefer correctness and validation over broad refactoring.
- Do not run long training jobs unless explicitly approved.
- Do not run GPU jobs in tests.
- Default tests must run on CPU.
- Default dtype for physics validation is `torch.complex128`.
- Do not use notebooks as the validation path.
- Do not treat print-only scripts as tests; tests must contain assertions.
- Preserve autograd: do not use `.detach()`, `.item()`, `.data`, or `torch.no_grad()` inside differentiable energy paths unless explicitly justified and isolated from training.
- Existing demos may be moved or wrapped, but do not delete useful code without noting why in `docs/CLAUDE_PROGRESS.md`.
- If a convention is ambiguous, record the assumption in `docs/CLAUDE_PROGRESS.md` and proceed with `docs/PHYSICS_SPEC.md`.

## Validation loop

For every meaningful implementation change:

1. Run the smallest relevant pytest target.
2. Run `python scripts/validation_score.py --fast` at checkpoints.
3. Record command, result, and next action in `docs/CLAUDE_PROGRESS.md`.
4. Continue only if the failure is understood.

## Standard commands

```bash
python -m pip install -r requirements.txt
bash scripts/run_fast_validation.sh
python scripts/validation_score.py --fast
python scripts/validation_score.py --full
```

## Pause conditions

Pause and ask the user instead of continuing if:

- A change would require long GPU training.
- The validation target appears physically inconsistent.
- A dependency beyond torch/numpy/scipy/pytest/tqdm/matplotlib seems necessary.
- Passing tests would require weakening physics thresholds without justification.
- The implementation reaches a stable but scientifically suspicious result, such as variational energy below exact ground energy by more than tolerance.
