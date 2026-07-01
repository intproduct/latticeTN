# Claude Code autonomous execution plan

Use this plan to turn the old manual multi-prompt workflow into a single autonomous project.

## Objective

Make the latticeTN project pass the validation protocol for a 1D spin-1/2 Heisenberg chain using PyTorch autograd + MPS/MPO, verified against exact diagonalization.

## Stopping condition

Stop when:

```bash
python scripts/validation_score.py --fast
```

returns exit code 0, and `docs/NUMERICAL_REPORT.md` exists with a clear pass/fail table.

If `--full` exists and is affordable on CPU, run:

```bash
python scripts/validation_score.py --full
```

and include the result in the report.

## Checkpoint loop

Claude Code should work in checkpoints:

1. Read `CLAUDE.md`, `docs/PHYSICS_SPEC.md`, and `docs/VALIDATION_PROTOCOL.md`.
2. Inspect current code and tests.
3. Create or update `docs/CLAUDE_PROGRESS.md`.
4. Identify the next failing validation stage.
5. Make one focused change.
6. Run the smallest relevant test.
7. Run `python scripts/validation_score.py --fast` when the local test passes.
8. Log command results and next step.
9. Continue until the stopping condition is met or a pause condition is triggered.

## Priority order

Do not skip stages.

1. Exact dense reference models.
2. MPS/MPO dense debug bridges.
3. TFI MPO dense consistency.
4. Rayleigh quotient energy path.
5. Heisenberg MPO dense consistency.
6. Heisenberg random-MPS energy consistency.
7. Short Heisenberg variational optimization.
8. Numerical report.

## Allowed refactors

Allowed:

- Create a `tests/` folder.
- Convert print-only tests into pytest tests.
- Add helper modules for dense references and debugging.
- Add clear wrappers around legacy classes.
- Move demo scripts into `examples/` or `scripts/` if imports are updated.

Avoid until tests exist:

- Large package restructure.
- Rewriting all MPS/MPO classes from scratch.
- Changing boundary-condition semantics.
- Changing dtype/device defaults without documenting why.

## Progress log format

Append to `docs/CLAUDE_PROGRESS.md` after each checkpoint:

```md
## Checkpoint <number>: <short title>

Goal:

Files changed:

Commands run:

Result:

Current failing test or bottleneck:

Next action:
```
