# One-shot Claude Code autonomous prompt

Paste this in Claude Code from the repository root after copying this pack into the repo.

```text
/goal Make `python scripts/validation_score.py --fast` pass and produce `docs/NUMERICAL_REPORT.md` with the Heisenberg validation table.

Use the project skill `/latticetn-autovalidate`.

Run this as an eval-driven autonomous implementation loop, not as a one-off edit. Before changing code, read `CLAUDE.md`, `docs/PHYSICS_SPEC.md`, `docs/VALIDATION_PROTOCOL.md`, and `docs/AUTOMATION_EXECUTION_PLAN.md`.

Follow the validation stages in order. Make one focused change at a time. Run the smallest relevant pytest target after each change. At checkpoints, run `python scripts/validation_score.py --fast`. Append command results and next action to `docs/CLAUDE_PROGRESS.md`.

Stop only when `python scripts/validation_score.py --fast` exits 0 and `docs/NUMERICAL_REPORT.md` contains exact energies, variational energies, errors, commands, pass/fail status, and limitations.

Do not run long training jobs. Do not use GPU in tests. Do not weaken physics thresholds. Preserve autograd. Pause if the variational energy falls below exact ground energy beyond tolerance or if a physics convention is ambiguous.
```

Alternative recurring local loop, if your Claude Code version exposes `/loop`:

```text
/loop Use `.claude/loop.md` to perform one latticeTN validation iteration. Stop when `python scripts/validation_score.py --fast` passes and `docs/NUMERICAL_REPORT.md` is current.
```
