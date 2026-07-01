Run one autonomous maintenance iteration for latticeTN.

1. Read `CLAUDE.md`, `docs/VALIDATION_PROTOCOL.md`, and `docs/CLAUDE_PROGRESS.md` if present.
2. Run `python scripts/validation_score.py --fast`.
3. If it passes, verify that `docs/NUMERICAL_REPORT.md` is present and current, then stop.
4. If it fails, identify the next failing validation stage, make one focused fix, run the smallest relevant test, then update `docs/CLAUDE_PROGRESS.md`.
5. Do not run long GPU jobs or weaken physics thresholds.
