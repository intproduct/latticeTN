#!/usr/bin/env python3
"""Compatibility wrapper. Prefer scripts/validation_score.py for Claude Code."""
from validation_score import main

if __name__ == "__main__":
    raise SystemExit(main())
