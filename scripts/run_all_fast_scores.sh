#!/usr/bin/env bash
# Run all fast score scripts in sequence (CPU-only; NO GPU smoke by default).
# Exits non-zero if any score fails. Use a real Python interpreter; the `python`
# on Windows PATH may be a broken MS Store stub — export PYTHON=... to override.
set -u

PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=""   # enforce CPU for the score runs

SCORES=(
  validation_score
  benchmark_score
  canonical_score
  contraction_score
  ad_variational_score
  ad_local_opt_score
)

failed=0
for s in "${SCORES[@]}"; do
  echo "==================== $s --fast ===================="
  "$PYTHON" "scripts/${s}.py" --fast
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "FAILED: $s (exit $rc)"
    failed=$((failed + 1))
  fi
done

if [ "$failed" -ne 0 ]; then
  echo ""
  echo "run_all_fast_scores: $failed score(s) FAILED."
  exit 1
fi
echo ""
echo "run_all_fast_scores: all ${#SCORES[@]} scores PASS."
