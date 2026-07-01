# Documentation Index

Navigation for the `latticeTN` documentation.

## Start here

- [README.md](../README.md) — project overview, install, quick validation, minimal example.
- [REPO_STATUS.md](../REPO_STATUS.md) — current repository state (mainline vs. baseline vs. legacy).
- [ROADMAP.md](../ROADMAP.md) — future directions (not implemented).

## User documentation

- [USER_GUIDE.md](USER_GUIDE.md) — **the** user-facing guide: install, concepts,
  examples, observables, canonicalization, adding models, scores, pitfalls.
- [API_OVERVIEW.md](API_OVERVIEW.md) — module-by-module reference.

## Policy & direction

- [AD_MAINLINE_POLICY.md](AD_MAINLINE_POLICY.md) — standing policy: what is the
  AD mainline, what is allowed alongside it, autograd rules.
- [AD_MAINLINE_AUDIT.md](AD_MAINLINE_AUDIT.md) — most recent mainline audit (PASS).
- [PROJECT_DIRECTION.md](PROJECT_DIRECTION.md) — project direction statement.

## Physics & validation

- [PHYSICS_SPEC.md](PHYSICS_SPEC.md) — physics conventions (S = sigma/2, J, OBC, complex128).
- [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md) — validation procedure.
- [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) — review checklist.
- [AUTOMATION_EXECUTION_PLAN.md](AUTOMATION_EXECUTION_PLAN.md) — automation plan.

## Stage specs / protocols / reports

| Stage | Spec | Protocol | Report |
|---|---|---|---|
| Stage 1/2 validation | — | VALIDATION_PROTOCOL | NUMERICAL_REPORT |
| Stage 2 benchmark | BENCHMARK_SPEC | BENCHMARK_PROTOCOL | BENCHMARK_REPORT |
| Stage 3A canonical | CANONICALIZATION_SPEC | CANONICALIZATION_PROTOCOL | CANONICALIZATION_REPORT |
| Stage 3B contraction | CONTRACTION_SPEC | CONTRACTION_PROTOCOL | CONTRACTION_REPORT |
| Stage 4A/4B DMRG baseline | DMRG_SPEC, DMRG_BENCHMARK_SPEC | DMRG_PROTOCOL, DMRG_BENCHMARK_PROTOCOL | DMRG_REPORT, DMRG_BENCHMARK_REPORT |
| Stage 4R global AD-MPS | AD_VARIATIONAL_SPEC | AD_VARIATIONAL_PROTOCOL | AD_VARIATIONAL_REPORT |
| Stage 5A gauge AD | AD_GAUGE_SPEC | AD_GAUGE_PROTOCOL | AD_GAUGE_REPORT |
| Stage 5A AD local opt | AD_LOCAL_OPT_SPEC | AD_LOCAL_OPT_PROTOCOL | AD_LOCAL_OPT_REPORT |
| GPU readiness | GPU_TESTING_PROTOCOL | — | GPU_REPORT |

## Progress logs

`CLAUDE_PROGRESS*.md` (per-stage command/result/next-action logs). The
`/latticetn-autovalidate` skill automates the full validation procedure.

## Legacy

- [../legacy/README.md](../legacy/README.md) — archived pre-package prototypes
  (not used, kept for traceability).
