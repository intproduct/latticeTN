# Documentation Index

Navigation for the `latticeTN` documentation.

## Start here

- [README.md](../README.md) — project overview, install, quick validation, minimal example.
- [REPO_STATUS.md](../REPO_STATUS.md) — current repository state (mainline vs. baseline vs. legacy).
- [ROADMAP.md](../ROADMAP.md) — future directions (not implemented).

## User documentation

- [USER_GUIDE.md](USER_GUIDE.md) — **the** user-facing guide (EN): install, concepts,
  examples, observables, canonicalization, adding models, scores, pitfalls.
- [USER_GUIDE.zh-CN.md](USER_GUIDE.zh-CN.md) — 中文用户指南（面向中文用户解释
  设计动机，非机械翻译）。
- [API_OVERVIEW.md](API_OVERVIEW.md) — module-by-module reference.

## Tutorials (EN)

- [01 — Quickstart (Heisenberg)](tutorials/01_quickstart_heisenberg.md)
- [02 — Global AD-MPS](tutorials/02_global_ad_mps.md)
- [03 — One-site AD local](tutorials/03_one_site_ad_local.md)
- [04 — Two-site AD local](tutorials/04_two_site_ad_local.md)
- [05 — CPU/GPU benchmark](tutorials/05_cpu_gpu_benchmark.md)
- [06 — Add a new model](tutorials/06_add_new_model.md)

## 教程 (中文)

- [01 — 快速开始 (Heisenberg)](tutorials.zh-CN/01_快速开始_Heisenberg.md)
- [02 — Global AD-MPS](tutorials.zh-CN/02_Global_AD_MPS.md)
- [03 — 单点 AD 局部](tutorials.zh-CN/03_One_site_AD_local.md)
- [04 — 两点 AD 局部](tutorials.zh-CN/04_Two_site_AD_local.md)
- [05 — CPU/GPU benchmark](tutorials.zh-CN/05_CPU_GPU_benchmark.md)
- [06 — 添加新模型](tutorials.zh-CN/06_添加新模型.md)

## Doc site (optional)

The docs can be browsed as a local MkDocs site:
`pip install -r requirements-docs.txt && mkdocs serve` (see `mkdocs.yml`).
No GitHub Pages auto-deploy is configured; the Markdown stays readable on
GitHub directly.

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
| Stage 5B two-site AD | AD_TWO_SITE_SPEC | AD_TWO_SITE_PROTOCOL | AD_TWO_SITE_REPORT |
| Stage 6A CPU/GPU AD benchmark | AD_GPU_BENCHMARK_SPEC | AD_GPU_BENCHMARK_PROTOCOL | AD_GPU_BENCHMARK_REPORT |
| Stage 7A spinless fermion t-V | FERMION_SPEC | FERMION_PROTOCOL | FERMION_REPORT |
| Stage 7B model builder + registry | MODEL_BUILDER_SPEC | MODEL_BUILDER_PROTOCOL | MODEL_BUILDER_REPORT |
| Stage 7C spinful Hubbard | HUBBARD_SPEC | HUBBARD_PROTOCOL | HUBBARD_REPORT |
| Stage 12A-P0 physics compliance | — | VALIDATION_PROTOCOL | STAGE12A_P0_PHYSICS_COMPLIANCE |
| Stage 12B traditional TDVP | — | — | STAGE12B_TDVP_REPORT |
| GPU readiness | GPU_TESTING_PROTOCOL | — | GPU_REPORT |

## Progress logs

`CLAUDE_PROGRESS*.md` (per-stage command/result/next-action logs). The
`/latticetn-autovalidate` skill automates the full validation procedure.

## Legacy

- [../legacy/README.md](../legacy/README.md) — archived pre-package prototypes
  (not used, kept for traceability).
