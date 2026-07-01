# Legacy / archived prototypes

This directory holds **archived, pre-`latticetn`-package prototype scripts**
kept for traceability only. They are **not** part of the active codebase, are
**not** collected by the default test suite (`pytest.ini` scopes collection to
`tests/`), and are **not imported** by the `latticetn/` package or by any
`scripts/` score runner.

## Directory layout

### `stage0_prototypes/`

Early Stage-0 exploratory code written before the `latticetn/` package was
introduced. These files import each other (e.g. `test_ad_tn.py` imports
`AD_MPS_fixed`/`AD_MERA`/`AD_PEPS`) and do **not** use the canonical
`latticetn.mps` / `latticetn.mpo` / `latticetn.ad_*` modules.

Contents include:
- `AD_MPS.py`, `AD_MPS_fixed.py`, `AD_DMRG.py`, `AD_MERA.py`, `AD_PEPS.py` —
  prototype AD tensor-network sketches (MPS / MERA / PEPS).
- `operators.py` — a standalone root operator module (the canonical one is
  `latticetn/operators.py`).
- `check_mps_shape.py`, `check_mpo_shape.py` — ad-hoc shape-check scripts.
- `test_*.py`, `test_*.ipynb` — early prototype "tests" (plain scripts with
  `print`s, mostly without assertions; not real tests).
- `update_log.md` — an early development log.

## Why they are here (not deleted)

The repo-prep goal requires that useful history not be silently deleted; these
files are moved here instead so the repository root is a clean, installable
Python package while the early prototypes remain traceable. Their
`git mv` provenance is preserved by the upcoming initial commit.

## Policy

- **Do not import** anything from `legacy/` in `latticetn/`, `scripts/`,
  `tests/`, or `examples/`.
- **Do not extend** these prototypes. New work belongs in `latticetn/`.
- If a prototype idea becomes real, reimplement it inside `latticetn/` /
  `tests/` / `examples/` with proper conventions (`S = sigma/2`,
  `torch.complex128`, autograd-clean loss paths) per `docs/AD_MAINLINE_POLICY.md`.

## Conventions reminder (these prototypes may NOT follow them)

The active codebase uses `S = sigma/2`, open boundary, `torch.complex128`,
index order `(left, phys, right)` for MPS and `(left, right, phys_in,
phys_out)` for MPO. Archived prototypes predate and may violate these
conventions — treat their physics with caution.
