# Project Direction (latticeTN)

## North star

latticeTN is an **automatic-differentiation tensor network** project (PyTorch).
The scientific target (`CLAUDE.md`) is to solve the finite open-boundary 1D
spin-1/2 Heisenberg chain with **MPS + MPO + PyTorch autograd**, verified against
exact diagonalization.

## What is the mainline

The solver is **AD-MPS variational optimization**: trainable MPS parameters →
differentiable Rayleigh quotient `E = <psi|H|psi>/<psi|psi>` → `loss.backward()`
→ torch optimizer. See `docs/AD_MAINLINE_POLICY.md` for the full policy.

## Stages so far

| Stage | Status | Role |
|---|---|---|
| 1 | done | MPS + MPO + autograd Heisenberg validation vs ED (CPU) |
| 2 | done | Reproducible Heisenberg benchmark (observables, entropy, chi sweep) |
| 2.5 | done | Opt-in GPU device-parity smoke (matched-name GPU, never cuda:0) |
| 3A | done | MPS canonicalization + SVD compression (gauge/projection/diagnostic tools) |
| 3B | done | Native (no-to_dense) MPS/MPO contractions — the differentiable scalable path |
| 4A | done | Two-site DMRG primitives (classical, reference only) |
| 4B | done | Scalable DMRG: matrix-free H_eff + Lanczos (classical reference only) |
| 4R | done | **AD-MPS variational solver — the autograd mainline (realign)** |
| 5A | done | Gauge-stabilized AD-MPS (canonical/tensor_norm/none projection) |

## Forward direction

Continue AD-local-tensor optimization, NOT more classical DMRG/Lanczos:
gauge-projected AD, optional bond-growing AD (using SVD compression as a
post-step projection), mixed-canonical projection with a moving orthogonality
center. Classical solvers remain reference oracles only.

## Non-goals (explicitly out of scope)

TEBD, TDVP, finite-temperature methods, formal GPU performance benchmarking,
large-N/long-training jobs, and any "DMRG/Lanczos as the mainline" drift.
