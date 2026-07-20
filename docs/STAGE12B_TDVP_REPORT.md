# Stage 12B — Traditional TDVP Report

## Scope and algorithm identity

Stage 12B implements a classical, non-autograd TDVP baseline for finite
open-boundary MPS. It does **not** implement AD-TDVP; that remains Stage 12C.

The implementation solves the projected real-time equation

```text
d|psi(A)>/dt = -i P_T H |psi(A)>
```

with a symmetric projector-splitting integrator:

- one-site TDVP evolves center tensors forward and zero-site bond matrices
  backward while moving the mixed-canonical center by QR/LQ;
- two-site TDVP evolves two-site blocks forward and overlapping one-site
  centers backward, then uses SVD to grow or truncate the bond;
- every local exponential uses a matrix-free Hermitian Lanczos action;
- MPO environments are contracted natively; no dense many-body Hamiltonian is
  constructed in the TDVP algorithm;
- CPU and CUDA tensors use the same code path. CUDA validation is opt-in under
  the repository GPU-selection policy.

## Public API

```python
from latticetn.tdvp import TDVP

solver = TDVP(
    mps,
    mpo,
    dt=0.01,
    method="two_site",       # or "one_site"
    max_bond_dim=16,
    truncation_tol=1e-10,
    device="cpu",           # "cuda:N" is supported
)
result = solver.evolve(steps=100, observables={"name": callback})
```

`TDVPResult` contains the evolved MPS, times, energy and norm histories,
observable histories, and per-step two-site truncation/bond diagnostics.

## Scientific validation

Physics convention throughout:

```text
H = sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
S = sigma / 2, J = 1, open boundary, complex128
```

| Requirement | Evidence | Result |
|---|---|---|
| Lanczos exponential action | 12-dimensional Hermitian matrix vs `torch.matrix_exp` | PASS, max error below `2e-12` |
| One-site norm conservation | N=6, dt=0.01, 10 steps, full fixed chi | PASS, drift below `1e-12` |
| One-site energy conservation | same run | PASS, drift below `1e-11` |
| One-site ED state/local observable | N=8 Neel, t=0.05, chi=16 | PASS, fidelity above `1-1e-11`, local Sz error below `1e-11` |
| One-site fixed bond dimension | N=6 | PASS, all tensor shapes unchanged |
| Two-site adaptive bond growth | N=8 product MPS | PASS, chi grows from 1 and respects `chi_max` |
| Two-site ED evolution | N=8 Neel, t=0.05, chi_max=16 | PASS, fidelity above `1-1e-11` |
| Two-site norm/energy conservation | same full-chi run | PASS, norm drift below `1e-12`, energy drift below `1e-11` |
| Truncated physical quench | N=8, dt=0.02, t=0.2, chi_max=4 | PASS, norm drift `8.9e-16`, energy drift `5.4e-10` |
| Magnetization dynamics | same quench | PASS, midpoint Sz `0.5 -> 0.4802646` |
| Entanglement growth | same quench | PASS, midpoint entropy `0 -> 0.0554012` nats |
| CUDA code path | opt-in `tests/test_tdvp_gpu.py` | clean skip unless an approved GPU is selected |

Commands:

```bash
python -m pytest -q tests/test_tdvp_krylov.py \
  tests/test_tdvp_effective_hamiltonian.py \
  tests/test_tdvp_one_site.py tests/test_tdvp_two_site.py
python scripts/run_tdvp_heisenberg_quench.py --N 8 --dt 0.02 \
  --steps 10 --chi-max 8 --truncation-tol 1e-10 --device cpu
python scripts/validation_score.py --fast
```

## Limitations

- One-site TDVP cannot grow bonds. A bond-one Neel input therefore remains on
  the product-state manifold; ED validation embeds the same state into the
  full fixed-chi manifold with `canonical.from_dense`.
- Two-site truncation is controlled by relative discarded singular-value
  weight per update plus a hard `max_bond_dim`; truncation makes the step
  non-unitary, so the completed symmetric step explicitly selects the unit-
  norm representative.
- The implementation targets time-independent finite open-boundary MPOs.
  Time-dependent MPOs, periodic boundaries, symmetry-block sparse TDVP, and
  imaginary-time TDVP are not part of Stage 12B.
- ED comparisons are deliberately limited to N=8 in default CPU tests.

## Stage boundary

This module is deliberately separated into the public evolution driver,
effective-Hamiltonian contractions, and Krylov action. Stage 12C can reuse the
interface and contractions while replacing the classical tangent evolution
with an explicitly differentiable AD-TDVP research path.
