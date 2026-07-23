# NUMERICAL_REPORT.md

Numerical validation of the latticeTN autograd tensor-network library against
exact diagonalization (ED) for the finite open-boundary 1D spin-1/2
antiferromagnetic Heisenberg chain.

## Physics convention

```
H = J * sum_{i=0}^{N-2} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
S = sigma / 2        (spin convention, NOT Pauli)
J = 1.0
boundary = open
dtype   = torch.complex128
device  = cpu
```

For open spin-1/2 chains the maximum MPS bond dimension needed to represent a
state exactly is `2**(N//2)`; with `chi >= 2**(N//2)` the variational ansatz is
exact-representable, so the variational energy can reach the exact ground energy
`E0` to machine precision.

Reference ground-state energies (dense ED via `numpy.linalg.eigh`):

| N  | exact E0 (ED)   | E0 / N        | Bethe E0/N (thermodynamic) |
|----|-----------------|---------------|----------------------------|
| 2  | -0.7500000000   | -0.3750000000 | -0.4431471806              |
| 4  | -1.6160254038   | -0.4040063509 | -0.4431471806              |
| 6  | -2.4935771339   | -0.4155961890 | -0.4431471806              |
| 8  | -3.3749325987   | -0.4218665748 | -0.4431471806              |

E0/N approaches the Bethe-ansatz value `1/4 - ln(2)` from above, as expected.

## Verification pipeline

The MPS-MPO energy path (`latticetn.MPS.energy_with_MPO`) computes the Rayleigh
quotient `<psi|H|psi> / <psi|psi>` and is fully autograd-differentiable (no
`.detach()/.data/.item()/no_grad` inside the energy computation). It is checked
against the dense formula `psi.conj() @ H @ psi / (psi.conj() @ psi)` and matches
to machine precision (see `tests/test_energy_rayleigh.py`,
`tests/test_heisenberg_energy_dense_compare.py`). The Heisenberg and TFI MPO
`.to_dense()` match the dense Hamiltonians exactly
(`tests/test_heisenberg_mpo_dense.py`, `tests/test_tfi_mpo_dense.py`).

## Variational validation table

Solver: `scripts/run_heisenberg_small.py` (Adam on the Rayleigh energy,
in-place Frobenius normalization performed OUTSIDE the autograd energy path,
fixed seed, CPU, complex128). Variational principle verified: `final_E >= E0`
within tolerance (`below_ground = false`) in every case. The variational energy
never falls below the exact ground energy.

| N | chi | exact ground E0 | variational initial E | variational final E | absolute error | relative error | command                                                                                                       | pass/fail |
|---|-----|-----------------|-----------------------|---------------------|----------------|----------------|---------------------------------------------------------------------------------------------------------------|-----------|
| 2 | 2   | -0.7500000000   | -0.0226460898         | -0.7499999998       | 1.94e-10       | 2.59e-10       | `python scripts/run_heisenberg_small.py --N 2 --chi 2 --steps 200 --lr 1e-2 --seed 0 --device cpu`           | PASS      |
| 4 | 4   | -1.6160254038   | -0.3684620285         | -1.6160254038       | 6.44e-14       | 3.98e-14       | `python scripts/run_heisenberg_small.py --N 4 --chi 4 --steps 300 --lr 1e-2 --seed 0 --device cpu`           | PASS      |
| 6 | 8   | -2.4935771339   | -0.3337416240         | -2.4935554092       | 2.17e-05       | 8.71e-06       | `python scripts/run_heisenberg_small.py --N 6 --chi 8 --steps 300 --lr 1e-2 --seed 0 --device cpu`           | PASS      |

Notes:
- N=4, chi=4 reaches `E0` to ~1e-13 (chi = 2^(N/2) is exact-representable and
  300 Adam steps fully converge).
- N=6, chi=8 reaches `E0` to ~2.2e-5 in 300 steps (chi = 2^(N/2) is
  exact-representable; the residual is optimizer convergence, not an
  expressivity gap, and tightens with more steps).

## Test suite (`scripts/validation_score.py`)

All required fast tests pass (exit 0):

```
python scripts/validation_score.py --fast     # Score: PASS  (exit 0)
pytest -q tests/test_reference_models.py tests/test_mps_dense.py tests/test_mpo_dense.py \
       tests/test_tfi_mpo_dense.py tests/test_energy_rayleigh.py \
       tests/test_heisenberg_mpo_dense.py tests/test_heisenberg_energy_dense_compare.py \
       tests/test_heisenberg_variational_smoke.py
# 69 passed
```

`--full` additionally runs the small Heisenberg solve:

```
python scripts/validation_score.py --full    # exit 0
# -> python scripts/run_heisenberg_small.py --N 6 --chi 8 --steps 300 --lr 1e-2 --seed 0 --device cpu
```

## Known limitations

- **System size**: validated only for N <= 8 (ED) and N <= 6 (variational) on
  CPU. Larger N requires denser ED memory (2^N) and longer/2-site optimization.
- **Ground-state optimizer**: the original validation table uses single-site
  full-tensor Adam with Frobenius re-normalization. Classical DMRG, two-site AD,
  and traditional TDVP now exist as separate modules but do not alter that
  original AD validation path.
- **Boundary**: only open boundary conditions. Periodic boundary MPS/MPO is not
  in the validation path (prototype code in `AD_MPS*.py` is preserved but not
  validated).
- **dtype/device**: scientific validation defaults to complex128/CPU. TDVP uses
  device-preserving tensor contractions and has an opt-in CUDA parity test;
  default tests never launch GPU work.
- **Models**: Heisenberg (XXX) and TFI MPOs are validated against dense
  references. XXZ / longer-range / higher-spin models are out of scope for this
  stage.
- **TFI convention**: `H_TFI = -J Sz Sz - h sum_i Sx` (spin convention), chosen
  for consistency with the Heisenberg spin convention. The prototype
  `AD_MPS_fixed.generate_TFI_MPO` uses Pauli matrices; the two are not
  interchangeable (see `docs/CLAUDE_PROGRESS.md`).

## Extending to larger systems / other models

- **Larger N / DMRG**: implement a 2-site sweep with SVD-based bond truncation in
  `latticetn/` (reuse `MPS.to_dense`, `MPO.to_dense`, `energy_with_MPO` as the
  building blocks). Add a fixed-chi truncation and a Davidson/Lanczos local
  eigensolver on the 2-site effective object. This keeps `chi << 2^(N/2)`
  tractable for N up to hundreds.
- **XXZ**: add `generate_xxz(Jz, Jxy)` to `latticetn/mpo.py` by reusing the same
  D=5 nearest-neighbor automaton with anisotropic `Sz Sz` vs `Sx Sx + Sy Sy`
  coefficients. Validate against a dense `xxz_dense` in `operators.py`.
- **TFI ground-state scan**: reuse `generate_tfi` + the variational solver to
  scan `h` and locate the critical point `h_c = 0.5` (in the spin convention
  `H = -J Sz Sz - h Sx`, the critical transverse field is `h_c = J/2`).
- **Higher spin**: generalize `spin_operators` (already parameterized by
  dimension) and the MPO generator; the contraction code is spin-agnostic.
# Stage 12A canonical-output validation

The gauge-retracted Global AD extension is documented in
`docs/STAGE12A_GAUGE_RETRACTION_REPORT.md`. Its focused CPU tests verify exact
dense and charge-block state/energy invariance, unit final physical norm,
canonical residual, hard-sector preservation, explicit optimizer resets, and
raw-versus-physical norm metadata.

| Stage 12A check | Result | Command |
|---|---|---|
| Dense exact QR/SVD, periodic stability, and negative truncation control | PASS (5 passed) | `python -m pytest -q tests/test_stage12a_dense_canonical.py` |
| Spinless/Hubbard blockwise QR | PASS (3 passed) | `python -m pytest -q tests/test_stage12a_sector_canonical.py` |
| Global AD cadence/reset/output metadata and hard-sector dense-QR rejection | PASS (4 passed) | `python -m pytest -q tests/test_stage12a_runner.py` |
| Repository fast validation | PASS | `python scripts/validation_score.py --fast` |

# Stage 12B traditional TDVP validation

Stage 12B adds a classical projector-splitting TDVP baseline, separate from the
future AD-TDVP Stage 12C. One-site evolution keeps fixed bond dimensions;
two-site evolution grows/truncates bonds through an adaptive SVD. Both use
native MPO environments and a matrix-free Hermitian Lanczos exponential action.

| Stage 12B check | Result | Command |
|---|---|---|
| Krylov and effective-Hamiltonian actions | PASS | `python -m pytest -q tests/test_tdvp_krylov.py tests/test_tdvp_effective_hamiltonian.py` |
| One-site norm, energy, gauge, fixed chi, N=8 ED fidelity/local Sz | PASS | `python -m pytest -q tests/test_tdvp_one_site.py` |
| Two-site adaptive chi, norm/energy, N=8 ED fidelity, quench observables | PASS | `python -m pytest -q tests/test_tdvp_two_site.py` |
| N=8 Heisenberg Neel quench | PASS: norm drift `8.9e-16`, energy drift `5.4e-10`, fidelity `0.999999993776` at t=0.2 for chi_max=8 | `python scripts/run_tdvp_heisenberg_quench.py --N 8 --dt 0.02 --steps 10 --chi-max 8 --truncation-tol 1e-10 --device cpu` |

See `docs/STAGE12B_TDVP_REPORT.md` for the algorithm, complete validation table,
API example, and limitations.

# Stage 12A-P0 scientific compliance validation

The eight P0 findings from the audit of commit `9d4c857` were reproduced and
repaired. This gate covers symmetry-sector degeneracy and graph reachability,
operator-square soft penalties, requested-chi Global AD initialization,
local-dimension geometry and RNG behavior, normalized native observables,
best-energy/best-state identity, and strict invalid-number handling.

| Compliance check | Result | Command |
|---|---|---|
| Eight P0 red-line invariants, including Hubbard `N=20` at `chi=32,48,60,64,80,96` | PASS | `python -m pytest -q tests/test_p0_scientific_compliance.py` |
| Full CPU repository regression | PASS (100%) | `python -m pytest -q -p no:cacheprovider` |
| Formal fast validation including the P0 gate | PASS | `python scripts/validation_score.py --fast` |

Historical hard-sector and Global AD benchmark files are not retroactively
promoted. Their labeling/rerun policy and the finding-by-finding scientific
evidence are recorded in `docs/STAGE12A_P0_PHYSICS_COMPLIANCE.md`.
