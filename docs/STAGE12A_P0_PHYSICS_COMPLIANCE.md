# Stage 12A-P0 Physics Compliance Repair

Date: 2026-07-23

Audited baseline: Git commit `9d4c857`

Scope: Stage 12A and earlier ground-state/MPS paths; TDVP is not part of this
audit.

## Decision

All eight reported P0 findings existed in the audited baseline. They were not
cosmetic defects: six had direct executable counterexamples, while best-state
and numerical-cleaning defects could silently associate results with the wrong
state or hide invalid arithmetic.

The active code paths are repaired and guarded by
`tests/test_p0_scientific_compliance.py`. The code freeze can be lifted for new
CPU-small validation work. Historical hard-sector and Global AD numerical data
remain frozen until rerun with the repaired representation; passing the new
tests does not retroactively change the ansatz used to generate old files.

## Scientific basis

An MPS bond of dimension `chi` represents independent Schmidt channels. For a
symmetry-adapted tensor, a virtual index is not merely a charge label: it has
the structure `(q, alpha)`, where `alpha=1,...,d_q` is the degeneracy space.
Singh, Pfeifer, and Vidal explicitly separate symmetric tensors into structural
and degeneracy tensors; the degeneracy tensor contains the variational degrees
of freedom. Their U(1) construction also establishes exact charge preservation
and fixed-particle-number selection. Schollwöck's MPS/DMRG review relates bond
indices to Schmidt states and the variational MPS manifold.

A soft fixed-sector penalty must be

```text
lambda * <(Q - Q0)^2>
= lambda * (Var(Q) + (<Q> - Q0)^2).
```

Squaring only the mean error omits `Var(Q)`. Consequently,
`(|00>+|11>)/sqrt(2)` has `<N>=1` and zero mean-error penalty at target one,
despite `Var(N)=1`. This is an algebraic identity, independent of an optimizer.
Penalty-method literature likewise distinguishes operator-square constraints
from penalties that do not rigorously select the desired symmetry sector.

An expectation value for a not-necessarily-normalized MPS is the Rayleigh
ratio `<psi|O|psi>/<psi|psi>`. It must be invariant under
`|psi> -> c|psi>`. Finally, PyTorch's numerical-accuracy guidance states that
linear-algebra backends do not guarantee useful behavior for NaN/Inf inputs and
recommends explicit finite checks. Roundoff-sized Hermiticity or positivity
violations may be corrected within a dtype-scaled tolerance; larger violations
must invalidate the run.

Primary/authoritative references:

- U. Schollwöck, *The density-matrix renormalization group in the age of matrix
  product states*, [arXiv:1008.3477](https://arxiv.org/abs/1008.3477).
- S. Singh, R. N. C. Pfeifer, G. Vidal, *Tensor network decompositions in the
  presence of a global symmetry*,
  [arXiv:0907.2994](https://arxiv.org/abs/0907.2994).
- S. Singh, R. N. C. Pfeifer, G. Vidal, *Tensor network states and algorithms
  in the presence of a global U(1) symmetry*,
  [arXiv:1008.4774](https://arxiv.org/abs/1008.4774).
- K. Kuroiwa and Y. O. Nakagawa, *Penalty methods for a variational quantum
  eigensolver*, [arXiv:2010.13951](https://arxiv.org/abs/2010.13951).
- PyTorch, [Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
  and [`torch.isfinite`](https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html).

## Finding-by-finding disposition

| P0 finding | Baseline evidence/root cause | Repair | Authoritative verification |
|---|---|---|---|
| Independently trimmed hard-sector graph | Each bond selected nearest charges independently, so locally present charges could lose a path to a boundary. Larger `chi` could therefore disconnect a block that happened to work at smaller `chi`. | Channel allocations are nested prefixes and are iteratively clipped against both left and right transition capacities until every retained channel has a full-chain path. | Hubbard `N=20`, `chi=32,48,60,64,80,96` all canonicalize; every charge multiplicity at larger `chi` contains the preceding one. |
| No charge degeneracy | `BondChargeSectors.dims` was unused; tensor shapes were `len(charges)`, giving one channel per charge. | Each bond stores unique `charges` plus `dims`; dense indices use `expanded_charges`. Masks allow every compatible pair of degeneracy channels, and block QR operates on all channels of a charge. | Center bond dimension equals requested `chi`; at least one `d_q>1`; tensor shape equals `sum(d_q)`. |
| Incorrect soft-sector penalty | Used `lambda*(<Q>-Q0)^2`; variance was absent. | Uses `lambda*(Var(Q)+(<Q>-Q0)^2)` separately for spinless `N`, Hubbard `N_up`, and Hubbard `N_down`. | The exact two-site superposition counterexample returns penalty `lambda`, not zero. |
| Global AD ignored requested `chi` | `initialization=auto` returned a bond-one product state, while Global AD cannot change Parameter shapes. | Non-hard Global AD auto initialization is random on the requested MPS geometry. Hard-sector initialization already constructs the requested charge-degenerate geometry. | `N=4, chi=4` reports initial bonds `[2,4,2]` and `initial_max_bond=4`. |
| Hubbard MPS used spin-1/2 geometry | Bond cap was hard-coded as `2**min(i,N-i)` for every local dimension. | Bond cap is `dim**min(i,N-i)`. | `N=4, dim=4, chi=64` has internal bonds `[4,16,4]`. |
| Native observables were numerators | Local/two-site/correlation helpers omitted division by the MPS norm. | Public `native_*expect` helpers divide by a strictly positive validated norm; raw MPO numerator retains its explicit name. | Local and two-site observables are invariant under a factor-seven MPS rescaling and match normalized dense references. |
| `best_energy` had no best state | Only a Python scalar and step were retained. Final observables used the last state. | Global and two-site AD snapshot the tensors whenever the best monitored energy improves, restore that MPS, canonicalize/normalize it, and verify that it reproduces the scalar. | `best_energy == final_energy == observable energy`; result metadata states `final_state_source="best_mps"`. |
| Invalid numbers were cleaned | NaN truncation could become zero; negative variance/norm was unconditionally clamped; complex energy discarded its imaginary component; selected diagnostics swallowed exceptions and returned NaN. | Central finite/Hermiticity/positivity/truncation guards correct only dtype-scaled roundoff. Significant negative, imaginary, NaN, Inf, zero-norm, or structural failures raise. Broad diagnostic exception swallowing was removed. | Red-line tests inject NaN weights, a `-1e-3` variance, and a `1e-2 i` energy component and require failure. |

## Hard-sector representation and monotonicity

For a cut after `i` sites, the maximum spinless degeneracy for charge `q` is
bounded by

```text
min(C(i,q), C(N-i, N_target-q)).
```

For Hubbard `(q_up,q_down)`, the corresponding left/right basis counts are
products of binomial coefficients. The implementation uses these exact
fixed-sector capacity bounds, selects at most `chi` `(q,alpha)` channels, and
then enforces the local inequalities

```text
d_i(q) <= sum_(q_left,s: q_left+q_s=q) d_(i-1)(q_left)
d_i(q) <= sum_(q_right,s: q+q_s=q_right) d_(i+1)(q_right).
```

Because channel candidates are selected by a nested prefix and the clipping map
is monotone, increasing `chi` cannot remove an existing charge/degeneracy
channel. This is the executable inclusion invariant used by the red-line test.

## Numerical-failure policy

The shared `latticetn.numerics` policy is:

```text
finite + within dtype-scaled roundoff:
    accept; correct a tiny negative real scalar to zero
NaN/Inf, non-positive norm, significant imaginary part,
significant negative variance/weight, or out-of-range truncation:
    raise FloatingPointError; mark the run invalid
```

The returned real tensors retain their autograd graph. Checks inspect values
but do not detach, replace, or re-create the differentiable result.

## Validation evidence

| Gate | Result | Command |
|---|---:|---|
| P0 red-line regressions | PASS | `python -m pytest -q tests/test_p0_scientific_compliance.py` |
| Focused charge/observable/runner/two-site regressions | PASS | See `docs/CLAUDE_PROGRESS.md`, checkpoint `Stage 12A-P0-01` |
| Repository CPU suite | PASS, 100% | `python -m pytest -q -p no:cacheprovider` |
| Formal fast validation | PASS | `python scripts/validation_score.py --fast` |

Default validation is CPU-only and uses `torch.complex128` for the scientific
red-line cases. No long training or GPU run was performed.

## Historical-result policy

- Existing `N=12, chi=32` Hubbard phase-diagram data may be described only as
  results of the historical one-channel-per-charge restricted ansatz. It is
  not evidence for conventional bond dimension 32 or monotone `chi`
  convergence.
- The historical `N=20, chi=64` job must not be resumed from the old
  representation. It requires a fresh run after the present repair.
- Any historical Global AD result with all initial/final bonds equal to one is
  a product-state-manifold optimization and must be rerun before scientific
  use.
- Historical files that report a scalar `best_energy` without a matching saved
  state must use their validated `final_energy` or be rerun. New runner output
  always associates the reported best scalar with the restored best MPS.
- No prior benchmark file is silently relabeled as repaired data.

## Remaining limitations

- The implementation remains a dense masked representation, not a
  memory-saving block-sparse tensor backend. It now has the correct
  `(q,alpha)` variational structure but does not receive block-sparse speedups.
- Hard-sector two-site AD splitting is still unsupported; hard-sector runs use
  masked Global AD and charge-block QR.
- Large production phase diagrams and convergence studies are intentionally not
  part of CPU tests. They require fresh, separately budgeted runs.
