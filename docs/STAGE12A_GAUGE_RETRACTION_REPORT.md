# Stage 12A: Gauge-Retracted AD and Canonical Normalization

## Theory statement

For an exact, non-truncating canonicalization map `C` and the map `pi` from an
MPS tensor representative to its physical state/ray,

```text
pi(C(A)) = pi(A).
```

An AD step first creates a new physical MPS state. Canonicalization is then run
under `torch.no_grad()` and changes only the gauge representative used by later
AD steps. This is **not** an AD/canonicalization commutation claim: Euclidean
AD depends on tensor coordinates and, in general,

```text
C(AD(A)) != AD(C(A)).
```

Stage 12B differentiable canonicalized-MPS optimization is explicitly deferred.

## Dense implementation

`latticetn/canonical.py` now exposes `left_canonicalize`,
`right_canonicalize`, `mixed_canonicalize`, `normalize_center`, and
`canonical_residual`. Frequent retraction uses exact reduced QR. An exact,
non-truncating SVD left sweep is available for prototype/diagnostic comparison.
Truncated `svd_compress` remains a separate physical compression operation.

Final normalization first establishes mixed canonical form and then scales only
the center tensor. Exact gauge retraction preserves the vector; center scaling
selects the unit representative of the same physical ray.

## Hard-sector implementation

`latticetn/charge_sectors.py` implements blockwise left and mixed QR sweeps.
QR blocks are keyed by the virtual charge and never mix different charges:

```text
spinless: q_right = q_left + n_s
Hubbard:  q_right = q_left + (n_up[s], n_down[s])
```

The block-diagonal QR residual is absorbed into the adjacent tensor. Masks are
reapplied after retraction. Dense QR is rejected for hard-sector Global AD;
the explicit mode is `projection="sector_canonical"`.

## Global AD integration

`MethodConfig`, the unified runner, and the legacy benchmark CLI expose:

```text
projection
canonical_interval
normalize_final_state
reset_optimizer_on_canonicalize
canonicalization_method
```

At each configured interval Global AD applies exact no-grad canonicalization,
normalizes the center, and rebuilds the optimizer when requested. Results record
projection events and optimizer-reset steps. Every accepted Global AD output is
finally canonicalized and normalized, even when the internal projection is
`none` or `tensor_norm`. Finalization verifies energy invariance, unit norm, and
hard-sector forbidden zeros before observables and JSON serialization.

Result metadata includes:

```text
projection
canonical_interval
canonicalization_method
raw_norm_before_projection
physical_norm_after_projection
canonical_residual
optimizer_reset_events
```

The repository currently has no reusable MPS checkpoint serializer; therefore
there is no legacy unnormalized checkpoint path to migrate. Any future
checkpoint implementation must pass through the same physical-output
finalization helper before writing tensors.

## Exact test outcomes

| Check | Outcome |
|---|---:|
| Stage 12A dense focused tests | 5 passed |
| Stage 12A spinless/Hubbard sector tests | 3 passed |
| Stage 12A runner/metadata tests | 4 passed |
| Combined canonical/runner CPU regression selection | 29 passed |
| CLI and hard-sector CPU regression selection | 6 passed |
| Existing Global AD optimizer/gauge regression selection | 13 passed |
| Broader CPU suite previously-failing selection after compatibility fixes | 8 passed |
| `scripts/validation_score.py --fast` | PASS (exit 0) |

Exact dense and sector tests use `torch.complex128`. State, energy, norm,
canonical residual, target charges, and forbidden masks are checked with
`1e-12`-scale tolerances. The truncated-SVD negative control changes state and
energy as expected.

The standalone evidence script is:

```text
standalone_stage12a_gauge_retraction_test.py
```

It compares pure Global AD, periodic QR retraction, and periodic exact SVD
retraction on dense TFI and reports energy, physical norm, canonical residual,
gradient norm, ED overlap, and phase-aligned ED state distance.

## Remaining limitations

- Stage 12A does not transport Adam/LBFGS state through a gauge change; reset is
  explicit and enabled by default for canonical retractions.
- Hard-sector canonicalization currently supports exact QR only. Exact SVD is a
  dense diagnostic mode, not a charge-block production mode.
- Canonicalization is outside autograd. Stage 12B is deferred.
- No large production benchmark was run as part of tests.
- The local Torch build reports CUDA availability but cannot execute kernels on
  the installed GPU (`cudaErrorNoKernelImageForDevice`); Stage 12A validation
  was therefore CPU-only as required.

## Recommended manual A/B benchmarks

Spinless `V=0`, `N=80`, `chi=32`:

```powershell
python scripts/run_ad_model_benchmark.py --model spinless_tv --method ad_global --N 80 --chi 32 --sweeps 20 --local-steps 10 --device cpu --dtype complex128 --init spinless_cdw --optimizer adam --lr 0.01 --target-n 40 --sector-mode hard --V 0 --stabilization tensor_norm --no-ed --output outputs/stage12a_spinless_tensor_norm.json
python scripts/run_ad_model_benchmark.py --model spinless_tv --method ad_global --N 80 --chi 32 --sweeps 20 --local-steps 10 --device cpu --dtype complex128 --init spinless_cdw --optimizer adam --lr 0.01 --target-n 40 --sector-mode hard --V 0 --stabilization sector_canonical --canonical-interval 10 --reset-optimizer-on-canonicalize --no-ed --output outputs/stage12a_spinless_sector_canonical.json
```

Hubbard `U=4`, `N=40`, `chi=32`:

```powershell
python scripts/run_ad_model_benchmark.py --model hubbard --method ad_global --N 40 --chi 32 --sweeps 20 --local-steps 10 --device cpu --dtype complex128 --init hubbard_neel --optimizer adam --lr 0.01 --target-nup 20 --target-ndown 20 --sector-mode hard --U 4 --stabilization tensor_norm --no-ed --output outputs/stage12a_hubbard_tensor_norm.json
python scripts/run_ad_model_benchmark.py --model hubbard --method ad_global --N 40 --chi 32 --sweeps 20 --local-steps 10 --device cpu --dtype complex128 --init hubbard_neel --optimizer adam --lr 0.01 --target-nup 20 --target-ndown 20 --sector-mode hard --U 4 --stabilization sector_canonical --canonical-interval 10 --reset-optimizer-on-canonicalize --no-ed --output outputs/stage12a_hubbard_sector_canonical.json
```

Compare energy, charge/spin gaps, density-stagger amplitude, particle-hole
symmetry, physical norm, canonical residual, runtime, and gradient norm. For
spinless `V=0`, also compare finite-`N` energy and charge gap with the exact
free-fermion reference.
