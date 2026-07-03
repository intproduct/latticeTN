# Stage 9: Charge-Aware Fixed-Sector AD and Hard U(1) Constraints

Stage 9 upgrades the Stage 8 soft fixed-sector workflow to a correctness-first
hard-sector path. Tensors are still ordinary dense PyTorch tensors, but each MPS
site tensor now has a charge mask derived from virtual bond charge metadata.
Forbidden tensor entries are forced to zero during initialization, after
backward, and after optimizer steps.

## What Stage 9 Adds

- `latticetn.charge_sectors`
  - spinless U(1) virtual bond sectors
  - Hubbard U(1) x U(1) virtual bond sectors
  - spinless and Hubbard tensor charge masks
  - `ChargeAwareMPS` metadata wrapper
  - mask application, forbidden-gradient zeroing, and forbidden-entry checks
  - hard-sector product MPS initializers
- `scripts/run_ad_model_benchmark.py --sector-mode none|soft|hard`
  - `none`: no sector penalty or hard mask
  - `soft`: Stage 8 soft expectation-value penalty
  - `hard`: Stage 9 charge masks with masked AD updates

## Stage 8 vs Stage 9

Stage 8:

- dense MPS
- fixed-sector product initialization
- sector diagnostics
- optional soft penalty such as `lambda_n * (<N> - N_target)^2`

Stage 9:

- dense MPS plus virtual charge metadata
- hard boolean masks on every MPS tensor
- forbidden values zeroed after initialization and optimizer steps
- forbidden gradients zeroed after backward
- diagnostics for `max_forbidden_abs` and `max_forbidden_grad_abs`

The hard mask prevents represented basis amplitudes from leaving the target
sector. This is stronger than a soft penalty, which only discourages leakage in
the loss.

## Charge Rules

Spinless fermion U(1):

```text
q(local_state) = n in {0, 1}
Q_right = Q_left + n[s]
```

For target total particle number `N_target`, bond charges are cumulative left
particle numbers. The left boundary is `0`; the right boundary is `N_target`.

Hubbard U(1) x U(1):

```text
q(local_state) = (n_up, n_down)
Q_up_right   = Q_up_left   + n_up[s]
Q_down_right = Q_down_left + n_down[s]
```

The local basis remains:

```text
0 -> |0>
1 -> |up>
2 -> |down>
3 -> |up,down>
```

The left boundary is `(0, 0)`; the right boundary is
`(target_nup, target_ndown)`.

## Current AD Implementation

The Stage 9 hard-sector runner uses a masked global AD fallback:

1. Build a `ChargeAwareMPS`.
2. Apply charge masks before optimization.
3. Compute the differentiable Rayleigh quotient through the existing native
   MPS/MPO contraction.
4. Run `loss.backward()`.
5. Zero forbidden gradients.
6. Take an optimizer step.
7. Reapply charge masks.
8. Report sector diagnostics and forbidden-entry diagnostics.

This preserves the hard sector for the current dense MPS tensors.

## Current Limitation

Stage 9 does not yet implement blockwise charge-preserving two-site SVD. It also
does not provide a high-performance block-sparse tensor backend. If a future
stage returns to two-site hard-sector updates, the split should be upgraded to a
blockwise SVD by intermediate charge sector, then reassembled into dense tensors
or migrated to a true block-sparse layout.

## Runner Examples

PowerShell spinless CPU smoke:

```powershell
$env:PYTHONPATH="."
python .\scripts\run_ad_model_benchmark.py `
  --model spinless_tv `
  --N 4 `
  --chi 4 `
  --sweeps 1 `
  --device cpu `
  --dtype complex128 `
  --init spinless_cdw `
  --optimizer adam `
  --local-steps 1 `
  --lr 0.01 `
  --target-n 2 `
  --sector-mode hard `
  --no-ed
```

PowerShell Hubbard CPU smoke:

```powershell
$env:PYTHONPATH="."
python .\scripts\run_ad_model_benchmark.py `
  --model hubbard `
  --N 4 `
  --chi 4 `
  --sweeps 1 `
  --device cpu `
  --dtype complex128 `
  --init hubbard_neel `
  --optimizer adam `
  --local-steps 1 `
  --lr 0.01 `
  --target-nup 2 `
  --target-ndown 2 `
  --sector-mode hard `
  --no-ed
```

PowerShell spinless GPU/auto benchmark:

```powershell
$env:PYTHONPATH="."
python .\scripts\run_ad_model_benchmark.py `
  --model spinless_tv `
  --N 80 `
  --chi 64 `
  --sweeps 6 `
  --device auto `
  --dtype complex64 `
  --init spinless_cdw `
  --optimizer lbfgs `
  --local-steps 1 `
  --lbfgs-iters 10 `
  --lr 1.0 `
  --target-n 40 `
  --sector-mode hard `
  --no-ed
```

PowerShell Hubbard GPU/auto benchmark:

```powershell
$env:PYTHONPATH="."
python .\scripts\run_ad_model_benchmark.py `
  --model hubbard `
  --N 40 `
  --chi 64 `
  --sweeps 6 `
  --device auto `
  --dtype complex64 `
  --init hubbard_neel `
  --optimizer lbfgs `
  --local-steps 1 `
  --lbfgs-iters 10 `
  --lr 1.0 `
  --target-nup 20 `
  --target-ndown 20 `
  --sector-mode hard `
  --no-ed
```

Bash spinless CPU smoke:

```bash
PYTHONPATH=. python scripts/run_ad_model_benchmark.py \
  --model spinless_tv \
  --N 4 \
  --chi 4 \
  --sweeps 1 \
  --device cpu \
  --dtype complex128 \
  --init spinless_cdw \
  --optimizer adam \
  --local-steps 1 \
  --lr 0.01 \
  --target-n 2 \
  --sector-mode hard \
  --no-ed
```

Bash Hubbard CPU smoke:

```bash
PYTHONPATH=. python scripts/run_ad_model_benchmark.py \
  --model hubbard \
  --N 4 \
  --chi 4 \
  --sweeps 1 \
  --device cpu \
  --dtype complex128 \
  --init hubbard_neel \
  --optimizer adam \
  --local-steps 1 \
  --lr 0.01 \
  --target-nup 2 \
  --target-ndown 2 \
  --sector-mode hard \
  --no-ed
```

## Benchmark Policy

The Stage 9 runner continues the Stage 8 large-system policy:

- ED status is `skipped by design`.
- Classical DMRG is not used.
- Lanczos is not used.
- Dense Hamiltonians are not constructed.
- Device selection is dynamic through `--device auto|cpu|cuda`.
- CUDA tests cleanly skip when CUDA is not available.
