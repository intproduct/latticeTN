# Stage 5A Gauge-Stabilized AD-MPS Specification

## Goal

Upgrade the Stage 4R AD-MPS variational solver with **projection / gauge
stabilization** applied after each optimizer step (outside the autograd loss
graph). The main loss remains the differentiable Rayleigh quotient
`rayleigh_energy_native(mps, mpo)`.

No Lanczos/DMRG expansion, no TEBD/TDVP, no GPU performance benchmark.
DMRG/Lanczos are reference baselines only and never enter the AD path.

## Physics conventions (unchanged)

- `H = J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})`
- `S = sigma / 2`, `J = 1.0`, open boundary, `torch.complex128`, CPU.

## Required capabilities (in `latticetn/ad_variational.py`)

1. `train_ad_mps(..., projection="none"|"tensor_norm"|"canonical")`.
2. `tensor_norm` — per-tensor L2 renormalization (Stage 4R behavior).
3. `canonical` — left-canonical QR projection (Stage 3A `left_canonical`)
   written back onto trainable `.data` under `no_grad`.
4. After projection, MPS tensors remain trainable leaf parameters.
5. Loss path untouched: no projection / detach / .data / no_grad / unnecessary
   item in the loss; projection runs only after `optimizer.step()`.
6. History records: energy, grad_norm, state_norm, canonical_error, projection.
7. Generate `docs/AD_GAUGE_REPORT.md` comparing the three projections.

## Test requirements

- Loss scalar / finite / requires_grad; `backward()` -> all grads not-None + finite.
- AST/structural: loss path free of dmrg/lanczos/projection/detach/.data/no_grad/item.
- canonical projection: dense-state fidelity ~1 OR Rayleigh energy invariant;
  canonical error decreases.
- N=4/6 Adam training: energy decreases; not below exact ground beyond tolerance.
- canonical final energy not materially worse than `tensor_norm` baseline
  (expected to be at least as good; gauge stabilization improves conditioning).
- CPU-only, small systems, fast. Conventions unchanged.

## Constraints

- Stage 1/2/3A/3B/4A/4B/4R thresholds not relaxed; existing interfaces not broken.
- `energy_with_MPO` / `rayleigh_energy_native` not modified.
- No large dependencies; no long training; GPU tests out of the default path.
