# Stage 8: Fermion Fixed-Sector Diagnostics and AD Benchmarking

Stage 8 adds dense-MPS support for fermion charge metadata, fixed-sector
product-state initialization, particle-number diagnostics, optional soft sector
penalties, and a generalized AD benchmark runner for open-boundary 1D models.

## What This Stage Adds

- `latticetn.charges`: spinless and Hubbard local-basis charge metadata.
- `latticetn.initial_states`: bond-dimension-one product MPS initializers for
  spin, spinless fermion, and Hubbard fixed sectors.
- `latticetn.sector_observables`: additive diagonal observables such as total
  particle number, Hubbard `N_up`, `N_down`, `N_tot`, `S_z`, variances, and
  leakage reports.
- `scripts/run_ad_model_benchmark.py`: an AD-mainline benchmark runner for
  `heisenberg`, `tfi`, `spinless_tv`, and `hubbard`.

## What This Stage Does Not Do

- No TDVP implementation.
- No block-sparse U(1) MPS/MPO tensors.
- No quantum-number-conserving tensor storage.
- No exact diagonalization in the large-system AD benchmark runner.
- No classical DMRG or Lanczos dependency in the Stage 8 benchmark runner.

Dense fixed-sector monitoring means the initial state and diagnostics know
about the desired sector, but the dense tensors are still free to represent
superpositions outside that sector during AD. A true block-sparse U(1)
implementation would enforce the sector structurally in tensor storage and
contractions; that is intentionally left for a later stage.

## CPU Examples

PowerShell:

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
  --no-ed
```

Bash:

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
  --no-ed
```

## GPU/Auto Examples

PowerShell:

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
  --no-ed
```

Bash:

```bash
PYTHONPATH=. python scripts/run_ad_model_benchmark.py \
  --model spinless_tv \
  --N 80 \
  --chi 64 \
  --sweeps 6 \
  --device auto \
  --dtype complex64 \
  --init spinless_cdw \
  --optimizer lbfgs \
  --local-steps 1 \
  --lbfgs-iters 10 \
  --lr 1.0 \
  --target-n 40 \
  --no-ed
```

PowerShell Hubbard auto example:

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
  --no-ed
```

Bash Hubbard auto example:

```bash
PYTHONPATH=. python scripts/run_ad_model_benchmark.py \
  --model hubbard \
  --N 40 \
  --chi 64 \
  --sweeps 6 \
  --device auto \
  --dtype complex64 \
  --init hubbard_neel \
  --optimizer lbfgs \
  --local-steps 1 \
  --lbfgs-iters 10 \
  --lr 1.0 \
  --target-nup 20 \
  --target-ndown 20 \
  --no-ed
```

`--device auto` selects CUDA when available and otherwise falls back to CPU. If
CUDA is selected, the runner prints the detected GPU name without assuming any
specific model.

## Sector Penalty

Spinless:

```text
loss = energy + lambda_n * (<N> - N_target)^2
```

Hubbard:

```text
loss = energy
     + lambda_nup * (<N_up> - N_up_target)^2
     + lambda_ndown * (<N_down> - N_down_target)^2
```

The penalty coefficients default to zero. With zero penalty, the runner uses
the existing two-site AD mainline. When a penalty is enabled, it uses a global
AD loss so the sector term participates directly in backpropagation.

## Benchmark Runner Policy

The Stage 8 benchmark runner prints and records:

- `ED status = skipped by design`
- `classical DMRG/Lanczos = not used`
- `dense_hamiltonian_built = false`
- `dmrg_lanczos_used = false`

The runner imports neither `dmrg` nor `lanczos`, and it does not call dense
Hamiltonian builders. ED remains available only in the small-system reference
tests elsewhere in the project.
