# AD real-model benchmark protocol

## Purpose

This stage benchmarks the latticeTN AD-mainline solver on finite open-boundary 1D MPO models beyond tiny exact-diagonalization cases.

The solver under test is **two-site AD local-tensor optimization**:

```text
MPS + MPO
-> two-site mixed-canonical center Theta
-> differentiable local Rayleigh quotient E(Theta)
-> loss.backward() + torch optimizer step
-> SVD split/compression after the optimizer step
-> sweep left-to-right / right-to-left
```

Classical DMRG, Lanczos, and dense local eigensolvers are reference baselines only. They are not part of this benchmark runner.

## Supported models

### Heisenberg chain

```text
H = J * sum_i S_i . S_{i+1}
S = sigma / 2
boundary = open
local dim = 2
recommended init = neel
```

Useful thermodynamic-limit reference:

```text
e_inf = 1/4 - ln(2) = -0.4431471805599453
```

For finite open chains, `E/N` is not expected to equal `e_inf` exactly.

### Transverse-field Ising chain

```text
H = -J * sum_i Sz_i Sz_{i+1} - h * sum_i Sx_i
boundary = open
local dim = 2
recommended init = all_up or neel
```

### Spinless fermion t-V chain

```text
H = -t * sum_i (c^dag_i c_{i+1} + h.c.)
    + V * sum_i (n_i - 1/2)(n_{i+1} - 1/2)
    - mu * sum_i (n_i - 1/2)
boundary = open
local dim = 2
recommended init = cdw, i.e. 1010...
```

### Spinful Hubbard chain

```text
H = -t * sum_{i,sigma} (c^dag_{i,sigma} c_{i+1,sigma} + h.c.)
    + U * sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2)
    - mu * sum_i (n_{i,up} + n_{i,down} - 1)
    - field * sum_i (n_{i,up} - n_{i,down})
boundary = open
local dim = 4
basis = |0>, |up>, |down>, |up,down>
recommended init = hubbard_neel, i.e. |up>, |down>, |up>, |down>, ...
```

## Runner

```bash
python scripts/run_ad_model_benchmark.py --help
```

The runner must report:

- model and parameters;
- N, chi, sweeps;
- device, dtype, optimizer settings;
- initial energy;
- per-sweep energy, E/N, delta energy, max truncation, max gradient norm, max bond, runtime;
- final JSON output;
- explicit `ed_skipped = true`;
- explicit `dmrg_lanczos_used = false`.

## CPU smoke commands

```bash
python scripts/run_ad_model_benchmark.py --model heisenberg --N 8 --chi 8 --sweeps 2 --device cpu --dtype complex128 --optimizer lbfgs --local-steps 1 --lbfgs-iters 4 --output docs/tmp_ad_heisenberg_n8.json
python scripts/run_ad_model_benchmark.py --model hubbard --N 4 --chi 8 --sweeps 1 --device cpu --dtype complex128 --optimizer lbfgs --local-steps 1 --lbfgs-iters 3 --output docs/tmp_ad_hubbard_n4.json
```

## PowerShell GPU benchmark command

```powershell
$env:PYTHONPATH="."
python .\scripts\run_ad_model_benchmark.py `
  --model heisenberg `
  --N 80 `
  --chi 64 `
  --sweeps 6 `
  --device cuda `
  --dtype complex64 `
  --init auto `
  --optimizer lbfgs `
  --local-steps 1 `
  --lbfgs-iters 10 `
  --lr 1.0 `
  --stabilization tensor_norm `
  --output docs\ad_heisenberg_N80_chi64.json
```

Expected sanity range from a previous Tesla V100 run:

```text
N=80, chi=64, complex64, 6 directional sweeps
E/site around -0.4408
ED skipped
classical DMRG/Lanczos not used
```

## Benchmark table plan

Recommended first table:

| model | N | chi | sweeps | optimizer | dtype | E/N | runtime | notes |
|---|---:|---:|---:|---|---|---:|---:|---|
| heisenberg | 80 | 32 | 6 | LBFGS | complex64 | ... | ... | OBC |
| heisenberg | 80 | 64 | 6 | LBFGS | complex64 | ... | ... | OBC |
| hubbard U=4 | 20 | 32 | 6 | LBFGS | complex64 | ... | ... | half-filled init |
| hubbard U=4 | 20 | 64 | 6 | LBFGS | complex64 | ... | ... | half-filled init |

## Acceptance criteria

- CPU helper tests pass.
- Small CPU benchmark commands finish and write JSON.
- The runner has no dependency on `dmrg.py` or `lanczos.py`.
- Large-N commands never build dense Hamiltonians.
- GPU runs are optional and are not part of the default test suite.
- Documentation records commands, hardware, final energies, and known limitations.

## Known limitations

- No quantum-number/block-sparse tensors yet.
- No PBC/cylinder/2D support in this stage.
- No finite-temperature or spectral functions.
- Energy-density comparison to thermodynamic references must account for finite open-boundary effects.
- AD optimization settings are not yet fully tuned across all models.
