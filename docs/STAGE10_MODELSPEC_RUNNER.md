# Stage 10: ModelSpec, MPO Builder, and Runner API Contract

Stage 10 defines a stable backend contract for structured 1D Hamiltonian jobs.
It prepares the core package for a future API server or frontend without
building either of those layers yet.

Stage 10 does not implement:

- a frontend page
- TDVP
- natural-language Hamiltonian parsing

The contract is:

```text
ModelSpec -> MPO
MethodConfig -> algorithm call
RuntimeConfig -> device/dtype/logging
ObservableSpec -> measurements
ResultSchema -> JSON output
```

## ModelSpec

`latticetn.model_spec.ModelSpec` describes a 1D Hamiltonian with structured
data:

```python
ModelSpec(
    name="hubbard",
    N=4,
    local_basis="hubbard",
    boundary="obc",
    parameters={"t": 1.0, "U": 4.0, "mu": 0.0, "h": 0.0},
    sector={"mode": "hard", "target_nup": 2, "target_ndown": 2},
)
```

Terms use `TermSpec` and `OperatorRef`. Stage 10 accepts structured term
patterns such as `onsite`, `nearest_neighbor`, `nearest_neighbor_hopping`, and
`two_site`. Fully generic custom-term MPO generation is intentionally limited;
unsupported terms fail with clear errors rather than silently mapping to another
model.

## Preset Registry

`latticetn.model_registry` exposes schemas for:

- `heisenberg`
- `tfi`
- `spinless_tv`
- `hubbard`
- `xxz`

`xxz` is registered as experimental/not implemented. The registry API includes:

```python
get_model_schema(model_id)
build_model_spec(model_id, N, parameters, boundary="obc", sector=None)
build_mpo_from_model_spec(model_spec, dtype, device)
```

This is the future backend source for an `/api/models` endpoint.

## Hamiltonian Builder

`latticetn.hamiltonian_builder.build_mpo(spec, dtype, device)` is the unified
Hamiltonian-to-MPO entry point. Preset models dispatch to the existing validated
hand-written generators:

- `MPO.generate_heisenberg`
- `MPO.generate_tfi`
- `MPO.generate_spinless_fermion`
- `MPO.generate_hubbard`

The builder does not construct dense Hamiltonians for large-system benchmark
paths.

## Config And Result Schema

`latticetn.config_schema` defines:

- `MethodConfig`
- `RuntimeConfig`
- `ObservableSpec`
- `ResultSchema`

The result dict contains:

- `model`
- `method`
- `runtime`
- `summary`
- `sweep_history`
- `observables`
- `diagnostics`

AD-DMRG jobs report:

```json
{
  "ad_used": true,
  "ed_used": false,
  "classical_dmrg_used": false,
  "lanczos_used": false
}
```

Classical DMRG jobs report:

```json
{
  "ad_used": false,
  "ed_used": false,
  "classical_dmrg_used": true,
  "lanczos_used": true
}
```

## Python API

```python
from latticetn.model_registry import build_model_spec
from latticetn.runner import run_latticetn_job

model = build_model_spec(
    "hubbard",
    N=4,
    sector={"mode": "hard", "target_nup": 2, "target_ndown": 2},
)

result = run_latticetn_job(
    model,
    {
        "name": "ad_dmrg",
        "chi": 4,
        "sweeps": 1,
        "optimizer": "adam",
        "local_steps": 1,
        "lr": 0.01,
        "sector_mode": "hard",
    },
    {"device": "cpu", "dtype": "complex128", "no_ed": True},
    {"names": ["energy", "sector", "bond_dims"]},
)
```

## CLI

Example job:

```text
examples/jobs/hubbard_ad_hard_N4.json
```

PowerShell:

```powershell
$env:PYTHONPATH="."
python .\scripts\run_latticetn_job.py `
  --job-json .\examples\jobs\hubbard_ad_hard_N4.json `
  --output .\outputs\hubbard_ad_hard_N4_result.json
```

Bash:

```bash
PYTHONPATH=. python scripts/run_latticetn_job.py \
  --job-json examples/jobs/hubbard_ad_hard_N4.json \
  --output outputs/hubbard_ad_hard_N4_result.json
```

The older AD benchmark CLI remains compatible:

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
  --no-ed `
  --output .\outputs\hubbard_N40_ad_result.json
```

It can also accept a structured model JSON via `--model-spec-json`.

## Observables

Stage 10 minimally supports:

- `energy`
- `energy_per_site`
- `sector`
- `bond_dims`
- `truncation`
- `gradient_norm`
- `runtime`

Unsupported observables raise clear `NotImplementedError` errors rather than
returning placeholder data.

## Benchmark Policy

Large-system AD benchmark paths continue to skip ED by design. ED is reserved
for small-system reference tests. The AD runner does not use classical DMRG,
Lanczos, or dense Hamiltonian construction. Device selection remains dynamic
through `cpu`, `cuda`, or `auto`.
