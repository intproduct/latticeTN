# Stage 2.5 GPU Testing Protocol

This stage adds **opt-in** GPU correctness smoke tests, verifying that the
MPS + MPO + PyTorch autograd Heisenberg path runs correctly on CUDA. It is
**not** a performance benchmark, does not enlarge system sizes, and does not
run long training. Stage 1 and Stage 2 CPU validation must remain green and
unchanged.

## Physics conventions (unchanged)

- `H = J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})`
- `S = sigma / 2` (spin convention, NOT Pauli)
- `J = 1.0`, open boundary
- default dtype `torch.complex128`
- CPU-only for Stage 1/2 validation; GPU is an additional, opt-in path

## Opt-in rules

1. Default `pytest` MUST NOT depend on a GPU.
2. GPU tests run only when `LATTICETN_RUN_GPU=1` is set.
3. If `torch.cuda.is_available()` is false, GPU tests cleanly skip.
4. GPU tests are NOT in `validation_score.py` or `benchmark_score.py` test
   lists; they are a separate, opt-in path.

## GPU selection (multi-GPU machines)

This machine has multiple GPUs. The smoke test does **not** default to
`cuda:0`. Instead it selects a GPU whose name contains the filter string.

Environment variables:

- `LATTICETN_RUN_GPU=1` — enables the GPU path.
- `LATTICETN_GPU_NAME_FILTER="Pro 4000 Blackwell"` — substring to match in the
  GPU name (default `"Pro 4000 Blackwell"`).

Discovery (in order):

1. `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader`
2. If `nvidia-smi` is unavailable, `torch.cuda.get_device_name(i)` for each
   visible device.

Selection rules:

1. Iterate over all visible CUDA devices.
2. Pick the first device whose name contains the filter (case-insensitive).
3. Use that device only. Set/confirm `CUDA_VISIBLE_DEVICES` or address it by
   its torch logical index resolved by name.
4. If multiple GPUs match, use the first and record all matches in the report.
5. If NO GPU matches, do NOT fall back to any other GPU (including `cuda:0`).
   Cleanly skip the smoke and record the no-match reason in the report.

The device actually used is resolved by **re-matching the name** against the
current torch-visible devices, so the logical index is always the matched GPU
regardless of ordering differences between `nvidia-smi` and torch.

## Required correctness checks

1. CPU/GPU Heisenberg MPO dense matrix agree (within `1e-10`).
2. CPU/GPU random-MPS energy agree (within `1e-8`).
3. GPU `energy_with_MPO` can call `backward()`.
4. All MPS parameters have non-None gradients after backward on GPU.
5. A short variational optimization lowers the energy on GPU.
6. GPU final energy must not fall below the exact ground energy beyond
   tolerance (`< exact - 1e-6`).
7. No CPU/CUDA tensor mixing (all MPS/MPO tensors and grads on the chosen GPU).

## Autograd rule

The differentiable energy path (`mps.energy_with_MPO`, the optimization loop)
does NOT use `.detach()`, `.data`, or unnecessary `.item()`. Energy values are
read as plain Python floats only for the **report path**, outside the
construction of any backward graph. Parameter normalization reuses the Stage 1
`_full_normalize` routine (mutates `.data` under `no_grad`, outside the energy
path), which is allowed by `CLAUDE.md`.

## Recommended commands

Opt-in smoke via the score script:

```bash
LATTICETN_RUN_GPU=1 LATTICETN_GPU_NAME_FILTER="Pro 4000 Blackwell" \
    python scripts/gpu_score.py --smoke
```

Opt-in smoke via pytest:

```bash
LATTICETN_RUN_GPU=1 LATTICETN_GPU_NAME_FILTER="Pro 4000 Blackwell" \
    pytest -q tests/test_gpu_device_parity.py tests/test_gpu_heisenberg_smoke.py
```

Without `LATTICETN_RUN_GPU=1`, the score script prints a message and exits 0,
and the pytest tests skip.

## Stop conditions

Stage 2.5 is complete when:

1. `python scripts/validation_score.py --fast` exits 0.
2. `python scripts/benchmark_score.py --fast` exits 0.
3. With a matching GPU present,
   `LATTICETN_RUN_GPU=1 LATTICETN_GPU_NAME_FILTER="Pro 4000 Blackwell"
   python scripts/gpu_score.py --smoke` exits 0.
4. With no CUDA: GPU tests cleanly skip and `docs/GPU_REPORT.md` states the
   numerical tests were not run.
5. With CUDA but no name-matching GPU: GPU tests cleanly skip (no fallback to
   other GPUs) and `docs/GPU_REPORT.md` states no matching GPU was found.
6. `docs/GPU_REPORT.md` records all required fields (see that file).
7. `docs/CLAUDE_PROGRESS_GPU.md` records changes, commands, results, limits.

## Hard constraints

- No formal performance benchmark.
- No enlarged system sizes (N=4/6 only).
- No long training (short opt, ~20 steps).
- GPU tests do not replace CPU physics validation.
- No default occupation of other GPUs; never default to `cuda:0`.
- No changes to Stage 1/2 physics conventions or thresholds.
- No `.detach()`/`.data`/unnecessary `.item()` in the differentiable energy path.
- No large new dependencies (torch/numpy only).
