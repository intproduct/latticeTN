# Claude Progress Log: Stage 2.5 GPU Readiness

Append checkpoint entries here.

## Checkpoint 2026-07-01 — Stage 2.5 complete

```text
Date/time: 2026-07-01
Stage: Stage 2.5 GPU readiness / CPU-GPU device parity (final)
Files changed (NEW):
  - scripts/run_gpu_smoke.py  : shared GPU discovery + selection + smoke checks
  - scripts/gpu_score.py      : opt-in orchestrator, writes docs/GPU_REPORT.md
  - tests/test_gpu_device_parity.py    : opt-in device-parity pytest tests
  - tests/test_gpu_heisenberg_smoke.py : opt-in heisenberg-opt pytest tests
  - docs/GPU_TESTING_PROTOCOL.md       : protocol
  - docs/GPU_REPORT.md                 : generated report (PASS on matched GPU)
  - docs/CLAUDE_PROGRESS_GPU.md        : this file
Files NOT changed: Stage 1/2 sources and tests untouched; conventions/ thresholds unchanged.
Commands run (and results):
  - python scripts/validation_score.py --fast                            -> exit 0 (PASS)
  - python scripts/benchmark_score.py --fast                             -> exit 0 (PASS)
  - pytest -q tests/test_gpu_device_parity.py tests/test_gpu_heisenberg_smoke.py
        (default, no env)                                                -> 10 skipped
  - LATTICETN_RUN_GPU=1 LATTICETN_GPU_NAME_FILTER="Pro 4000 Blackwell"
        pytest -q tests/test_gpu_device_parity.py tests/test_gpu_heisenberg_smoke.py
                                                                          -> 10 passed
  - LATTICETN_RUN_GPU=1 LATTICETN_GPU_NAME_FILTER="Pro 4000 Blackwell"
        python scripts/gpu_score.py --smoke
        -> used GPU index=2 name='NVIDIA RTX PRO 4000 Blackwell', exit 0 (PASS)
  - python scripts/gpu_score.py --smoke (no env)                         -> exit 0 (opt-in msg)
  - LATTICETN_RUN_GPU=1 LATTICETN_GPU_NAME_FILTER="NONEXISTENT_XYZ"
        python scripts/gpu_score.py --smoke                              -> exit 0 (clean skip)
GPU smoke numerical result (matched GPU, N=4 chi=4 steps=20 seed=0):
  - exact ground energy = -1.616025403784439
  - CPU energy = -0.07810439850125714, GPU energy = -0.07810439850125718
  - |CPU - GPU| diff = 4.163e-17 (machine precision)
  - MPO dense CPU/GPU match = True
  - backward() OK on GPU = True
  - all MPS params grad not None = True
  - short opt energy decreased: -0.0781 -> -1.3683 (=True)
  - below exact ground energy = False
  - no CPU/CUDA tensor mixing = True
  - overall pass = True
Environment facts:
  - CUDA available = True; discovery source = nvidia-smi
  - Visible GPUs (physical): 0=NVIDIA RTX PRO 4000 Blackwell (24467 MB),
    1=Tesla V100-SXM2-16GB, 2=Tesla V100-SXM2-16GB
    (Note: torch-visible ordering can differ from nvidia-smi physical order;
     the device is selected by NAME re-match, so the logical index used was 2
     while the matched physical name is 'NVIDIA RTX PRO 4000 Blackwell'.)
  - GPU name filter = "Pro 4000 Blackwell"; matched GPU actually used.
Current failing item: none
Next action: none (Stage 2.5 stop conditions met).
Known limitations:
  - GPU smoke is a correctness check only, not a performance benchmark.
  - MPS/GPU path reuses Stage 1 contractions; large-N GPU scaling is out of scope.
  - Only GPUs matching LATTICETN_GPU_NAME_FILTER are used; other visible GPUs
    are deliberately ignored (no fallback to cuda:0 or any other GPU).
  - A UserWarning about converting a requires_grad tensor to a scalar is emitted
    when reading energy values for the report (float(...)). This is the report
    path, OUTSIDE the differentiable energy path / before the backward graph, so
    it does not violate the CLAUDE.md autograd rule.
Notes:
  - conventions unchanged: S=sigma/2, J=1.0, open boundary, complex128.
  - GPU tests are opt-in (LATTICETN_RUN_GPU=1) and NOT in the default
    validation_score.py / benchmark_score.py test lists.
```
