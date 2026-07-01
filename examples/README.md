# Examples

Short, readable, **CPU-only** examples showing how to use `latticeTN` as a
library. They are **not** tests (run the `scripts/*_score.py` runners for
validation) and they do not run long training.

- `heisenberg_ad_mps.py` — build an MPS + Heisenberg MPO, compute the native
  Rayleigh energy, compare to exact, then train it with the global AD-MPS
  solver (Stage 4R).
- `local_ad_sweep.py` — train the same system with AD local-tensor
  optimization (Stage 5A), showing the center-tensor sweep and optional
  stabilization.

## Run

```bash
pip install -e .               # if not already installed
python examples/heisenberg_ad_mps.py
python examples/local_ad_sweep.py
```

Both print the exact ground energy and the AD final energy. They use small N and
few steps so they finish in seconds on CPU.
