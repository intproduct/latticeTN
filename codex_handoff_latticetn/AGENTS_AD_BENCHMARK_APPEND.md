# AGENTS.md addendum: AD large-model benchmark stage

Append this section to the repository root `AGENTS.md` when starting the AD benchmark stage.

## Current stage override

The current stage is **AD-mainline large-model benchmarking**. The earlier small-N ED validation remains important, but this stage extends validation to large open-boundary chains where exact diagonalization must be skipped.

Primary solver under test:

```text
two-site AD local-tensor optimization
= ADTwoSiteOptimizer + differentiable Rayleigh quotient + loss.backward() + torch optimizer
```

Classical DMRG/Lanczos/eigh may be used only as separate reference baselines. They must not be called inside AD benchmark runners.

## Stage success criteria

- `scripts/run_ad_model_benchmark.py` supports Heisenberg, TFI, spinless t-V, and Hubbard MPOs.
- CPU-only helper tests pass.
- Large-N commands explicitly skip ED and never build dense Hamiltonians.
- Result JSON includes `ed_skipped: true`, `dense_hamiltonian_built: false`, `dmrg_lanczos_used: false`, and `ad_mainline: true`.
- GPU benchmark commands are documented but are not part of default pytest.

## Extra pause condition

Pause before claiming real-material capability. This stage only establishes 1D benchmark-model capability.
