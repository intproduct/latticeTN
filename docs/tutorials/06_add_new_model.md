# 06 — Add a new model

How to extend latticeTN to a **new** Hamiltonian model (e.g. XXZ, or your own),
keeping the AD mainline intact. The existing second model, **TFI**
(transverse-field Ising), is the template. This tutorial walks the full
workflow: dense reference → MPO builder → MPO-to-dense test → native energy
test → AD solver → benchmark.

## Goal

- Add a new model end-to-end without touching the AD mainline or physics
  conventions.
- Keep `S = σ/2`, `J`-scaling, open boundary, `complex128`.
- Reuse the existing `train_ad_*` solvers on the new MPO.
- Add the right tests so the new model is validated exactly like Heisenberg/TFI.

## The AD mainline is unchanged

Adding a model means adding a **Hamiltonian** (a dense reference + an MPO
builder). The solver path stays:

```
MPS parameters -> differentiable Rayleigh quotient -> loss.backward() -> optimizer.step()
```

The new MPO just plugs into `rayleigh_energy_native(mps, mpo)` and the
`train_ad_*` drivers. **No** new algorithm; **no** SVD/QR/`eigh` in the loss;
DMRG/Lanczos remain reference baselines only.

## Workflow (8 steps)

### 1. Dense reference — `latticetn/operators.py`

Add `def <model>_dense(N, ..., dtype=tc.complex128, device="cpu") -> tc.Tensor`
returning the `(2**N, 2**N)` dense Hamiltonian, built from
`spin_operators(dtype, device)` (**keep `S = σ/2`**). Templates:
`heisenberg_dense`, `tfi_dense`.

```python
# latticetn/operators.py
def xxz_dense(N, Jz=1.0, Jxy=1.0, dtype=tc.complex128, device="cpu") -> tc.Tensor:
    ops = spin_operators(dtype=dtype, device=device)   # S = sigma/2
    Sz, Sp, Sm = ops["Sz"], ops["S+"], ops["S-"]
    I = ops["I"]
    H = tc.zeros((2**N, 2**N), dtype=dtype, device=device)
    for i in range(N - 1):                              # open boundary
        SzSz = tc.kron(tc.kron(tc.eye(2**i, dtype=dtype, device=device), Sz),
                       tc.kron(Sz, tc.eye(2**(N-i-2), dtype=dtype, device=device)))
        SpSm = tc.kron(tc.kron(tc.eye(2**i, dtype=dtype, device=device), Sp),
                       tc.kron(Sm, tc.eye(2**(N-i-2), dtype=dtype, device=device)))
        SmSp = tc.kron(tc.kron(tc.eye(2**i, dtype=dtype, device=device), Sm),
                       tc.kron(Sp, tc.eye(2**(N-i-2), dtype=dtype, device=device)))
        H = H + Jz*SzSz + 0.5*Jxy*(SpSm + SmSp)        # SxSx+SySy = (1/2)(S+S-+S-S+)
    return H
```

### 2. MPO builder — `latticetn/mpo.py`

Add a `generate_<model>(self, ...)` method on the `MPO` class that fills
`self.tensors[i]` for `i in range(self.length)` and returns `self`. Templates:
`generate_heisenberg` (D=5), `generate_tfi` (D=3). The MPO is the **MPS-shaped**
factorized Hamiltonian; `to_dense()` contracts it back to the dense matrix.

The exact virtual-bond construction is model-specific; the Heisenberg builder
uses a 5-state bond (idle / carry-Sz / carry-S- / carry-S+ / done) and writes
the start/pair operators at each bond. For a nearest-neighbor model you
generalize the same pattern; for longer range, increase the bond dimension D.

### 3. MPO-to-dense test — `tests/test_<model>_mpo_dense.py`

Assert the MPO contracts to the dense reference, for several N and parameter
values. Template: `tests/test_tfi_mpo_dense.py`, `tests/test_heisenberg_mpo_dense.py`.

```python
# tests/test_xxz_mpo_dense.py
import torch as tc
from latticetn.mpo import MPO
from reference_models import xxz_dense, exact_ground_energy   # re-export from operators

DTYPE = tc.complex128

def test_xxz_mpo_matches_dense_several_sizes():
    for N in [2, 3, 4, 5]:
        for (Jz, Jxy) in [(1.0, 1.0), (1.0, 0.5), (2.0, 1.0)]:
            H_mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_xxz(Jz=Jz, Jxy=Jxy).to_dense()
            H_ref = xxz_dense(N, Jz=Jz, Jxy=Jxy, dtype=DTYPE)
            assert tc.allclose(H_mpo, H_ref, atol=1e-12)

def test_xxz_mpo_hermitian():
    for N in [2, 4, 6]:
        H = MPO.from_bonds(N, 2, dtype=DTYPE).generate_xxz().to_dense()
        assert tc.allclose(H, H.conj().T, atol=1e-12)

def test_xxz_mpo_open_boundary_shape():
    mpo = MPO.from_bonds(6, 2, dtype=DTYPE).generate_xxz()
    assert mpo.tensors[0].shape[0] == 1 and mpo.tensors[5].shape[1] == 1   # open ends
```

> Note: the test files import the dense reference from `tests/reference_models.py`
> (which re-exports `latticetn.operators`), not directly from `latticetn.operators`.
> Add your `<model>_dense` re-export there too.

### 4. Native energy test — `tests/test_<model>_native_energy.py`

Assert the differentiable native contraction `rayleigh_energy_native(mps, mpo)`
matches the dense energy on a known state. Template:
`tests/test_native_mpo_energy_contraction.py`.

```python
def test_xxz_native_energy_matches_dense():
    N = 4
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_xxz()
    mps = MPS(N, 2, 4, dtype=DTYPE)
    e_native = rayleigh_energy_native(mps, mpo)
    e_dense  = (mps.to_dense().conj() @ mpo.to_dense() @ mps.to_dense()).real / (mps.to_dense().conj() @ mps.to_dense()).real
    assert abs(float(e_native) - float(e_dense)) < 1e-8
```

### 5. Observables (optional)

If your model has natural observables (e.g. staggered magnetization), add
dense + native variants with cross-checks, mirroring `latticetn.observables`.

### 6. Benchmark script — `scripts/run_<model>_benchmark.py` + `scripts/<model>_score.py`

Mirror `scripts/run_ad_mps_heisenberg.py` / `ad_two_site_score.py`: build the
new MPO, train with `train_ad_*`, compare to exact (small N) and DMRG, write a
report. Keep the score script's required-terms check.

### 7. AD solver

Train on the new MPO with the **existing** solvers — no new algorithm:

```python
mps = MPS(N, 2, chi, dtype=tc.complex128)
mpo = MPO.from_bonds(N, 2, dtype=tc.complex128, device="cpu").generate_xxz(Jz=1.0, Jxy=0.5)
res = train_ad_two_site(mps, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                        optimizer="lbfgs", max_bond_dim=chi)
```

### 8. Baseline comparison

Compare the AD final energy to exact (small N) and to classical DMRG
(`dmrg.run_dmrg`). The AD energy must be **at/above** exact within tolerance
and **never below ground** by more than `1e-8`. DMRG is a **reference
baseline**, never the AD mainline.

## Run commands

```bash
# MPO-to-dense + native energy tests:
python -m pytest -q tests/test_xxz_mpo_dense.py tests/test_xxz_native_energy.py

# Full new-model score (once the score script exists):
python scripts/xxz_score.py --fast
```

## Expected output

- `test_xxz_mpo_matches_dense_several_sizes` → PASS (atol=1e-12).
- AD solver reaches the exact ground energy within the model's tolerance (e.g.
  `1e-5` at N=6 for a well-conditioned model), `below_ground=False`.

## Common errors

- **MPO-to-dense mismatch > 1e-12** — your MPO builder has a wrong start/pair
  operator or boundary handling. Compare against the Heisenberg builder's
  5-state pattern; check that the left boundary keeps only row 0 and the right
  boundary keeps only the final column.
- **Energies off by a factor of 4** — you used Pauli `σ` instead of `S = σ/2`.
  Build the dense reference from `spin_operators(...)`, not `pauli_matrices`.
- **`below_ground = True`** — a bug. Check Hermiticity (`H == H.conj().T`),
  dtype (`complex128`), and that the MPO is not accidentally non-Hermitian.
- **Putting `eigh`/`svd`/`qr` in the energy path** — don't. The loss path must
  stay autograd-clean; the AST policy tests enforce this.
- **Importing `dmrg`/`lanczos` from an AD module** — forbidden; they are
  reference baselines only, used from score scripts, never from the loss path.

## Conventions to keep

- `S = σ/2` (never silently mix with Pauli).
- Open boundary; MPS `(left, phys, right)`, MPO `(left, right, phys_in, phys_out)`.
- Default `torch.complex128`, default device CPU.
- The loss path stays autograd-clean (see `docs/AD_MAINLINE_POLICY.md`).
