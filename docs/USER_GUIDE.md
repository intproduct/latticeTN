# User Guide

This guide explains how to **use `latticeTN` as an automatic-differentiation
tensor-network library**. It is written for users, not as a development log.

> **One-sentence orientation:** build an MPS + MPO, compute the differentiable
> Rayleigh quotient `E = <psi|H|psi>/<psi|psi>`, and minimize it with
> `loss.backward()` + a torch optimizer. That is the AD mainline. Everything
> else (SVD/QR/canonicalization, DMRG, Lanczos, exact diagonalization) is an
> auxiliary or reference tool.

---

## 1. Installation

### 1.1 Create an environment

```bash
python -m venv .venv && source .venv/bin/activate     # POSIX
# or:  conda create -n latticetn python=3.11 && conda activate latticetn
```

> On Windows, the `python` on PATH may be a broken MS Store stub — use a real
> interpreter (e.g. a conda env's `python.exe`) explicitly.

### 1.2 Install runtime + dev dependencies + the package

```bash
pip install -r requirements.txt        # torch, numpy, scipy, tqdm, matplotlib
pip install -r requirements-dev.txt    # + pytest, pytest-timeout
pip install -e .                       # editable install of the latticetn package
```

After this, `import latticetn` works from any directory.

### 1.3 CPU / GPU notes

- All default tests and score scripts run **CPU-only** with `torch.complex128`.
- GPU is **opt-in**: run `python scripts/gpu_score.py --fast` (Stage 2.5
  correctness smoke) or `LATTICETN_RUN_GPU=1 python scripts/ad_gpu_benchmark_score.py --fast`
  (Stage 6A CPU/GPU AD benchmark) deliberately.
- Never mix CPU and CUDA tensors in one contraction — keep `device` consistent
  across an MPS / MPO / operators. The Stage 2.5 smoke uses name-matched device
  selection (multi-GPU boxes); the Stage 6A benchmark targets the machine's
  single GPU and uses `cuda:0` when `LATTICETN_RUN_GPU=1` and CUDA is available.

---

## 2. Core concepts

| Concept | What it is | Where |
|---|---|---|
| **MPS** | Matrix-product state; open boundary; each site tensor `(left, phys, right)` is a trainable `nn.Parameter`. | `latticetn.mps.MPS` |
| **MPO** | Matrix-product operator; site tensor `(left, right, phys_in, phys_out)`. | `latticetn.mpo.MPO` |
| **Rayleigh quotient** | `E = <psi|H|psi> / <psi|psi>` — the variational energy. | `contractions.rayleigh_energy_native` |
| **Differentiable contraction** | Contract MPS/MPO tensors directly (einsum sweeps), no `to_dense`; scales polynomially in N, chi; gradients flow to the MPS. | `latticetn.contractions` |
| **Observables** | `<Sz_i>`, `<Sz_i Sz_j>`, bond energy, entanglement entropy. Native (scalable, differentiable) and dense (small-N reference) variants. | `latticetn.contractions`, `latticetn.observables` |
| **Canonicalization / compression** | Gauge fixing (left/right/mixed canonical) and SVD bond compression — **auxiliary tools**, not optimizers. | `latticetn.canonical` |
| **Global AD-MPS** | Train **all** MPS tensors at once on the Rayleigh quotient (Stage 4R). | `latticetn.ad_variational` |
| **AD local-tensor optimization** | Train **one center tensor at a time**, sweep the orthogonality center by QR (Stage 5A). | `latticetn.ad_local` |
| **Classical DMRG / Lanczos** | Reference baseline / oracle, **never** the AD mainline. | `latticetn.dmrg`, `latticetn.lanczos` |

### 2.1 The AD mainline (the one true solver)

```
MPS parameters (trainable nn.Parameter)
   -> differentiable Rayleigh quotient  E = <psi|H|psi>/<psi|psi>
   -> loss.backward()        (PyTorch autograd)
   -> torch optimizer step   (Adam / LBFGS)
   -> [optional post-step stabilization: none|tensor_norm|qr|canonical]
```

The loss path (`rayleigh_energy_native`) is **autograd-clean**: it uses no
`detach()`, `.data`, `torch.no_grad()`, unnecessary `.item()`, no `eigh`/`svd`/
`qr`, and never calls `dmrg`/`lanczos`. This is AST-enforced by the tests.

### 2.2 What SVD/QR/canonicalization are — and are not

They are **optional post-step stabilization / projection / compression** tools:
gauge fixing, conditioning, bond truncation, diagnostics. They run under
`torch.no_grad()` mutating `.data`, **outside the loss graph**, and are
scale/gauge-invariant for the Rayleigh quotient. They are **never the
optimizer**. If you ever reach for `svd`/`eigh`/`qr` to "solve" for a tensor,
you have left the AD mainline.

### 2.3 What DMRG / Lanczos are

Reference baselines / oracles: classical two-site DMRG and the Krylov local
eigensolver exist for **correctness comparison and benchmarking**, not as the
project's solver. They live behind their own opt-in score scripts and are never
imported by the AD modules.

### 2.4 Tutorials (step-by-step, runnable)

This guide is the reference manual; for hands-on, copy-pasteable walkthroughs
with expected output and common errors, see the tutorials (also in
[中文](USER_GUIDE.zh-CN.md)):

| # | Tutorial | Covers |
|---|---|---|
| 01 | [Quickstart (Heisenberg)](tutorials/01_quickstart_heisenberg.md) | build MPS+MPO, evaluate the differentiable energy, compare to ED |
| 02 | [Global AD-MPS](tutorials/02_global_ad_mps.md) | train all tensors at once (Adam) |
| 03 | [One-site AD local](tutorials/03_one_site_ad_local.md) | center-tensor sweep (LBFGS), QR center movement |
| 04 | [Two-site AD local](tutorials/04_two_site_ad_local.md) | two-site `Θ` block, bond growth via SVD split |
| 05 | [CPU/GPU benchmark](tutorials/05_cpu_gpu_benchmark.md) | opt-in GPU parity + speedup |
| 06 | [Add a new model](tutorials/06_add_new_model.md) | dense ref → MPO → tests → AD solver → benchmark |

You can also browse them as a local site: `pip install -r requirements-docs.txt
&& mkdocs serve` (see `mkdocs.yml`).

---

## 3. Minimal example: Heisenberg MPO + energy

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.contractions import rayleigh_energy_native
from latticetn.operators import heisenberg_dense, exact_ground_energy

N, chi, dtype = 6, 8, tc.complex128
mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)

E = rayleigh_energy_native(mps, mpo)            # differentiable, requires_grad
E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))
print("E =", float(E.real), " exact E0 =", E0)   # E >> E0 for a random MPS
```

`rayleigh_energy_native` is the same quantity as `MPS.energy_with_MPO` but is
reached through the standalone, scalable contraction module — and it is the
function the AD solvers use as their loss.

### 3.1 Spinless fermion t-V chain (Stage 7A, Jordan-Wigner)

Stage 7A adds the open-boundary 1D spinless fermion t-V chain on top of the
**unchanged** AD mainline — the loss path is operator-agnostic, so you just
swap the Hamiltonian/MPO. This is 1D Jordan-Wigner fermions, **not** a full
graded fermionic tensor network; the JW parity string `F = (-1)^n` is the key
(it makes fermionic operators on different sites anticommute).

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.contractions import rayleigh_energy_native
from latticetn.operators import spinless_fermion_dense, exact_ground_energy
from latticetn.fermion_operators import fermion_operators

N, chi, dtype = 6, 8, tc.complex128
# H = -t sum (c^d_i c_{i+1}+h.c.) + V sum (n_i-1/2)(n_{i+1}-1/2) - mu sum (n_i-1/2)
mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_spinless_fermion(
    t=1.0, V=0.5, mu=0.0)

E = rayleigh_energy_native(mps, mpo)            # differentiable, same loss path
E0, _ = exact_ground_energy(
    spinless_fermion_dense(N, t=1.0, V=0.5, mu=0.0, dtype=dtype, device="cpu"))
print("E =", float(E.real), " exact E0 =", E0)

# fermion observables (dense-reference; NN hopping carries the JW string)
from latticetn.observables import (mps_fermion_local_density,
    mps_fermion_nn_hopping)
print("<n_0> =", float(mps_fermion_local_density(mps, 0).real))
print("<c^d_0 c_1 + h.c.> =", float(mps_fermion_nn_hopping(mps, 0).real))
```

The three AD mainline solvers (`ad_variational.train_ad_mps`,
`ad_local.train_ad_local`, `ad_two_site.train_ad_two_site`) work on the fermion
MPO unchanged. CPU/GPU timing uses the **unified GPU selector**
(`scripts/gpu_selector.py`), which selects a V100/TITAN V and clean-skips
otherwise (no fallback). Run `python scripts/fermion_score.py --fast` (CPU) or
`LATTICETN_RUN_GPU=1 python scripts/fermion_score.py --fast` (opt-in GPU).

### 3.2 General 1D model builder + benchmark registry (Stage 7B)

Stage 7B abstracts the Heisenberg and spinless-fermion t-V Hamiltonians behind
a **unified 1D model-construction layer** (`latticetn/model_builder.py`). This
is a **model/MPO construction layer, NOT a new solver** — the AD mainline is
unchanged; the builders dispatch to the existing validated generators so the
physics is byte-identical to Stage 1/7A.

```python
import torch as tc
from latticetn.model_builder import (heisenberg_model,
    spinless_fermion_tv_model, build_dense, build_mpo)
from latticetn.benchmarking import benchmark_model

# Build a model spec, then its dense + MPO Hamiltonians.
spec = spinless_fermion_tv_model(N=6, t=1.0, V=0.5, mu=0.0)
H = build_dense(spec)          # == operators.spinless_fermion_dense (Stage 7A)
mpo = build_mpo(spec)          # == MPO.generate_spinless_fermion (Stage 7A)

# Unified CPU/GPU benchmark registry (Stage 7A+ timing contract).
r = benchmark_model(spec, chi=8, seed=0, steps=120)
# r["cpu"], r["gpu"] (or None + skip reason), r["speedup"], r["exact_energy"], ...
```

The `ModelSpec` carries an explicit `statistics` ("boson"|"fermion") and a
list of terms (`OnsiteTerm`, `TwoSiteTerm` for bosonic/spin, `FermionHopTerm`
+ `DensityDensityTerm` for fermionic/JW). Fermion terms keep the JW parity
string (no hard-core-boson degradation). Run
`python scripts/model_builder_score.py --fast` (CPU) or
`LATTICETN_RUN_GPU=1 python scripts/model_builder_score.py --fast` (opt-in GPU,
V100/TITAN V).

### 3.3 Spinful Hubbard chain (Stage 7C, Jordan-Wigner)

Stage 7C adds the open-boundary 1D **spinful Hubbard chain** on top of the
**unchanged** AD mainline. The model is

```
H = -t sum_{i,s} (c^d_{i,s} c_{i+1,s} + h.c.)
    + U sum_i (n_{i,up}-1/2)(n_{i,down}-1/2)
    - mu sum_i (n_{i,up}+n_{i,down}-1) - h sum_i (n_{i,up}-n_{i,down})
```

Local basis `|0>, |up>, |down>, |up,down>` (d=4); global mode ordering is
fixed to **site-major** `(0_up,0_down,1_up,1_down,...)`. This is 1D
Jordan-Wigner fermions, **not** a full graded fermionic tensor network; the
per-site JW parity `P = F_up x F_down` (plus the intra-site `F_up` already
inside the local `cdown`/`cdagdown`) is the key.

```python
import torch as tc
from latticetn.model_builder import hubbard_model, build_dense, build_mpo
from latticetn import contractions as K
from latticetn.operators import hubbard_dense, exact_ground_energy
from latticetn.fermion_operators import hubbard_local_operators

N, chi, dtype = 4, 4, tc.complex128
spec = hubbard_model(N, t=1.0, U=4.0, mu=0.0, h=0.0)
H = build_dense(spec)          # == operators.hubbard_dense (full 2N-mode JW)
mpo = build_mpo(spec)          # == MPO.generate_hubbard (bond dim 6, d=4)
assert tc.allclose(mpo.to_dense(), H, atol=1e-12)

# differentiable Rayleigh energy (operator-agnostic AD mainline):
tc.manual_seed(0)
from latticetn.mps import MPS
mps = MPS(N, 4, chi, dtype=dtype)
E = K.rayleigh_energy_native(mps, mpo)        # <psi|H|psi>/<psi|psi>

# reference baseline (NOT the AD path):
E0, _ = exact_ground_energy(hubbard_dense(N, t=1.0, U=4.0, dtype=dtype))

# Hubbard observables (dense-reference; spin-resolved NN hopping carries the
# surviving per-site parity P at the left-factor site):
from latticetn.observables import (mps_hubbard_local_density,
    mps_hubbard_double_occ, mps_hubbard_local_sz, mps_hubbard_nn_hopping)
print("<n_up_0>   =", float(mps_hubbard_local_density(mps, 0, "up").real))
print("<docc_0>   =", float(mps_hubbard_double_occ(mps, 0).real))
print("<Sz_0>     =", float(mps_hubbard_local_sz(mps, 0).real))
print("<c^d_{0,up} c_{1,up}+h.c.> =",
      float(mps_hubbard_nn_hopping(mps, 0, "up").real))
```

The three AD mainline solvers (`train_ad_mps`, `train_ad_local`,
`train_ad_two_site`) work on the Hubbard MPO unchanged. CPU/GPU timing uses
the **unified GPU selector** (V100/TITAN V only, no fallback). Run
`python scripts/hubbard_score.py --fast` (CPU) or
`LATTICETN_RUN_GPU=1 python scripts/hubbard_score.py --fast` (opt-in GPU).

---

## 4. Global AD-MPS training (Stage 4R)

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps

tc.manual_seed(0)
mps = MPS(6, 2, 8, dtype=tc.complex128)
mpo = MPO.from_bonds(6, 2, dtype=tc.complex128, device="cpu").generate_heisenberg(J=1.0)

ad = ADVariationalMPS(mps, mpo)                 # all site tensors trainable
loss = ad.loss()                                # differentiable Rayleigh quotient
loss.backward()                                 # grads flow to every site tensor
# train:
res = train_ad_mps(ad, num_steps=300, lr=1e-2, optimizer="adam",
                   projection="tensor_norm")    # post-step L2 renormalization
print("final E =", res["final_energy"])
print("energy history length:", len(res["energy_history"]))
```

**What happens inside `train_ad_mps`:** each step zeros grads, computes
`ad.loss()` (= `rayleigh_energy_native`), calls `.backward()`, takes an
optimizer step, then optionally applies a post-step `projection` (`none` /
`tensor_norm` / `canonical`) under `no_grad`. The optimizer is Adam or LBFGS.
`projection` is **stabilization, not the solver**.

---

## 5. AD local-tensor optimization (Stage 5A)

Instead of training all tensors at once, freeze every tensor except one
**center tensor**, optimize it, sweep the orthogonality center by QR, repeat.

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_local import train_ad_local

tc.manual_seed(0)
mps = MPS(6, 2, 8, dtype=tc.complex128)
mpo = MPO.from_bonds(6, 2, dtype=tc.complex128, device="cpu").generate_heisenberg(J=1.0)

res = train_ad_local(mps, mpo,
                     num_sweeps=4, local_steps=20, lr=1.0,
                     optimizer="lbfgs",          # LBFGS: near-quadratic local problem
                     stabilization="qr")         # optional post-step projection
print("final E =", res["final_energy"])
```

**Reading the result:** `energy_history[0]` is the initial energy; one entry is
appended per sweep. `sweeps` records per-sweep direction and energy.

**What is the optimizer vs. what is auxiliary:**
- The **optimizer** is `loss.backward()` + a torch optimizer step on the single
  center tensor.
- `stabilization` (`none`/`tensor_norm`/`qr`/`canonical`) is **optional
  post-step stabilization**, under `no_grad` mutating `.data` — a gauge/
  conditioning aid, **not** the solver.
- The QR sweeps in `move_center` are **orthogonality-center movement** (gauge
  transport), not optimization.

For the lower-level interface (`ADLocalOptimizer`, `.move_center`,
`.set_center`), see [`API_OVERVIEW.md`](API_OVERVIEW.md).

---

## 5b. Two-site AD local optimization (Stage 5B)

Two-site AD extends the local sweep to a **two-site block**: contract two
adjacent site tensors into a single trainable `Θ(l, s_i, s_{i+1}, r)`, train it
on the differentiable local Rayleigh quotient `E(Θ)=<Θ|H_eff|Θ>/<Θ|Θ>`, then
split `Θ` back by SVD with optional `max_bond_dim`/`cutoff`. This is the
autograd analogue of two-site DMRG — **gradient descent on `Θ`, not a local
eigensolver** — and it lets the bond dimension **grow or truncate** as the
sweep proceeds.

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_two_site import train_ad_two_site

tc.manual_seed(0)
mps = MPS(6, 2, 8, dtype=tc.complex128)
mpo = MPO.from_bonds(6, 2, dtype=tc.complex128, device="cpu").generate_heisenberg(J=1.0)

res = train_ad_two_site(mps, mpo,
                        num_sweeps=4, local_steps=20, lr=1.0,
                        optimizer="lbfgs",
                        max_bond_dim=8, cutoff=None)   # optional bond growth / cap
print("final E =", res["final_energy"])
print("bond dims =", res["final_bond_dims"], "max =", res["max_bond"])
```

**What is the optimizer vs. what is auxiliary:**
- The **optimizer** is `loss.backward()` + a torch optimizer step on the single
  two-site `Θ` tensor (a 4-axis `nn.Parameter`).
- `H_eff` is built from **frozen, detached** left/right MPO environments + the
  two MPO tensors; `Θ` is the only trainable leaf. The loss is a pure einsum.
- The SVD in `split()` is **post-step compression** (under `no_grad` on detached
  data); the inter-bond QR re-canonicalization is **gauge fixing**. Neither is
  the solver.
- Because the rest of the chain is orthonormal, `E(Θ)` equals the global
  Rayleigh quotient — minimizing it lowers the global energy (two-site
  variational principle).

Use two-site AD when you want **bond adaptivity** (the ansatz chooses its own χ
via the SVD split). Use one-site AD (§5) when you want a fixed χ. Both reach
machine precision on small N; see tutorial
[04](tutorials/04_two_site_ad_local.md).

---

## 6. Observables & entanglement

### 6.1 Native (scalable, differentiable) observables

```python
from latticetn.operators import spin_operators
from latticetn.contractions import (native_local_expect, native_two_site_expect,
                                    native_bond_energy_heisenberg, native_correlation,
                                    native_norm_sq)

ops = spin_operators(dtype=mps.dtype, device=mps.device)   # S = sigma/2
sz = ops["Sz"]
print("<psi|psi>     =", float(native_norm_sq(mps).real))
print("<Sz_2>        =", float(native_local_expect(mps, sz, 2).real))
print("<Sz_1 Sz_4>   =", float(native_two_site_expect(mps, sz, 1, sz, 4).real))
print("<S_2.S_3>     =", float(native_bond_energy_heisenberg(mps, 2).real))
print("corr(Sz,1,4)  =", float(native_correlation(mps, sz, 1, 4).real))
```

These contract the MPS directly (polynomial in N, chi) — use them in scalable
code and inside losses.

### 6.2 Entanglement entropy (canonical, non-differentiable)

```python
from latticetn.canonical import entanglement_entropy
S_vN = entanglement_entropy(mps, cut=3)   # nats, across bond [0..3)|[3..N)
```

This brings the MPS to mixed-canonical form and reads the Schmidt spectrum — a
**diagnostic**, not part of the loss.

### 6.3 Dense (small-N reference) observables

```python
from latticetn.observables import mps_expect_local, mps_bond_energy_heisenberg, mps_entanglement_entropy
from latticetn.operators import exact_ground_energy, heisenberg_dense
```

Use the dense variants only for small systems / cross-checks against exact
diagonalization; they do not scale and are not part of the scalable training
path.

### 6.4 native vs. dense

- **Native** (`contractions`): contracts MPS tensors directly — scalable,
  differentiable, the path used in training and scalable tests.
- **Dense** (`observables`, `to_dense`): rebuilds the full state vector — exact
  but exponential in N; use only for small-N validation against ED.

Do **not** put dense paths inside scalable tests or the training loop.

---

## 7. Canonicalization & compression (auxiliary, not the optimizer)

```python
from latticetn import canonical as Can

lc  = Can.left_canonical(mps)          # exact QR sweep; state preserved up to phase
rc  = Can.right_canonical(mps)
mc  = Can.mixed_canonical(mps, center=3)
errs = Can.left_orthonormal_all(mps)   # gauge diagnostic

compressed, info = Can.svd_compress(mps, chi=16)
print(info["truncation_errors"], info["bond_dims"])
```

**When to use these:** gauge fixing before a local solve, conditioning a state
that has drifted in gauge, bond compression for memory, or diagnostics
(canonical error, truncation error). **Never** as the optimization mechanism —
they are non-differentiable and run under `no_grad`.

---

## 8. Reference solvers (correctness / benchmark only)

### 8.1 Exact diagonalization

```python
from latticetn.operators import heisenberg_dense, exact_ground_energy
H = heisenberg_dense(8, dtype=tc.complex128, device="cpu")
E0, psi0 = exact_ground_energy(H)     # golden reference (small N only)
```

### 8.2 Classical DMRG

```python
from latticetn import dmrg as D
r = D.run_dmrg(mps, mpo, chi=16, num_sweeps=4, solver="dense")
print("DMRG E =", r["final_energy"], " below ground?", r["below_ground"])
```

### 8.3 Lanczos

`latticetn.lanczos.lanczos_lowest_eigenpair` is the Krylov local eigensolver used
by DMRG's `solver="lanczos"`. It is a reference tool, **not** the AD solver.

> **Remember:** exact / DMRG / Lanczos exist to *check* that the AD solver
> reaches the right variational minimum. They are not the project's mainline.
> Never import them in an AD loss path.

---

## 9. Adding a new model

For a full runnable walkthrough (dense ref → MPO → MPO-to-dense test → native
energy test → AD solver → benchmark, with code and expected output), see
tutorial [06 — Add a new model](tutorials/06_add_new_model.md). Summary:

To add, say, an XXZ chain:

1. **Dense reference Hamiltonian** — add `xxz_dense(N, Jz, Jxy, ...)` to
   `latticetn/operators.py` (build with `spin_operators`, `S = sigma/2`).
2. **MPO** — add a `.generate_xxz(...)` builder to `latticetn/mpo.py`.
3. **MPO-to-dense test** — in `tests/`, assert the MPO reproduces
   `xxz_dense` for small N (cf. `test_heisenberg_mpo_dense.py`).
4. **Native energy test** — assert `rayleigh_energy_native` on a known state
   matches the dense expectation (cf. `test_native_mpo_energy_contraction.py`).
5. **Observables** — add bond-energy / correlation helpers if the model needs
   new ones, with dense cross-checks.
6. **Benchmark script** — a `scripts/run_<model>_benchmark.py` + score.
7. **AD solver** — train with `ADVariationalMPS` / `train_ad_local` on the new
   MPO; keep the loss path autograd-clean.
8. **Baseline comparison** — compare AD final energy to exact (small N) and to
   classical DMRG. AD must be at/above exact within tolerance and never below
   ground by more than tolerance.

Keep `S = sigma/2`, complex128, open boundary; add a `*_score.py` and a report;
do **not** modify the default `validation_score`/`benchmark_score` lists to
depend on the new model.

---

## 10. Running validation

| Score | Covers | Command |
|---|---|---|
| Stage 1/2 physical validation | MPS overlap, dense ED | `python scripts/validation_score.py --fast` |
| Stage 2 Heisenberg benchmark | energy vs exact/DMRG scaling | `python scripts/benchmark_score.py --fast` |
| Stage 3A canonicalization | left/right/mixed, compression | `python scripts/canonical_score.py --fast` |
| Stage 3B native contractions | native norm/obs/MPO energy | `python scripts/contraction_score.py --fast` |
| Stage 4R global AD-MPS | differentiable Rayleigh + Adam | `python scripts/ad_variational_score.py --fast` |
| Stage 5A AD local optimization | center-tensor sweep + LBFGS | `python scripts/ad_local_opt_score.py --fast` |
| Stage 5B two-site AD local optimization | two-site Theta sweep + bond growth | `python scripts/ad_two_site_score.py --fast` |
| Stage 6A CPU/GPU AD benchmark | CPU/GPU parity + speedup (opt-in GPU) | `python scripts/ad_gpu_benchmark_score.py --fast` |
| DMRG baseline (classical) | two-site DMRG reference | `python scripts/dmrg_score.py --fast` |
| GPU smoke (opt-in) | device handling | `python scripts/gpu_score.py --fast` |

Run all fast scores at once (no GPU): `bash scripts/run_all_fast_scores.sh`.

List per-stage files: `python scripts/<name>_score.py --list`.

> Stage 6A GPU portion: set `LATTICETN_RUN_GPU=1` (and have CUDA available) to
> run the GPU columns; otherwise it clean-skips and the CPU benchmark still
> passes. Uses `cuda:0` (the machine's single GPU).

---

## 11. Common pitfalls

- **`S = sigma/2` vs Pauli.** Spin operators are `S = sigma/2` (e.g.
  `spin_operators()`). Pauli matrices (`pauli_matrices()`) are **not** `S`.
  Never silently mix them; the Heisenberg Hamiltonian is `J·Σ S_i·S_{i+1}`.
- **Open boundary.** Bonds 0 and N are size 1; no periodic support in the main
  paths. Old periodic prototypes live under `legacy/` and are not used.
- **dtype/device.** Default `torch.complex128` on CPU. Keep `dtype`/`device`
  identical across an MPS, MPO, and operators — mixing CPU/CUDA tensors raises
  in a contraction.
- **Below-ground energy.** A variational energy **below** exact ground by more
  than tolerance is physically wrong (a bug, not a win). Pause and investigate
  if you see it.
- **Dense path in scalable tests.** `to_dense` / dense observables are
  exponential — use them only for small-N cross-checks, never inside scalable
  tests or the training loop.
- **Treating SVD/Lanczos/`eigh` as the AD solver.** They are **not**. The
  optimizer is `loss.backward()` + a torch optimizer step. SVD/QR/
  canonicalization are post-step stabilization; DMRG/Lanczos/`eigh` are
  reference baselines.
- **Breaking autograd.** In a differentiable loss path, do **not** use
  `.detach()`, `.data`, `torch.no_grad()`, or unnecessary `.item()` — they cut
  the gradient graph. Keep all such calls in explicitly-marked post-step /
  diagnostics helpers (outside the loss). The AST tests enforce this.

---

## 12. Recommended workflow

1. **Small exact test first** — build the MPS/MPO, compare `to_dense` overlap
   and dense energy to exact ED for N ≤ 6.
2. **Native contraction** — switch to `rayleigh_energy_native` / native
   observables; confirm they match the dense values.
3. **Global AD** — train with `ADVariationalMPS` / `train_ad_mps`; confirm
   convergence to exact within tolerance, no below-ground.
4. **Local AD** — train with `train_ad_local`; confirm it matches the global AD
   minimum and DMRG reference.
5. **Benchmark** — run the relevant `*_score.py --fast`; record results.

Every meaningful change: run the smallest relevant pytest target, then the
appropriate `*_score.py --fast`, and record command/result/next action in the
relevant `docs/CLAUDE_PROGRESS_*.md`.

---

See also: [`API_OVERVIEW.md`](API_OVERVIEW.md), [`INDEX.md`](INDEX.md),
[`AD_MAINLINE_POLICY.md`](AD_MAINLINE_POLICY.md),
[`PHYSICS_SPEC.md`](PHYSICS_SPEC.md).
