# 用户指南（中文）

本指南解释如何**把 latticeTN 当作自动微分（AD）张量网络库来用**，面向用户
而非开发日志。

> **一句话定位**：构建一个 MPS + MPO，计算可微 Rayleigh 商
> `E = <ψ|H|ψ>/<ψ|ψ>`，用 `loss.backward()` + torch 优化器最小化它。这就是
> AD 主线。其余（SVD/QR/正则化、DMRG、Lanczos、精确对角化）都是辅助工具或
> 参考基线。

> 本指南与英文版 [USER_GUIDE.md](USER_GUIDE.md) 配套，但不是逐句翻译——
> 它面向中文用户解释**为什么这么设计**。可运行的分步教程见
> [tutorials.zh-CN/](tutorials.zh-CN/)。

---

## 1. 安装

### 1.1 创建环境

```bash
python -m venv .venv && source .venv/bin/activate     # POSIX
# 或:  conda create -n latticetn python=3.11 && conda activate latticetn
```

> Windows 上 PATH 里的 `python` 可能是坏的 MS Store 占位符——用真实解释器
> （如 conda env 的 `python.exe`）显式指定。

### 1.2 安装运行/开发依赖 + 包

```bash
pip install -r requirements.txt        # torch, numpy, scipy, tqdm, matplotlib
pip install -r requirements-dev.txt    # + pytest, pytest-timeout
pip install -e .                       # 可编辑安装 latticetn 包
```

之后 `import latticetn` 在任意目录可用。

### 1.3 CPU / GPU 说明

- 所有默认测试与 score 脚本都 **CPU-only**、`torch.complex128`。
- GPU 是 **opt-in**：刻意运行 `python scripts/gpu_score.py --fast`
  （Stage 2.5 正确性 smoke）或
  `LATTICETN_RUN_GPU=1 python scripts/ad_gpu_benchmark_score.py --fast`
  （Stage 6A CPU/GPU AD benchmark）时才用。
- 绝不在一次收缩里混用 CPU 与 CUDA 张量——MPS/MPO/算子的 `device` 要一致。
  Stage 2.5 smoke 用名称匹配选设备（多 GPU 机器）；Stage 6A benchmark 针对单
  GPU 机器，`LATTICETN_RUN_GPU=1` 且 CUDA 可用时直接用 `cuda:0`。

---

## 2. 核心概念

| 概念 | 是什么 | 在哪 |
|---|---|---|
| **MPS** | 矩阵积态；开边界；每个 site 张量 `(left, phys, right)` 是可训练 `nn.Parameter`。 | `latticetn.mps.MPS` |
| **MPO** | 矩阵积算符；site 张量 `(left, right, phys_in, phys_out)`。 | `latticetn.mpo.MPO` |
| **Rayleigh 商** | `E = <ψ|H|ψ> / <ψ|ψ>`——变分能量。 | `contractions.rayleigh_energy_native` |
| **可微收缩** | 直接收缩 MPS/MPO 张量（einsum 扫描），不用 `to_dense`；N、χ 多项式缩放；梯度流回 MPS。 | `latticetn.contractions` |
| **观测量** | `<Sz_i>`、`<Sz_i Sz_j>`、键能、纠缠熵。native（可扩展、可微）与 dense（小 N 参考）两套。 | `latticetn.contractions`、`latticetn.observables` |
| **正则化/压缩** | 规范固定（左/右/混合正则）与 SVD 键压缩——**辅助工具**，不是优化器。 | `latticetn.canonical` |
| **全局 AD-MPS** | 一次性训练**所有** MPS 张量（Stage 4R）。 | `latticetn.ad_variational` |
| **单点 AD 局部** | 每次只训**一个中心张量**，QR 平移正交中心扫描（Stage 5A）。 | `latticetn.ad_local` |
| **两点 AD 局部** | 每次只训**一个两点 block Θ**，SVD 分裂可选增长/截断键（Stage 5B）。 | `latticetn.ad_two_site` |
| **经典 DMRG / Lanczos** | 参考基线/预言机，**绝不是** AD 主线。 | `latticetn.dmrg`、`latticetn.lanczos` |

### 2.1 AD 主线（唯一的求解器）

```
MPS 参数（可训练 nn.Parameter）
   -> 可微 Rayleigh 商  E = <ψ|H|ψ>/<ψ|ψ>
   -> loss.backward()        (PyTorch autograd)
   -> torch optimizer 步进   (Adam / LBFGS)
   -> [可选后处理稳定化: none|tensor_norm|qr|canonical]
```

> **设计动机**：传统张量网络（DMRG/Lanczos）靠**本征求解器**（`eigh`）做局部
> 更新，割裂了梯度——无法把整条链的能量对参数求导。latticeTN 的核心选择是
> 把 Rayleigh 商写成**纯 einsum 扫描**（`rayleigh_energy_native`），不调
> `eigh`/`svd`/`qr`、不 `detach`，于是 `E` 是带 `requires_grad` 的标量，
> `backward()` 能直接对它求导。**优化器永远是 `backward()` +
> `optimizer.step()`**，其余线性代数只能是后处理稳定化或参考基线。

损失路径（`rayleigh_energy_native`）**autograd-clean**：不用 `detach()`、
`.data`、`torch.no_grad()`、不必要的 `.item()`，不调 `eigh`/`svd`/`qr`，绝不
调 `dmrg`/`lanczos`。由 AST 测试强制。

### 2.2 SVD/QR/正则化是什么、不是什么

它们是**可选的后处理稳定化/投影/压缩**工具：规范固定、条件数改善、键截断、
诊断。在 `torch.no_grad()` 下改 `.data`，**在 loss 图之外**，对 Rayleigh 商
尺度/规范不变。它们**绝不是优化器**。若你想用 `svd`/`eigh`/`qr` 去“解”一个
张量，你就离开了 AD 主线。

### 2.3 DMRG / Lanczos 是什么

参考基线/预言机：经典两点 DMRG 与 Krylov 局部本征求解器存在的目的是**正确性
对比与 benchmark**，不是项目的求解器。它们在各自的 opt-in score 脚本之后，
AD 模块绝不 import 它们。

### 2.4 教程（分步、可运行）

本指南是参考手册；要动手跑（含预期输出与常见错误），见教程：

| # | 教程 | 内容 |
|---|---|---|
| 01 | [快速开始 (Heisenberg)](tutorials.zh-CN/01_快速开始_Heisenberg.md) | 构建 MPS+MPO、算可微能量、对比 ED |
| 02 | [Global AD-MPS](tutorials.zh-CN/02_Global_AD_MPS.md) | 一次性训练所有张量 (Adam) |
| 03 | [单点 AD 局部](tutorials.zh-CN/03_One_site_AD_local.md) | 中心张量扫描 (LBFGS)、QR 平移 |
| 04 | [两点 AD 局部](tutorials.zh-CN/04_Two_site_AD_local.md) | 两点 Θ block、SVD 分裂增长键 |
| 05 | [CPU/GPU benchmark](tutorials.zh-CN/05_CPU_GPU_benchmark.md) | opt-in GPU 一致性 + speedup |
| 06 | [添加新模型](tutorials.zh-CN/06_添加新模型.md) | dense 参考 → MPO → 测试 → AD 求解器 → benchmark |

也可作本地站点浏览：`pip install -r requirements-docs.txt && mkdocs serve`
（见 `mkdocs.yml`）。

---

## 3. 最小示例：Heisenberg MPO + 能量

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.contractions import rayleigh_energy_native
from latticetn.operators import heisenberg_dense, exact_ground_energy

tc.manual_seed(0)
N, chi, dtype = 6, 8, tc.complex128
mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)

E = rayleigh_energy_native(mps, mpo)                      # 可微
E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))
print("Rayleigh E =", float(E.real), " exact E0 =", E0)
```

随机 MPS 不是基态，所以 `E > E0`（变分上界）。关键检查：`E` 有限且实、
`E >= E0 - 1e-8`、`E.requires_grad=True`。详见教程 01。

### 3.1 无自旋费米子 t-V 链（Stage 7A，Jordan-Wigner）

Stage 7A 在**不变**的 AD 主线之上增加了开边界一维无自旋费米子 t-V 链——
损失路径与算子无关，所以只需替换 Hamiltonian/MPO。这是 1D Jordan-Wigner
费米子，**不是**完整的 graded 费米子张量网络；JW 宇称串 `F = (-1)^n` 是关键
（它让不同格点上的费米子算子反对易）。

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.contractions import rayleigh_energy_native
from latticetn.operators import spinless_fermion_dense, exact_ground_energy
from latticetn.fermion_operators import fermion_operators

N, chi, dtype = 6, 8, tc.complex128
# H = -t sum (c†_i c_{i+1}+h.c.) + V sum (n_i-1/2)(n_{i+1}-1/2) - mu sum (n_i-1/2)
mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_spinless_fermion(
    t=1.0, V=0.5, mu=0.0)

E = rayleigh_energy_native(mps, mpo)            # 同一条可微损失路径
E0, _ = exact_ground_energy(
    spinless_fermion_dense(N, t=1.0, V=0.5, mu=0.0, dtype=dtype, device="cpu"))
print("E =", float(E.real), " exact E0 =", E0)

# 费米子可观测量（dense 参考；近邻 hopping 带有 JW 串）
from latticetn.observables import (mps_fermion_local_density,
    mps_fermion_nn_hopping)
print("<n_0> =", float(mps_fermion_local_density(mps, 0).real))
print("<c†_0 c_1 + h.c.> =", float(mps_fermion_nn_hopping(mps, 0).real))
```

三个 AD 主线 solver（`ad_variational.train_ad_mps`、`ad_local.train_ad_local`、
`ad_two_site.train_ad_two_site`）在费米子 MPO 上不变即可用。CPU/GPU 计时使用
**统一 GPU 选择器**（`scripts/gpu_selector.py`），只选 V100/TITAN V，否则
clean-skip（不回退）。运行 `python scripts/fermion_score.py --fast`（CPU）或
`LATTICETN_RUN_GPU=1 python scripts/fermion_score.py --fast`（opt-in GPU）。

### 3.2 通用一维模型构建器 + 基准注册表（Stage 7B）

Stage 7B 把 Heisenberg 与无自旋费米子 t-V 抽象成**统一的一维模型构建层**
（`latticetn/model_builder.py`）。这是**模型/MPO 构建层，不是新求解器**——AD
主线不变；构建器分发到既有已验证的生成器，物理与 Stage 1/7A 逐字节一致。

```python
import torch as tc
from latticetn.model_builder import (heisenberg_model,
    spinless_fermion_tv_model, build_dense, build_mpo)
from latticetn.benchmarking import benchmark_model

# 构造模型 spec，再构造 dense + MPO Hamiltonian。
spec = spinless_fermion_tv_model(N=6, t=1.0, V=0.5, mu=0.0)
H = build_dense(spec)          # == operators.spinless_fermion_dense（Stage 7A）
mpo = build_mpo(spec)          # == MPO.generate_spinless_fermion（Stage 7A）

# 统一 CPU/GPU 基准注册表（Stage 7A+ 计时契约）。
r = benchmark_model(spec, chi=8, seed=0, steps=120)
# r["cpu"]、r["gpu"]（或 None + skip 原因）、r["speedup"]、r["exact_energy"]……
```

`ModelSpec` 带有显式 `statistics`（"boson"/"fermion"）和项列表（`OnsiteTerm`、
`TwoSiteTerm` 用于玻色/自旋；`FermionHopTerm` + `DensityDensityTerm` 用于
费米/JW）。费米子项保留 JW 宇称串（不退化为 hard-core boson）。运行
`python scripts/model_builder_score.py --fast`（CPU）或
`LATTICETN_RUN_GPU=1 python scripts/model_builder_score.py --fast`（opt-in GPU，
V100/TITAN V）。

### 3.3 自旋 Hubbard 链（Stage 7C，Jordan-Wigner）

Stage 7C 在**不变的** AD 主线之上加入开放边界一维**自旋 Hubbard 链**。模型为

```
H = -t Σ_{i,s}(c†_{i,s} c_{i+1,s} + h.c.)
    + U Σ_i (n_{i,↑}-1/2)(n_{i,↓}-1/2)
    - μ Σ_i (n_{i,↑}+n_{i,↓}-1) - h Σ_i (n_{i,↑}-n_{i,↓})
```

局域基 `|0>、|↑>、|↓>、|↑↓>`（d=4）；全局模式排序固定为**site-major**
`(0↑,0↓,1↑,1↓,…)`。这是一维 Jordan-Wigner 费米子，**不是**完整的 graded
fermionic 张量网络；逐位 JW 宇称 `P = F_↑ ⊗ F_↓`（加上局域 `cdown`/`cdagdown`
内部已含的 `F_↑`）是关键。

```python
import torch as tc
from latticetn.model_builder import hubbard_model, build_dense, build_mpo
from latticetn import contractions as K
from latticetn.operators import hubbard_dense, exact_ground_energy

N, chi, dtype = 4, 4, tc.complex128
spec = hubbard_model(N, t=1.0, U=4.0, mu=0.0, h=0.0)
H = build_dense(spec)          # == operators.hubbard_dense（完整 2N 模 JW）
mpo = build_mpo(spec)          # == MPO.generate_hubbard（bond 维 6，d=4）
assert tc.allclose(mpo.to_dense(), H, atol=1e-12)

# 可微 Rayleigh 能量（与算子无关的 AD 主线）：
tc.manual_seed(0)
from latticetn.mps import MPS
mps = MPS(N, 4, chi, dtype=dtype)
E = K.rayleigh_energy_native(mps, mpo)        # <ψ|H|ψ>/<ψ|ψ>

# 参考基线（不是 AD 路径）：
E0, _ = exact_ground_energy(hubbard_dense(N, t=1.0, U=4.0, dtype=dtype))

# Hubbard 观测量（dense 参考；自旋分辨 NN 跃迁携带左因子位的存活 P）：
from latticetn.observables import (mps_hubbard_local_density,
    mps_hubbard_double_occ, mps_hubbard_local_sz, mps_hubbard_nn_hopping)
print("<n_↑_0>  =", float(mps_hubbard_local_density(mps, 0, "up").real))
print("<docc_0> =", float(mps_hubbard_double_occ(mps, 0).real))
print("<Sz_0>   =", float(mps_hubbard_local_sz(mps, 0).real))
print("<c†_{0,↑} c_{1,↑}+h.c.> =",
      float(mps_hubbard_nn_hopping(mps, 0, "up").real))
```

三个 AD 主线求解器（`train_ad_mps`、`train_ad_local`、`train_ad_two_site`）
在 Hubbard MPO 上原样工作。CPU/GPU 计时使用**统一 GPU 选择器**（仅
V100/TITAN V，不回退）。运行 `python scripts/hubbard_score.py --fast`（CPU）
或 `LATTICETN_RUN_GPU=1 python scripts/hubbard_score.py --fast`（opt-in GPU）。

---

## 4. 全局 AD-MPS 训练（Stage 4R）

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps

tc.manual_seed(0)
mps = MPS(6, 2, 8, dtype=tc.complex128)
mpo = MPO.from_bonds(6, 2, dtype=tc.complex128, device="cpu").generate_heisenberg(J=1.0)

ad = ADVariationalMPS(mps, mpo)                 # 所有 site 张量可训练
res = train_ad_mps(ad, num_steps=300, lr=1e-2, optimizer="adam",
                   projection="tensor_norm")    # 后处理 L2 重归一化
print("final E =", res["final_energy"])
```

> **设计动机**：全局 AD-MPS 是最朴素的写法——所有张量共享一个 loss，Adam
> 一步更新全部。实现极简、概念干净，是 AD 主线的“参考实现”。缺点：大 N 时
> 一阶 Adam 在 N 个耦合梯度间收敛慢。要大 N 快速到机器精度，用局部扫描
> （§5/§5b）。

`train_ad_mps` 每步：清梯度 → `ad.loss()`（= `rayleigh_energy_native`）→
`backward()` → 优化器步进 → 可选后处理 `projection`（在 `no_grad` 下）。
`projection` 是**稳定化，不是求解器**。详见教程 02。

---

## 5. 单点 AD 局部优化（Stage 5A）

冻结除一个**中心张量**外所有张量，优化它，QR 平移正交中心，重复扫描。

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
                     optimizer="lbfgs",          # LBFGS：局部问题近二次
                     stabilization="qr")         # 可选后处理投影
print("final E =", res["final_energy"])
```

> **设计动机**：局部思路——**冻结除中心外所有张量**，只在低维中心上最小化
> Rayleigh 商（几步 LBFGS），再 QR 平移中心到下一 site，重复一个 sweep。由于
> 链在中心周围正交，局部商**等于**全局 Rayleigh 商，最小化它即降低全局能量。
> 逐键 LBFGS 条件数远好于对所有张量做一阶 Adam，所以小-中 N 一秒到机器精度。

**优化器 vs 辅助**：
- **优化器** = `loss.backward()` + torch 优化器对中心张量的步进。
- `stabilization`（`none`/`tensor_norm`/`qr`/`canonical`）是**可选后处理稳定
  化**，在 `no_grad` 下改 `.data`——规范/条件数辅助，**不是**求解器。
- `move_center` 里的 QR 是**正交中心平移**（规范传输），不是优化。

低层接口（`ADLocalOptimizer`、`.move_center`、`.set_center`）见
[API_OVERVIEW.md](API_OVERVIEW.md)。教程 03。

---

## 5b. 两点 AD 局部优化（Stage 5B）

把局部扫描扩展到**两点 block**：把两个相邻 site 张量收缩成单一可训练
`Θ(l, s_i, s_{i+1}, r)`，在可微局部 Rayleigh 商 `E(Θ)=<Θ|H_eff|Θ>/<Θ|Θ>` 上
训练，再 SVD 分裂回两个 site 张量，可选 `max_bond_dim`/`cutoff`。这是两点
DMRG 的 autograd 对应物——**对 `Θ` 做梯度下降，而非局部本征求解**——且让键维
在扫描中**增长或截断**。

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
                        max_bond_dim=8, cutoff=None)   # 可选键增长/上限
print("final E =", res["final_energy"])
print("bond dims =", res["final_bond_dims"], "max =", res["max_bond"])
```

> **设计动机**：单点保持 χ 固定，无法适应纠缠。两点把两个 site 收缩成
> `Θ`，优化后 SVD 分裂，**分裂可增长或截断键**，于是 ansatz 自适应纠缠——
> 和两点 DMRG 一样，但局部更新是梯度下降。SVD 分裂是**后处理压缩**（`no_grad`
> 下 detached 数据），**不是求解器**；求解器始终是 `backward()` + 对 `Θ` 的
> 优化器步进。

`H_eff` 由**冻结、detached** 的左右 MPO 环境 + 两个 MPO 张量构成；`Θ` 是唯一
可训练叶。损失是纯 einsum。链其余部分正交时 `E(Θ)` 等于全局 Rayleigh 商。
要键自适应用两点；要固定 χ 用单点（§5）。两者小 N 均到机器精度。教程 04。

---

## 6. 观测量与纠缠

### 6.1 native（可扩展、可微）观测量

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

直接收缩 MPS（N、χ 多项式）——在可扩展代码与 loss 里用。

### 6.2 纠缠熵（正则、不可微）

```python
from latticetn.canonical import entanglement_entropy
S_vN = entanglement_entropy(mps, cut=3)   # nats，跨键 [0..3)|[3..N)
```

把 MPS 带到混合正则形式读 Schmidt 谱——**诊断**，不在 loss 里。

### 6.3 dense（小 N 参考）观测量

```python
from latticetn.observables import mps_expect_local, mps_bond_energy_heisenberg, mps_entanglement_entropy
from latticetn.operators import exact_ground_energy, heisenberg_dense
```

dense 变体仅用于小系统/与精确对角化交叉验证；不扩展，不在可扩展训练路径里。

### 6.4 native vs dense

- **native**（`contractions`）：直接收缩 MPS 张量——可扩展、可微、训练与可扩
  展测试用。
- **dense**（`observables`、`to_dense`）：重建全态矢量——精确但 N 指数；仅小 N
  验证用。

别把 dense 路径放进可扩展测试或训练循环。

---

## 7. 正则化与压缩（辅助，不是优化器）

```python
from latticetn import canonical as Can

lc  = Can.left_canonical(mps)          # 精确 QR 扫描；态保持（差一个相位）
rc  = Can.right_canonical(mps)
mc  = Can.mixed_canonical(mps, center=3)
errs = Can.left_orthonormal_all(mps)   # 规范诊断

compressed, info = Can.svd_compress(mps, chi=16)
print(info["truncation_errors"], info["bond_dims"])
```

**何时用**：局部求解前规范固定、改善漂移态条件数、内存键压缩、诊断（正则误差、
截断误差）。**绝不**作为优化机制——它们不可微，在 `no_grad` 下。

---

## 8. 参考求解器（仅正确性/benchmark）

### 8.1 精确对角化

```python
from latticetn.operators import heisenberg_dense, exact_ground_energy
H = heisenberg_dense(8, dtype=tc.complex128, device="cpu")
E0, psi0 = exact_ground_energy(H)     # 金标准（仅小 N）
```

### 8.2 经典 DMRG

```python
from latticetn import dmrg as D
r = D.run_dmrg(mps, mpo, chi=16, num_sweeps=4, solver="dense")
print("DMRG E =", r["final_energy"], " below ground?", r["below_ground"])
```

### 8.3 Lanczos

`latticetn.lanczos.lanczos_lowest_eigenpair` 是 DMRG `solver="lanczos"` 用的
Krylov 局部本征求解器。它是参考工具，**不是** AD 求解器。

> **切记**：精确/DMRG/Lanczos 存在是为了*检查* AD 求解器到了正确的变分极小。
> 它们不是项目主线。绝不在 AD loss 路径里 import 它们。

---

## 9. 添加新模型

完整可运行流程（dense 参考 → MPO → MPO-to-dense 测试 → native 能量测试 →
AD 求解器 → benchmark，含代码与预期输出）见教程
[06 — 添加新模型](tutorials.zh-CN/06_添加新模型.md)。摘要：加一个 XXZ 链——

1. **dense 参考 Hamiltonian** —— 在 `latticetn/operators.py` 加
   `xxz_dense(N, Jz, Jxy, ...)`（用 `spin_operators`，`S = sigma/2`）。
2. **MPO** —— 在 `latticetn/mpo.py` 加 `.generate_xxz(...)`。
3. **MPO-to-dense 测试** —— 在 `tests/` 断言 MPO 复现 `xxz_dense`
   （仿 `test_heisenberg_mpo_dense.py`）。
4. **native 能量测试** —— 断言 `rayleigh_energy_native` 在已知态上匹配 dense
   （仿 `test_native_mpo_energy_contraction.py`）。
5. **观测量** —— 按需加键能/关联，带 dense 交叉验证。
6. **benchmark 脚本** —— `scripts/run_<model>_benchmark.py` + score。
7. **AD 求解器** —— 用 `ADVariationalMPS`/`train_ad_local`/`train_ad_two_site`
   在新 MPO 上训练；保持 loss 路径 autograd-clean。
8. **基线对比** —— AD 最终能量 vs 精确（小 N）vs 经典 DMRG。AD 须在/高于
   精确（容差内），绝不低于基态超过容差。

保持 `S = sigma/2`、complex128、开边界；加 `*_score.py` 与报告；**不要**改默认
`validation_score`/`benchmark_score` 列表去依赖新模型。

> **设计动机**：加模型而非加算法，是因为 latticeTN 的核心赌注是：**只要
> Hamiltonian 能写成 MPO，可微 Rayleigh 商 + autograd 就能直接优化**。所以扩
> 展点是“提供新 Hamiltonian 表达”，不是“写新求解器”。这也要求新模型同时给
> dense 参考（小 N 金标准）与 MPO（可扩展可微），并用 MPO-to-dense 测试证明
> 一致——否则无法相信 MPO 路径的数值。

---

## 10. 运行验证

| Score | 覆盖 | 命令 |
|---|---|---|
| Stage 1/2 物理验证 | MPS overlap、dense ED | `python scripts/validation_score.py --fast` |
| Stage 2 Heisenberg benchmark | 能量 vs 精确/DMRG 缩放 | `python scripts/benchmark_score.py --fast` |
| Stage 3A 正则化 | 左/右/混合、压缩 | `python scripts/canonical_score.py --fast` |
| Stage 3B native 收缩 | native norm/obs/MPO 能量 | `python scripts/contraction_score.py --fast` |
| Stage 4R 全局 AD-MPS | 可微 Rayleigh + Adam | `python scripts/ad_variational_score.py --fast` |
| Stage 5A 单点 AD 局部 | 中心张量扫描 + LBFGS | `python scripts/ad_local_opt_score.py --fast` |
| Stage 5B 两点 AD 局部 | 两点 Θ 扫描 + 键增长 | `python scripts/ad_two_site_score.py --fast` |
| Stage 6A CPU/GPU AD benchmark | CPU/GPU 一致性 + speedup (opt-in GPU) | `python scripts/ad_gpu_benchmark_score.py --fast` |
| DMRG 基线（经典） | 两点 DMRG 参考 | `python scripts/dmrg_score.py --fast` |
| GPU smoke (opt-in) | 设备处理 | `python scripts/gpu_score.py --fast` |

一次跑全部 fast score（无 GPU）：`bash scripts/run_all_fast_scores.sh`。

列各阶段文件：`python scripts/<name>_score.py --list`。

> Stage 6A GPU 部分：设 `LATTICETN_RUN_GPU=1`（且 CUDA 可用）才跑 GPU 列；否则
> clean-skip，CPU benchmark 照样过。用 `cuda:0`（本机单 GPU）。

---

## 11. 常见陷阱

- **`S = sigma/2` vs Pauli。** 自旋算符 `S = sigma/2`（如 `spin_operators()`）。
  Pauli 矩阵（`pauli_matrices()`）**不是** `S`。绝不混用；Heisenberg
  Hamiltonian 是 `J·Σ S_i·S_{i+1}`，混用会差 4 倍。
- **开边界。** 键 0 与 N 是 size 1；主路径不支持周期。旧的周期原型在
  `legacy/`，不用。
- **dtype/device。** 默认 `torch.complex128`、CPU。MPS/MPO/算子的
  `dtype`/`device` 要一致——混 CPU/CUDA 张量在收缩里会报错。
- **below-ground 能量。** 变分能量**低于**精确基态超过容差是物理错误
  （bug，不是赚到）。见到就停下排查。
- **可扩展测试里的 dense 路径。** `to_dense`/dense 观测量是指数的——仅小 N
  交叉验证，绝不放进可扩展测试或训练循环。
- **把 SVD/Lanczos/`eigh` 当 AD 求解器。** **不是**。优化器是
  `loss.backward()` + torch 优化器步进。SVD/QR/正则化是后处理稳定化；
  DMRG/Lanczos/`eigh` 是参考基线。
- **破坏 autograd。** 可微 loss 路径里**不要**用 `.detach()`、`.data`、
  `torch.no_grad()`、不必要的 `.item()`——它们切断梯度图。把这些调用留在
  明确标记的后处理/诊断辅助里（loss 之外）。AST 测试强制。

---

## 12. 推荐工作流

1. **先小精确测试** —— 构建 MPS/MPO，N ≤ 6，对比 `to_dense` overlap 与 dense
   能量到精确 ED。
2. **native 收缩** —— 换 `rayleigh_energy_native`/native 观测量；确认匹配
   dense 值。
3. **全局 AD** —— 用 `ADVariationalMPS`/`train_ad_mps` 训练；确认收敛到精确
   （容差内）、无 below-ground。
4. **局部 AD** —— 用 `train_ad_local`（单点）与 `train_ad_two_site`（两点）；
   确认匹配全局 AD 极小与 DMRG 参考。
5. **benchmark** —— 跑相关 `*_score.py --fast`；记录结果。

每次有意义的改动：跑最小相关 pytest 目标，再跑对应 `*_score.py --fast`，把
命令/结果/下一步记在 `docs/CLAUDE_PROGRESS.md`。`/latticetn-autovalidate`
skill 自动化全套验证流程。

---

另见：[API_OVERVIEW.md](API_OVERVIEW.md)、[INDEX.md](INDEX.md)、
[AD_MAINLINE_POLICY.md](AD_MAINLINE_POLICY.md)、[PHYSICS_SPEC.md](PHYSICS_SPEC.md)。
