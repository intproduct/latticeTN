# 01 — 快速开始：Heisenberg MPO 与能量

本教程带你从干净的代码库出发，在五分钟内得到一个可微分的 Heisenberg
能量并完成验证。这是库里最小的端到端计算：构建 MPS + MPO，计算可微
Rayleigh 能量，再和精确对角化对比。

## 目标

- 构建一个随机开边界 MPS 与 Heisenberg MPO。
- 计算可微 Rayleigh 能量 `E = <ψ|H|ψ>/<ψ|ψ>`。
- 与精确基态能量（精确对角化）对比。
- 用一句话理解什么是 **AD 主线**、什么**不是**主线。

## 一句话主线

> **AD 主线**：`MPS 参数（可训练 nn.Parameter）→ 可微 Rayleigh 商 →
> loss.backward() → torch optimizer 步进`。
> **SVD/QR/正则化** 是后处理稳定化/投影/压缩 —— **不是求解器**。
> **DMRG/Lanczos/`eigh`** 是经典参考基线 —— **不是 AD 主线**。

本教程只*计算*可微能量（还不训练）；训练见教程 02–04。

## 前置条件

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

所有默认运行都是 **CPU-only**、`torch.complex128`。若 `import torch` 失败，
见 `docs/USER_GUIDE.zh-CN.md` §1（用真实解释器，别用 Windows MS Store 的
python 占位符）。

## 最小代码

存为 `quickstart.py` 并运行：

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.contractions import rayleigh_energy_native
from latticetn.operators import heisenberg_dense, exact_ground_energy

tc.manual_seed(0)
N, chi, dtype = 6, 8, tc.complex128

mps = MPS(N, 2, chi, dtype=dtype)                                   # 随机 MPS
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)

E = rayleigh_energy_native(mps, mpo)                                # 可微
E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))

print("Rayleigh E =", float(E.real))
print("exact  E0  =", E0)
print("|E - E0|   =", abs(float(E.real) - E0))
```

## 运行命令

```bash
python quickstart.py
```

## 预期输出

随机 MPS **不是**基态，所以 `E` 会高于 `E0`（变分上界）。具体值（seed 0,
N=6, chi=8）：

```text
Rayleigh E = -0.8...   # 随机 MPS 的能量，远高于基态
exact  E0  = -2.4935771339...
|E - E0|   = 1.6...
```

N=6 开边界 Heisenberg（J=1, S=σ/2）的精确基态能量是 `-2.4935771339...`。
需要检查的要点：

- `E` **有限**且**是实数**；
- `E >= E0 - 1e-8`（变分能量绝不能低于精确基态超过容差——若低于，物理上
  一定有问题）；
- `E` 有 `requires_grad=True`（它是可微的——autograd 能把梯度流回 MPS
  张量）。

## 设计动机（写给中文用户）

为什么强调“可微”？传统张量网络（DMRG/Lanczos）靠**本征求解器**（`eigh`）
做局部更新，这割裂了梯度：你无法用 `loss.backward()` 把整条链的能量对
参数求导。latticeTN 的核心选择是把 Rayleigh 商 `E = <ψ|H|ψ>/<ψ|ψ>` 写成
**纯 einsum 扫描**（`rayleigh_energy_native`），不调用 `eigh`/`svd`/`qr`，
也不 `detach`，于是 `E` 是一个带 `requires_grad` 的标量，PyTorch autograd
能直接对它 `backward()`。这就是“AD 主线”的含义：**优化器永远是
`backward()` + `optimizer.step()`**，其余线性代数只能是后处理稳定化或参考
基线。

为什么强调 `S = σ/2`？自旋算符 `S` 与 Pauli 矩阵 `σ` 差一个因子 2，能量差
4 倍。库内统一用 `S = σ/2`（`spin_operators()`），TFI/Heisenberg 都一样，
避免混淆。

## 常见错误

- **`ModuleNotFoundError: No module named 'latticetn'`** —— 忘了在仓库根目录
  跑 `pip install -e .`。
- **`RuntimeError: Expected all tensors to be on the same device`** —— CPU MPS
  配了 GPU MPO（或反之）。MPS/MPO/算子的 `device` 必须一致。本教程全 CPU，
  保留 `device="cpu"` 即可。
- **`E` 是复数不是实数** —— `rayleigh_energy_native` 已返回 `.real`；若你
  自己算 `<ψ|H|ψ>/<ψ|ψ>`，记得取实部（Hamiltonian 厄米，虚部是数值噪声）。
- **能量差 4 倍** —— 用了 Pauli `σ` 而非 `S = σ/2`。用 `spin_operators()`，
  不要用 `pauli_matrices()`。见 `docs/PHYSICS_SPEC.md`。
- **`E` 低于 `E0`** —— 变分能量低于精确基态超过 `1e-8` 是 **bug，不是赚到**。
  复查算子约定与 dtype（`complex128`）。

## 下一步

- 把 MPS 训练到基态：教程 **02 — Global AD-MPS**。
- 局部扫描：教程 **03（单点）**、**04（两点）**。
- API 细节：`docs/API_OVERVIEW.md`（`latticetn.contractions`）。
