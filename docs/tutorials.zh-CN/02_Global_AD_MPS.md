# 02 — Global AD-MPS 训练

用 PyTorch autograd + Adam **同时**训练 MPS 的所有 site 张量，最小化可微
Rayleigh 商。这是 Stage 4R AD 主线最朴素的形式：一个全局 loss、一个优化器、
每步更新所有张量。

## 目标

- 把快速开始里的 MPS 真正**最小化**到基态。
- 在小系统下达到精确 Heisenberg 基态能量（容差内）。
- 读懂训练历史（`energy_history`、`grad_norm_history`）。
- 理解“全局”指“所有张量同时”，以及它在何处高效/低效。

## 主线（回顾）

```
MPS 参数（可训练 nn.Parameter）
   -> 可微 Rayleigh 商  E = <ψ|H|ψ>/<ψ|ψ>
   -> loss.backward()   (autograd；梯度流到每个 site 张量)
   -> torch optimizer 步进 (Adam)
   -> [后处理：逐张量 L2 重归一化，在 loss 图之外]
```

逐张量 L2 重归一化是**稳定化，不是求解器**——它在 `no_grad` 下改 `.data`，
位于可微能量路径之外。Rayleigh 商有尺度不变性，所以它不改变物理。

## 设计动机（写给中文用户）

“全局 AD-MPS” 是最直白的写法：把所有 site 张量都设成 `nn.Parameter`，
对能量标量 `backward()`，Adam 一步更新全部。优点是**实现极简、概念干净**，
是 AD 主线的“参考实现”。缺点是：所有张量共享一个 loss，一阶 Adam 要
在 N 个耦合梯度间解耦，N 变大时收敛慢。因此它适合**小系统**或作为局部
求解器的对照基线。要在大 N 下快速到机器精度，用**局部扫描**（教程 03/04）：
每次只训一个（或两个）site，用 LBFGS 几步就地收敛，再 QR 平移正交中心。

## 最小代码

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps
from latticetn.operators import heisenberg_dense, exact_ground_energy

tc.manual_seed(0)
N, chi, dtype = 6, 8, tc.complex128

mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)
ad = ADVariationalMPS(mps, mpo)                       # 所有 site 张量可训练

res = train_ad_mps(ad, num_steps=300, lr=1e-2, optimizer="adam")

E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))
print("initial E =", res["initial_energy"])
print("final   E =", res["final_energy"])
print("exact  E0 =", E0)
print("abs err   =", abs(res["final_energy"] - E0))
print("below gnd =", res["final_energy"] < E0 - 1e-8)
```

## 运行命令

```bash
# 仓库已有现成 runner，带 JSON/markdown 报告：
python scripts/run_ad_mps_heisenberg.py --N 6 --chi 8 --steps 300 --print
```

## 预期输出

`N=6, chi=8, steps=300, lr=1e-2, Adam, seed=0`：

```text
initial E = -1.2...                # 随机 MPS 起点
final   E = -2.493316...           # 接近精确
exact  E0 = -2.4935771339
abs err   = 2.6e-04                # 在 AD_TOL[6] = 1e-3 内
below gnd = False
```

`N=4, chi=8, steps=200`：`final E ≈ -1.6160253893`，`abs err ≈ 1.5e-8`
（容差 `1e-6`）。能量单调下降逼近精确值，绝不穿过基态之下。

## 常见错误

- **能量不下降** —— 检查 `optimizer="adam"`、`lr=1e-2`。LBFGS 也支持，但
  smoke 默认是 Adam；`lr` 太小或 `num_steps` 太少会停在远离极小处。
- **N=6 时 `abs err` 偏大** —— 全局 AD-MPS 是对**所有**张量做一阶 Adam；
  chi=8 在 N=6 完全收敛需要更多步（300 是 smoke 数，`1e-3` 是 smoke 容差
  `AD_TOL[6]`）。要机器精度请用**局部**求解器（教程 03/04），逐张量 LBFGS。
- **`below gnd = True`** —— 物理上不可能；变分能量低于精确基态超过 `1e-8`
  就是 bug。检查 `S = σ/2` 与 `complex128`。
- **`UserWarning: Converting a tensor with requires_grad=True to a scalar`**
  —— 这来自库的**报告路径**（`float(admps.energy())` 记录历史），**不是**
  loss 路径。预期之内、无害；loss 路径本身保持 autograd-clean。

## 何时用全局 AD-MPS

- 想要最简单的“所有张量、一个优化器”基线。
- 小系统，或作为局部求解器的对照。
- 大 N 下要更快逐键收敛，转 **单点（03）** 或 **两点（04）** AD 局部优化。
