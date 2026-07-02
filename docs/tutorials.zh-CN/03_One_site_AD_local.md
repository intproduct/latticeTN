# 03 — 单点 AD 局部优化

每次只训练**一个正交中心张量**，用 QR 把正交中心沿链平移扫描。这是
Stage 5A AD 主线：单点 DMRG 的 autograd 对应物，**用梯度下降替代局部
本征求解器**。

## 目标

- 理解局部扫描为何在大 N 下比全局 Adam 收敛更好。
- 跑 `train_ad_local`，小系统达到机器精度。
- 认清 QR 是**正交中心平移，不是求解器**。

## 设计动机（写给中文用户）

全局 AD-MPS（教程 02）让所有张量共享一个 loss，优化器要在 N 个耦合梯度间
解耦，一阶 Adam 在大 N 收敛慢。局部思路：**冻结除中心张量外所有张量**，
只在（低维的）中心张量上最小化 Rayleigh 商（用几步 LBFGS），再把中心 QR
平移到下一个 site，重复——一个 sweep。由于链在中心周围是正交的，局部商
**等于**全局 Rayleigh 商（标准变分原理），所以最小化它就降低全局能量。

关键点：“平移中心”用的是 **QR**，而这步 QR 是**规范固定（gauge fixing），
不是优化器**。它在 `no_grad` 下改 `.data`，在 loss 图之外。求解器始终是
`loss.backward()` + torch 优化器对中心张量的步进。

## 主线（回顾，局部形式）

```
混合正则 MPS，正交中心在 site c
   -> 只有中心张量可训练 (nn.Parameter)
   -> loss = <ψ|H|ψ>/<ψ|ψ>  (可微；等于全局商)
   -> loss.backward() + optimizer 步进 (LBFGS) 只更新中心
   -> QR 平移中心 c -> c+1  (规范固定，在 loss 图之外)
   -> 左→右再右→左扫描
```

## 最小代码

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_local import train_ad_local
from latticetn.operators import heisenberg_dense, exact_ground_energy

tc.manual_seed(0)
N, chi, dtype = 6, 8, tc.complex128

mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)

res = train_ad_local(mps, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                     optimizer="lbfgs", stabilization="qr")

E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))
print("final   E =", res["final_energy"])
print("exact  E0 =", E0)
print("abs err   =", abs(res["final_energy"] - E0))
```

## 运行命令

```bash
python scripts/run_ad_local_opt.py --N 6 --chi 8 --num-sweeps 4 --local-steps 20 --print
```

## 预期输出

`N=6, chi=8, 4 sweeps, 20 local_steps, lr=1.0, LBFGS, stabilization='qr',
seed=0`：

```text
final   E = -2.4935771330
exact  E0 = -2.4935771339
abs err   = 8.9e-10            # 在 AD_LOCAL_TOL[6] = 1e-5 内（机器精度）
```

`N=4`：`final E ≈ -1.6160254037`，`abs err ≈ 4.2e-11`（容差 `1e-8`）。
对比全局 AD-MPS 在 N=6 的 `2.6e-4`：局部扫描**约一秒**就到机器精度——
逐键 LBFGS 的条件数远好于对所有张量做一阶 Adam。

## 常见错误

- **`stabilization="qr"` vs `"none"`** —— `qr` 在中心平移时重正则化链，保持
  扫描良好条件数；`none` 跳过此步，长链可能漂移。smoke 默认 `"qr"`。
- **扫描间能量反而上升** —— 多半传了 `optimizer="adam"` 且 `lr` 不适合局部
  问题；默认/smoke 选 LBFGS，因为局部 Rayleigh 问题对中心张量近二次。
- **QR 报 `RuntimeError`** —— CPU `complex128` 不应出现；若在 GPU 上见到，
  见教程 05（GPU QR/SVD 可用，但小系统开销主导）。
- **把 QR 当成求解器** —— QR 是**正交中心平移**，在 `no_grad` 下。求解器是
  `backward()` + `optimizer.step()`。若你发现自己在 loss 路径里调 QR/SVD，
  停下——这违反 autograd 规则。

## 何时用单点 AD 局部

- 想要小-中 N 下**固定**键维的机器精度基态。
- 不需要键增长（单点保持 χ 固定）。
- 要键增长/截断，见教程 **04 — 两点 AD 局部**。
