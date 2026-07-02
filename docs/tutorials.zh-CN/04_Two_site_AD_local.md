# 04 — 两点 AD 局部优化

每次只训练**一个两点 block `Θ`**，最小化可微局部 Rayleigh 商
`E(Θ)=<Θ|H_eff|Θ>/<Θ|Θ>`，扫描活动键，并在 SVD 分裂时**可选地增长或截断
键维**。这是 Stage 5B AD 主线：两点 DMRG 的 autograd 对应物——**对 `Θ`
做梯度下降，而非局部本征求解**。

## 目标

- 理解“两点”指一个可训练的两 site block `Θ`。
- 跑 `train_ad_two_site`，小系统达到机器精度。
- 认清 SVD 分裂是**压缩，不是求解器**。
- 可选地增长键维（`max_bond_dim`）逼近精确纠缠结构。

## 设计动机（写给中文用户）

单点 AD（教程 03）保持键维**固定**——你无法让 χ 适应纠缠结构。两点 AD 把
两个相邻 site 张量收缩成一个可训练 block `Θ(l, s_i, s_{i+1}, r)`，优化它，
再 **SVD 分裂**回两个 site 张量，带可选 `max_bond_dim`/`cutoff`。分裂可以
**增长或截断**键，于是 ansatz 在扫描中自适应纠缠——和两点 DMRG 一样，
但局部更新是对 `Θ` 的梯度下降，不是局部 `eigh`。

SVD 分裂是**后处理压缩**，在 `no_grad` 下对 detached 数据操作，**在 loss
图之外**，**不是求解器**。求解器始终是 `loss.backward()` + torch 优化器对
`Θ` 的步进。

## 主线（回顾，两点形式）

```
两点混合正则 MPS，活动键在 (i, i+1)
   -> 冻结左右 MPO 环境 L, R  (常数，no_grad 下)
   -> Θ = A_i * A_{i+1}        (单一可训练叶)
   -> loss = <Θ|H_eff|Θ>/<Θ|Θ> (可微 einsum)
   -> loss.backward() + optimizer 步进 (LBFGS) 只更新 Θ
   -> [后处理分裂：SVD Θ -> A_i, A_{i+1}，可选 max_bond_dim/cutoff]
   -> 在下一个键重正则化 (QR 规范固定，no_grad)
   -> 左→右再右→左扫描
```

由于链其余部分正交，`E(Θ)` **等于**全局 Rayleigh 商——最小化它即降低全局
能量。

## 最小代码

```python
import torch as tc
from latticetn.mps import MPS
from latticetn.mpo import MPO
from latticetn.ad_two_site import train_ad_two_site
from latticetn.operators import heisenberg_dense, exact_ground_energy

tc.manual_seed(0)
N, chi, dtype = 6, 8, tc.complex128

mps = MPS(N, 2, chi, dtype=dtype)
mpo = MPO.from_bonds(N, 2, dtype=dtype, device="cpu").generate_heisenberg(J=1.0)

res = train_ad_two_site(mps, mpo, num_sweeps=4, local_steps=20, lr=1.0,
                        optimizer="lbfgs", max_bond_dim=8, cutoff=None)

E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=dtype, device="cpu"))
print("final   E =", res["final_energy"])
print("exact  E0 =", E0)
print("abs err   =", abs(res["final_energy"] - E0))
print("final bond dims =", res["final_bond_dims"])
```

## 运行命令

```bash
python scripts/run_ad_two_site.py --N 6 --chi 8 --num-sweeps 4 --local-steps 20 --print
```

## 预期输出

`N=6, chi=8, 4 sweeps, 20 local_steps, lr=1.0, LBFGS, max_bond_dim=8,
cutoff=None, seed=0`：

```text
final   E = -2.4935771330
exact  E0 = -2.4935771339
abs err   = 8.4e-10            # 在 AD_TWO_SITE_TOL[6] = 1e-5 内（机器精度）
final bond dims = [2, 4, 8, 4, 2]
```

`N=4`：`final E ≈ -1.6160254036`，`abs err ≈ 1.8e-10`。键增长到
`[2, 4, 2]`（max 4）——精确基态的纠缠结构，完全由对 `Θ` 的梯度下降 +
SVD 分裂恢复。

## 常见错误

- **把 SVD 分裂当成求解器** —— `split()` 里的 SVD 是**压缩**，在 `no_grad`
  下。求解器是 `backward()` + 对 `Θ` 的优化器步进。若把 SVD/`eigh` 放进
  loss 路径，就违反 autograd 规则。
- **`max_bond_dim` 太小** —— 截断键、抬高能量误差。N=6 精确链需要 χ≥8；
  限到 4 到不了机器精度。设 `max_bond_dim` ≥ 预期纠缠，或 `None` 全增长。
- **截断误差全为 0** —— 键已在 `max_bond_dim` 以下饱和（无物可截）时正常。
  截断非零出现在你把 χ 限到自然键之下时。
- **`stabilization` 只接受 `none|tensor_norm`** —— 两点的选择比单点窄
  （无 `qr`/`canonical`）；SVD 分裂已处理规范，不需要 `qr`。

## 何时用两点 AD 局部

- 想要**键增长/自适应**（ansatz 经 SVD 分裂 + `max_bond_dim`/`cutoff`
  自选 χ）。
- 想要机器精度基态与 DMRG 式工作流，但局部更新用 autograd。
- 与 DMRG 参考对照（`dmrg.run_dmrg`）——两者应到同一变分极小（DMRG 是
  **参考基线**，绝非 AD 主线）。
