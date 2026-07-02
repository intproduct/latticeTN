# 05 — CPU/GPU benchmark

跑统一 benchmark，比较三个 AD 主线求解器（global AD-MPS、单点 AD 局部、
两点 AD 局部）在 CPU 与本机单 GPU 上的表现，精确对角化与 DMRG 仅作**参考
基线**。

## 目标

- 在 CPU（总是）和 GPU（opt-in）上跑 benchmark。
- 理解 GPU opt-in 规则（`LATTICETN_RUN_GPU=1`，`cuda:0`）。
- 读懂 CPU/GPU 能量一致性与 speedup 列。
- 内化关键提醒：**小系统上 GPU 可能比 CPU 慢**——speedup 只作趋势观察。

## 主线定位

benchmark **评估的是 AD 主线求解器**。**DMRG / Lanczos / 精确对角化是经典
参考基线**，不在 AD 优化路径中；仅为对照而跑，且在 CPU 上。GPU 只跑 AD
求解器优化。

## GPU opt-in 规则

本机只有一个 GPU，所以**不做名称过滤**（不同于 Stage 2.5 的多 GPU smoke）。
规则：

1. GPU 是 **opt-in**：仅当 `LATTICETN_RUN_GPU=1` 才跑。
2. opt-in 且 `torch.cuda.is_available()` 且 `device_count() > 0` 时，用
   `cuda:0`。
3. opt-in 但无可见 CUDA 设备 → **clean-skip**：报告记录原因并 exit 0；CPU
   部分照跑。
4. 默认运行 CPU-only，绝不要求 GPU。

## 运行命令

```bash
# CPU-only（默认；总有，无需 GPU）：
python scripts/ad_gpu_benchmark_score.py --fast

# GPU opt-in（CUDA 不可用或无可见设备则 clean-skip）：
LATTICETN_RUN_GPU=1 python scripts/ad_gpu_benchmark_score.py --fast

# 只列出所需文件：
python scripts/ad_gpu_benchmark_score.py --list
```

`--fast` preset 故意很小：N=4（χ=4）与 N=6（χ=8），短步数/扫描，CPU 上几秒
跑完。

## 预期输出

score 打印（CPU-only）：

```text
$ python -m pytest -q tests/test_ad_gpu_benchmark_config.py tests/test_ad_gpu_benchmark_smoke.py tests/test_ad_gpu_benchmark_report.py
... [ok]
$ python scripts/run_ad_gpu_benchmark.py --markdown-output docs/AD_GPU_BENCHMARK_REPORT.md --json-output docs/ad_gpu_benchmark_fast_results.json
ad gpu benchmark: pass=True gpu_ran=False report -> docs/AD_GPU_BENCHMARK_REPORT.md

AD GPU benchmark score: PASS
```

设 `LATTICETN_RUN_GPU=1` 且有 CUDA 时，`gpu_ran=True`，报告的 CPU/GPU 对比表
被填充。CPU 参考数值（`--fast` preset，seed 0）：

| N | chi | solver | final E | energy error | runtime_s | below ground |
|---:|---:|---|---:|---:|---:|:---:|
| 4 | 4 | global AD-MPS (Adam) | -1.6160223657 | 3.04e-06 | ~5.5 | False |
| 4 | 4 | 单点 AD 局部 (LBFGS) | -1.6160254037 | 4.19e-11 | ~0.3 | False |
| 4 | 4 | 两点 AD 局部 (LBFGS) | -1.6160254036 | 1.82e-10 | ~0.1 | False |
| 6 | 8 | global AD-MPS (Adam) | -2.4934815147 | 9.56e-05 | ~7.0 | False |
| 6 | 8 | 单点 AD 局部 (LBFGS) | -2.4935771330 | 8.91e-10 | ~0.9 | False |
| 6 | 8 | 两点 AD 局部 (LBFGS) | -2.4935771330 | 8.39e-10 | ~0.3 | False |

精确 E0：N=4 → -1.6160254038；N=6 → -2.4935771339。

## 关键提醒：小系统上 GPU 可能更慢

> "Runtime/speedup are记录的，但不要求 GPU 更快：小系统开销主导
> （host↔device 传输、短扫描）。"

`--fast` 的 case 极小：两点 N=4 扫描在 CPU 上约 **0.1 秒**，远低于 GPU 的
host↔device 传输 + kernel 启动延迟。所以此处 **speedup < 1×（GPU 比 CPU
慢）是预期且可接受的**——benchmark 合约是*数值一致性* + 诚实的 runtime 数，
不是性能取胜。把 speedup 列当**趋势观察**：只有在更大 N/χ（超出 `--fast`
范围）时才有意义。

硬性正确性检查（总是）：

- CPU/GPU 最终能量在 `ENERGY_AGREE_TOL = {4: 1e-6, 6: 1e-5}` 内一致（实际
  到机器 epsilon，`~1e-16`）。
- 任一设备的能量都不低于精确基态超过 `1e-6`（`below_ground=False`）。

## 常见错误

- **设了 `LATTICETN_RUN_GPU=1` 仍 `gpu_ran=False`** —— score 的 CPU-only env
  在未 opt-in 时隐藏 GPU；若已 opt-in 仍 skip，检查
  `torch.cuda.device_count() > 0`（比如 shell 里残留 `CUDA_VISIBLE_DEVICES=""`
  ——unset 它）。
- **GPU 上 `linalg.qr`/`svd` 报 `RuntimeError`** —— 当前 PyTorch + complex128
  不应出现；单点/两点扫描在后处理用 QR/SVD。若真出现，回退 CPU（CPU 运行
  始终是真相来源）。
- **混用 CPU 与 CUDA 张量** —— runner 把 CPU 张量按值拷到 GPU 以保证起点
  一致；别手搓半 CPU 半 GPU 的 MPS/MPO。
- **指望 GPU 更快** —— 见上提醒。这些小 case 的 0.4× speedup 是正常的，
  不是 bug。

## 下一步

- 更大 N 的 opt-in GPU 运行（N=8/10/12，更大 χ）以真正看到 GPU 收益，超出
  `--fast` 范围，属未来 stage。
- API：`docs/API_OVERVIEW.md` → “Benchmark / score scripts (Stage 6A)”。
