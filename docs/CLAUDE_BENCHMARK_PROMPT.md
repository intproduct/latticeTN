# Claude Code Stage 2 Prompt

Copy this into Claude Code from the repository root.

```text
/goal 将当前 latticeTN 项目从“Stage 1 小系统 Heisenberg 验证通过”升级为“Stage 2 Heisenberg MPS benchmark 工具”。

请使用项目 skill：`/latticetn-benchmark`。

请把这个任务作为基于评分脚本的自动化实现循环，而不是一次性代码修改。

开始前请先阅读：

- `CLAUDE.md`
- `docs/PHYSICS_SPEC.md`
- `docs/VALIDATION_PROTOCOL.md`
- `docs/NUMERICAL_REPORT.md`
- `docs/BENCHMARK_SPEC.md`
- `docs/BENCHMARK_PROTOCOL.md`

Stage 1 回归条件：

在任何修改后，必须保证：

`python scripts/validation_score.py --fast`

仍然通过。

Stage 2 总体目标：

把当前 MPS + MPO + autograd Heisenberg 小系统验证，扩展成可复现的 Heisenberg benchmark 工具。新的 benchmark 应该支持计算和验证：

1. total energy；
2. energy per bond；
3. local magnetization `<Sz_i>`；
4. nearest-neighbor bond energy `<S_i · S_{i+1}>`；
5. two-point correlation `<Sz_i Sz_j>`；
6. bipartite entanglement entropy across a cut；
7. chi sweep；
8. small seed sweep；
9. benchmark report。

物理约定：

继续使用 Stage 1 的 convention：

`H = J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})`

其中：

- `S = sigma / 2`
- `J = 1.0`
- open boundary condition
- CPU tests
- 默认高精度 dtype，例如 `torch.complex128`

不要切换到 Pauli convention。

新增或更新文件：

- `latticetn/observables.py`
- `docs/BENCHMARK_REPORT.md`
- `docs/CLAUDE_PROGRESS_BENCHMARK.md`
- `scripts/benchmark_score.py`
- `scripts/run_heisenberg_benchmark.py`
- `tests/test_observables_dense_compare.py`
- `tests/test_entanglement_entropy.py`
- `tests/test_heisenberg_chi_sweep_smoke.py`
- `tests/test_benchmark_score.py`

执行顺序：

1. 先运行 `python scripts/validation_score.py --fast`，确认 Stage 1 基线仍然通过；
2. 检查 `docs/BENCHMARK_SPEC.md` 和 `docs/BENCHMARK_PROTOCOL.md`；
3. 实现 dense reference observable，用于小系统对比；
4. 实现 MPS observable API；
5. 新增 observable tests；
6. 新增 entanglement entropy tests；
7. 新增或修复 `scripts/run_heisenberg_benchmark.py`；
8. 新增或修复 `scripts/benchmark_score.py`；
9. 新增 chi sweep smoke test；
10. 生成 `docs/BENCHMARK_REPORT.md`；
11. 最后运行：
    - `python scripts/validation_score.py --fast`
    - `python scripts/benchmark_score.py --fast`

停止条件：

只有在以下条件全部满足时才停止：

1. `python scripts/validation_score.py --fast` 返回 exit code 0；
2. `python scripts/benchmark_score.py --fast` 返回 exit code 0；
3. `docs/BENCHMARK_REPORT.md` 包含至少一个 benchmark 表格，表格中包括：N、chi、seed、exact energy、variational final energy、energy per bond、absolute error、steps 或 runtime、pass/fail；
4. `docs/BENCHMARK_REPORT.md` 说明 Heisenberg thermodynamic-limit energy per bond `1/4 - ln(2) ≈ -0.4431471805599453` 只是大 N 趋势参考，不是小 open-chain 的严格目标。

硬性约束：

- 不要运行长时间训练；
- 不要在 pytest 中跑大系统长优化；
- 不要使用 GPU；
- 不要为了通过测试而放宽阈值；
- 不要破坏 Stage 1 已通过的接口；
- 不要在 differentiable energy path 中使用 `detach()`、`.data` 或不必要的 `.item()`；
- variational energy 不应该低于 exact ground energy，超过容差时必须暂停并报告；
- 如果 chi 增大时能量没有改善，不要伪造结果，而是在 benchmark report 中记录；
- 如果优化不稳定，优先改 benchmark 脚本的 seed、steps、lr 或 initialization，不要改物理 convention。

每次阶段性修改后，请把以下内容追加到：

`docs/CLAUDE_PROGRESS_BENCHMARK.md`

包括：当前阶段、修改了哪些文件、运行了哪些命令、命令结果、当前失败项、下一步计划。

现在请开始执行：先确认 Stage 1 validation 是否仍然通过，然后从 observable dense comparison 开始。
```
