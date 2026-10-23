# P0 分布式 DSL 示例

[example_allgather_gemm.py](example_allgather_gemm.py) 用同一个 TileLang PrimFunc 表达双逻辑 rank 的 READY/ACK 通信和本地/远端 GEMM。

完整的构建、测试、报告验收及未来实机验证计划见 [P0 分布式 DSL 测试指南](../../docs/testing/distributed_p0.md)。接口与实现边界见 [设计文档](../../docs/design/distributed_p0_dsl.md) 和 [实现记录](../../docs/design/distributed_p0_implementation_validation.md)。

当前 profile `ascend950_cannsim_v1` 使用 CPU 通信模型；`protocol` 模式用 NumPy 计算，`rank-segments` 模式新编译源 TIR 中的 GEMM 并运行独立 CANN Simulator 片段。两种模式均不执行真实双卡传输，也不支持 P0-B 通信调度。

在仓库根目录、已构建 TileLang 并设置匹配扩展工具的 `CANNSIM_ROOT` 后运行：

```bash
python examples/distributed/example_allgather_gemm.py \
  --mode protocol --rounds 3 --slots 2 --seed 0 \
  --helper-root "$CANNSIM_ROOT" --output-dir output/distributed
```

运行 CANN 模式前请完成测试指南中的工具与环境检查。产物以本次 `run_dir` 为准；保留全局 oracle、通信轨迹和片段流水，不只查看局部 GEMM 的精度结果。
