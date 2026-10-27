# P0 分布式 DSL 示例

[example_allgather_gemm.py](example_allgather_gemm.py) 用同一个 TileLang PrimFunc 表达双 rank 的 READY/ACK 通信和本地/远端 GEMM。该源程序可进入本机联仿、原生分段及 P0-B 单 kernel 融合路径。

本机协议与 CANN 联仿见 [P0 分布式 DSL 测试指南](../../docs/testing/distributed_p0.md)；真实双卡的构建、启动、正反例及报告验收见 [Ascend 950 原生双卡测试指南](../../docs/testing/distributed_native.md)。接口与实现边界见 [设计文档](../../docs/design/distributed_p0_dsl.md) 和 [实现记录](../../docs/design/distributed_p0_implementation_validation.md)。

希望像普通算子一样调用时，使用 [example_jit.py](example_jit.py)：用 `@D.jit` 包装工厂函数，再将特化结果 `.bind(session)`，即可自动管理共享对称堆中的算子专属 workspace、信号和 epoch，并取得独立的 NumPy 输出。双 rank 启动方式、A → B → A 示例及第一版边界见 [分布式 JIT 指南](../../docs/testing/distributed_jit.md)。

联仿 profile `ascend950_cannsim_v1` 使用 CPU 通信模型；`protocol` 模式用 NumPy 计算，`rank-segments` 模式新编译源 TIR 中的 GEMM 并运行独立 CANN Simulator 片段。两种模式均不执行真实双卡传输，也不支持 P0-B 通信调度。

在仓库根目录、已构建 TileLang 并设置匹配扩展工具的 `CANNSIM_ROOT` 后运行：

```bash
python examples/distributed/example_allgather_gemm.py \
  --mode protocol --rounds 3 --slots 2 --seed 0 \
  --helper-root "$CANNSIM_ROOT" --output-dir output/distributed
```

运行 CANN 模式前请完成测试指南中的工具与环境检查。产物以本次 `run_dir` 为准；保留全局 oracle、通信轨迹和片段流水，不只查看局部 GEMM 的精度结果。

[run_native_distributed.py](run_native_distributed.py) 提供 `put`、`get` 和复用上述源 DSL 的 `gemm` 用例。原生 profile `ascend950_shmem_mte_phased_v1` 将通信降为 SHMEM AIV kernel，计算降为普通 TileLang kernel，复用 `tvm_ffi`；每 rank 独立进程绑定一张 Ascend 950，通信与计算在同一 stream 顺序执行。它不依赖 cannsim-helper。该 phased profile 保留作为基线；新增 `--execution p0a/p0b` 分别选择 pure AIV 或 mixed 单 kernel 路径，真实多卡仍待实机验收。

统一 CPU/SDK/真实双卡验收和性能入口为 [run_p0_validation.py](run_p0_validation.py)，支持按阶段和 case 重跑；完整命令、范围、报告字段见 [P0 融合验收指南](../../docs/testing/distributed_fused.md)。

先在无设备环境检查同一程序的源码：

```bash
python -m examples.distributed.run_native_distributed \
  --mode lower --case gemm --rounds 3 --slots 2 \
  --output-dir output/distributed/native_lower
```

本机已完成原生源码生成、SDK 编译和 Host ABI 检查，尚未执行真实双卡。实机应按原生指南准备固定 SHMEM/CANN 组合，依次运行 put、get、GEMM、多次 launch 和污染负例；不能把 `--mode build` 的成功当作设备精度通过。

若在设备初始化阶段失败，使用独立脚本 [diagnose_native_runtime.py](diagnose_native_runtime.py)
比较纯 `torch_npu` 单卡计算与 bridge 在设备初始化前、后加载的结果。每张卡的每种模式都使用
全新进程，串行执行；两种 bridge 模式还在加法通过后检查 Ascend 950 型号及物理设备映射，
以 `bridge_device_identity` 单独记录再次设备绑定和身份查询的结果。脚本不执行 SHMEM 初始化或双卡通信：

```bash
python examples/distributed/diagnose_native_runtime.py \
  --devices 0 1 --shmem-root "$SHMEM_ROOT" \
  --shmem-library "$SHMEM_LIBRARY" \
  --timeout 180 --output-dir output/distributed/diagnostics
```

只检查纯单卡时，传 `--modes plain`，不需要 SHMEM 或已构建的 TileLang。
完整报告与原始日志留在本地；`summary.json` 经人工审查、脱敏后可用于反馈。
参数、结果判读及边界见 [原生指南第 7 节](../../docs/testing/distributed_native.md#7-设备初始化与-bridge-加载排障)。
