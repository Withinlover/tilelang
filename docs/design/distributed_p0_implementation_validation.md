# P0 DSL 实现与本机验证记录

首次记录：2026 年 10 月 23 日；单 kernel 融合增补：2026 年 10 月 27 日。

目标：当前仓库 Ascend 950 / AscendC。第 2–10 节记录本机验证，未在本机执行真实双卡；第 11 节包含用户回传的服务器性能报告和后续 Host 排查，实机由用户执行。设计契约见 [P0 DSL 设计](distributed_p0_dsl.md)，Simulator 能力依据见[双卡仿真能力评估](cannsim_dual_rank_feasibility.md)。本机协议/CANN 流程见[联仿测试指南](../testing/distributed_p0.md)，新增原生路径的依赖、构建、启动与验收见[原生双卡测试指南](../testing/distributed_native.md)。

## 1. 交付范围

已实现 P0 源 DSL 到本机联仿的完整数据路径：一个 TileLang PrimFunc 同时描述通信和本地 GEMM，CPU 协议模型执行它的通信操作，CANN Simulator 执行从同一函数提取并重新编译的计算片段。双 rank 的远端 GEMM 输入来自 acquire 后的 Recv，最终结果以原始全局输入作独立校验。

联仿 profile 是 `ascend950_cannsim_v1`，不提供真实跨卡传输。本记录第 2–7 节保留其首次交付的证据与已观察测试结果。新增原生分段 profile `ascend950_shmem_mte_phased_v1` 从相同源 DSL 生成 SHMEM 通信 kernel 与独立 TileLang 计算 kernel；代码、原生编译与 Host ABI 已在本机验证，真实双卡执行仍待用户验收，见第 9 节。

第 2–9 节是此前联仿与分段路径的历史记录。当时尚未实现单 kernel P0-A/P0-B。
2026-10-27 新增两个显式融合 profile，详见第 10 节及[融合路径验收](../testing/distributed_fused.md)。
普通 `tilelang.compile`、`tilelang.lower` 和 Ascend pipeline 对原始 TD 函数的 gate 继续保留；
原目标名称 `ascend950_mapped_mte_v1` 未启用。历史测试数字不计入本次新增验证。

## 2. 联仿从源 DSL 到结果的证据链

```text
examples/distributed/example_allgather_gemm.py
  一个 @T.prim_func：TD.put_signal / wait / acquire / ACK + T.copy / T.gemm
    ↓ eager builder 规范化
  canonical PrimFunc
    registered tl.distributed.* op + BufferRegion effects
    稳定 ABI manifest + 内部 uint8 descriptor 参数
    ↓ 共享语义检查 + ascend950_cannsim_v1 能力和 launch 预检
  按 rank 展开同一 TIR 的结构化控制流
    ├─ TD 操作 → cannsim-helper CPU protocol Simulator
    └─ 本地 copy/GEMM 语句 → 提取普通 local PrimFunc
          → 当前仓库 Ascend lowering / codegen
          → 显式 cann-9.1-gemm64 SDK 转换
          → 本机 Bisheng / linker，dav-3510 ELF
          → cannsim-helper 新进程执行单卡计算片段
    ↓ 片段结果回到同一个协议模型和 Output 区域
  每 rank 完整输出 → 原始全局 BF16 输入的 FP64 oracle
```

这里没有复用上一阶段预编译的 GEMM `.o`。输入到 compiler 的局部语句来自源函数，四次输入搬运、一次 GEMM、一次输出搬运的操作、区域、方向和参数均保留在提取后的 TIR 中。实际产物保存为 `segment.ir.json`、`segment.tir`、`generated.asc`、`compiled.asc`、`segment.aibin`，同时保留编译命令、日志、SHA256 和 SDK 转换 diff。

本机 CANN 为 9.1。`cann-9.1-gemm64` 只转换经检查的 64×64 pure Cube GEMM SDK 接口形式；遇到未识别的生成源码、布局或操作会失败。它没有改变通信语义，也没有将通信替换成 CPU 预先计算的正确结果。片段使用三个设备参数 A/B/C；完整分布式函数的 descriptor 不作为真实 transport 描述符上传给它。

## 3. 首次联仿交付的实现落点

| 文件 | 职责 |
| --- | --- |
| [language/distributed](../../tilelang/language/distributed/__init__.py) | 编译规格、对称/信号 Buffer、不可直接解引用的 RemoteRegion、不透明 CommEvent 和 TD 操作 |
| [eager builder](../../tilelang/language/eager/builder.py) | 参数元数据传递；函数产出前物化 descriptor 与 manifest，保持同一 canonical 签名 |
| [distributed op](../../src/op/distributed.cc) | 注册 `tl.distributed.*`，声明 opaque effect，支持新进程反序列化 |
| [distributed checker](../../tilelang/analysis/distributed_checker.py) | 校验 ABI/op 元数据、方向/权限/区域/信号访问、event 生命周期和原生执行支持边界 |
| [Host session](../../tilelang/distributed/session.py)、[ABI](../../tilelang/distributed/abi.py)、[launch](../../tilelang/distributed/launch.py) | 单 CPU 协调器的分配、预检、epoch、状态、绑定和失败报告 |
| [capabilities](../../tilelang/ascend/distributed/capabilities.py) | CPU adapter 的 profile 检查；原生分段路径使用自己的显式 capability/运行时预检 |
| [cosim adapter](../../tilelang/ascend/distributed/cosim.py) | 执行各 rank 的模型操作及计算片段回调；计划生成现与原生路径共享 [plan.py](../../tilelang/ascend/distributed/plan.py) |
| [segment extraction/compiler](../../tilelang/ascend/distributed/segments.py) | 验证并提取真实局部 GEMM，复用当前 Ascend lowering/codegen，明确进行受限 SDK 转换和本机编译 |
| [AllGather + GEMM 示例](../../examples/distributed/example_allgather_gemm.py) | 源 DSL、Host 绑定、独立全局 oracle、报告和 CLI |

联仿适配器按 `(rank, object, element_offset, element_count)` 识别固定连续区域。完全相同的区域共享模型状态；部分相交区域、跨对象或跨 rank 的 NumPy storage alias 会拒绝，避免分别复制后掩盖真实别名。READY/ACK 由独立 SignalRef 表达，每个 int32 信号单元占用独立 64 字节位置。

未支持的 T.Stage、PerCoreTask、其他属性作用域、非平凡 SBlock predicate/iteration/match alias、一般 local copy 转换或未知计算语义不会被静默擦除。整个 rank 计划的 peer 与发布 epoch 在协议事件执行前预检。`fence` 保守实现为 `quiet`；正常出口再 drain 一次，但不替代协议中的尾部 ACK。

## 4. CPU profile 的 Host API 与限制

`D.init(process_group=None, device=None, profile="ascend950_cannsim_v1", world_size=2)` 创建一个 CPU 协调器，持有两个 rank 的独立数组。该 CPU profile 不接受真实 process group、设备绑定或 device stream；新增原生 profile 有独立的进程/device/stream 管理，见第 9 节。BF16 数组按原始 `uint16` bits 保存，不将 NumPy 的整数乘法误当 BF16 计算。

下面的代码展示完整调用顺序；算子和输入生成函数均来自可运行示例：

```python
import os
import numpy as np
import tilelang.distributed as D
from examples.distributed.example_allgather_gemm import allgather_gemm, make_inputs

program = allgather_gemm(rounds=3, slots=2)
kernel = D.compile(
    program,
    profile="ascend950_cannsim_v1",
    mode="rank-segments",  # protocol 只运行 CPU 模型与 NumPy 计算
    helper_root=os.environ["CANNSIM_ROOT"],
    output_dir="output/distributed",
)
ctx = D.init(world_size=2)
objects = {}
for spec in kernel.manifest["objects"]:
    allocation = dict(key=spec["key"], shape=spec["shape"], dtype=spec["dtype"])
    objects[spec["key"]] = (
        D.alloc_signals(ctx, **allocation)
        if spec["kind"] == "signal"
        else D.alloc_symmetric(ctx, remote_access=spec["remote_access"], **allocation)
    )
ch = D.channel(ctx, key="comm")
ticket = D.prepare_launch(
    ctx,
    kernel,
    channel=ch,
    epochs=3,
    protocol="allgather-gemm-v1",
    parameters={"rounds": 3, "slots": 2, "tile": 64},
)
a, b = make_inputs(rounds=3, seed=0)
D.copy_in(objects["Send"], {rank: a[rank] for rank in range(2)}, ticket=ticket)
D.copy_in(objects["B"], {rank: b for rank in range(2)}, ticket=ticket)
D.copy_in(
    objects["Output"],
    {rank: np.full((3, 2, 64, 64), np.nan, dtype=np.float32) for rank in range(2)},
    ticket=ticket,
)
run = D.launch(kernel, ticket=ticket, seed=0)
adapter_report = D.complete(run)
outputs = np.stack([objects["Output"].arrays[rank] for rank in range(2)])
D.close(ctx)
```

这段 API 示例返回 adapter 结果；全局 oracle 与最终验收合取由示例 CLI 的 `run()` 完成。`D.launch` 是同步调用但仍返回 Run；显式 `D.complete` 后才能申请下一 ticket。下一次 `prepare_launch` 使用新 epoch 区间，不清零已发布信号；测试已验证 `[3,2] → [6,5]` 的双 slot 信号值。失败 Session 保留 FAILED 状态，不再接受 launch/reuse；`D.RunFailedError.run.result` 保留 adapter 的失败证据。

descriptor 使用固定 header 和对象/rank 记录，包含 generation、launch、rank、epoch 区间和物理大小，所有地址字段为 0。它只承担 CPU profile 的 ABI 一致性检查。可复用身份包括 canonical IR 摘要和编译 profile，排除具体 epoch 和运行时地址；源 TD 全函数仍不进入普通 JIT；新增原生分段路径改用 SHMEM 初始化状态和普通子 kernel 参数，其 header/ABI 指纹与缓存规则见第 9 节。

## 5. 可复现命令

在当前仓库根目录执行。先按[测试指南](../testing/distributed_p0.md)准备扩展工具，再设置 `CANNSIM_ROOT`；本机使用仓库 `.venv` 运行 TileLang，worker 使用 helper 自己的 `.venv` 运行 CANN Host。

```bash
export CANNSIM_ROOT=/path/to/extended/cannsim-helper
export CANN_ROOT=/path/to/cann-9.1
source "$CANN_ROOT/set_env.sh"

# 纯协议 / CPU 计算检查
.venv/bin/python examples/distributed/example_allgather_gemm.py \
  --mode protocol --rounds 3 --slots 2 --seed 0 \
  --output-dir output/distributed

# 正例：同一 DSL 提取并新编译计算片段，执行 12 次真实 CANN GEMM
.venv/bin/python examples/distributed/example_allgather_gemm.py \
  --mode rank-segments --rounds 3 --slots 2 --seed 0 \
  --output-dir output/distributed

# 负例：4 次局部 GEMM 正常完成，全局校验失败；预期退出码为 1
.venv/bin/python examples/distributed/example_allgather_gemm.py \
  --mode rank-segments --rounds 1 --slots 2 --seed 0 \
  --fault corrupt_payload --output-dir output/distributed
```

每次执行都会创建独立子目录。`--consumer-delay` 的单位是逻辑 tick，`--segment-timeout` 的单位是 Host 墙钟秒；两者含义不同。实际 CANN 子进程超时或 worker 失败会传播为失败，清理所属进程组，并保留日志与已完成片段的产物。

## 6. 首次联仿交付的实际验证结果

| 项目 | 正例 | 通信污染负例 |
| --- | --- | --- |
| rank / rounds / slots / seed | 2 / 3 / 2 / 0 | 2 / 1 / 2 / 0 |
| 真实 CANN 计算片段 | 12/12 通过 | 4/4 通过 |
| 片段流水文件 | 24，含/不含 SET/WAIT 各一份 | 8，含/不含 SET/WAIT 各一份 |
| 完整全局输出 | 49,152 个元素 | 16,384 个元素 |
| 全局误差元素 | 0 | 64 |
| 最大绝对误差 | `2.2887252271175385e-7` | `1.080986112356186` |
| `rtol` / `atol` | `1e-5` / `1e-5` | `1e-5` / `1e-5` |
| 协议尾部状态 | 无 pending event、无 open lease，两 rank 均结束 | 同左；污染不破坏协议本身 |
| 顶层结果 | PASS | 按预期 FAIL |

负例在 rank 0、第 0 轮、owner 1 的 `Recv` tile 改动一个 BF16 元素 `[0,0]`。该片段实际输入包含改动，所以局部 GEMM 相对它所收到的输入仍正确；原始全局输入未改变，独立 oracle 因此检出整行 64 个错误。这证明结果确实经过 DSL 通信接收数据，而非直接从全局原始输入替换远端分片。

联仿交付代码重跑正例的 canonical IR SHA256 为 `7cd0ee2d586e45acc1e5efb99da8d7faf915ddb92d043bc5fb6a8f32dba39cbd`。提取的两种局部片段均编译出 SHA256 为 `92d04e122accfdd7a9419f178d823041b89d4c886b846ecb83c87a602859c763` 的同一计算 ELF；不同的输入来源和输出位置由各片段的 `dsl_context` 记录。本机记录没有可用源码 span，因此用序列化 IR 的操作/区域/参数对应关系证明提取来源，不声称已验证源码行映射。

以下为首次验收时生成的本地产物标识，未提交到 Git；在其他机器执行时会产生新的目录名。完整复现流程和外部 helper 补丁见[测试指南](../testing/distributed_p0.md)。

- 正例：总报告（`output/distributed/validation_20261023/p0_rank_segments_hz9qbbkt/report.md`）、manifest（`output/distributed/validation_20261023/p0_rank_segments_hz9qbbkt/manifest.json`）、流水索引（`output/distributed/validation_20261023/p0_rank_segments_hz9qbbkt/trace_index.json`）。
- 负例：总报告（`output/distributed/validation_20261023/p0_rank_segments_jzm0g_y4/report.md`）、manifest（`output/distributed/validation_20261023/p0_rank_segments_jzm0g_y4/manifest.json`）、流水索引（`output/distributed/validation_20261023/p0_rank_segments_jzm0g_y4/trace_index.json`）。
- 独立审计：正例审计（`output/distributed/validation_20261023/p0_rank_segments_hz9qbbkt/independent_audit.json`）、负例审计（`output/distributed/validation_20261023/p0_rank_segments_jzm0g_y4/independent_audit.json`）、审计脚本（`output/distributed/validation_20261023/independent_artifact_audit.py`）。

独立审计重新读取原始输入、片段实际输入/输出、ELF、SDK diff 和 trace，核对文件 hash、参数 ABI、每个输出 tile 的 rank/owner/round 位置，以及 AIC Core 0 的 MMAD/END/detail。全局数值使用 64 次 FP64 外积累加重新计算，独立于生产校验的矩阵乘法实现。两个审计均通过；负例审计通过表示成功验证“应失败且恰有预期通信污染”，不是把负例的数值结果判成正确。

## 7. 首次联仿交付的本机回归记录

以下为原生扩展之前的历史验收结果，不作为本次最终合并总数；当前原生扩展验证见第 9 节。该阶段新增功能的 focused 回归共 **131 passed**：

| 测试文件 | 已观察结果 | 关键覆盖 |
| --- | --- | --- |
| [源 DSL](../../testing/python/language/test_tilelang_language_distributed.py) | 25 passed | TD 参数、op、权限、事件和规范化签名 |
| [共享语义检查](../../testing/python/analysis/test_distributed_checker.py) | 28 passed | manifest/op schema、非法程序、event 和区域约束 |
| [Host Session 测试](../../testing/python/runtime/test_tilelang_distributed_session.py) | 20 passed | 实际 frontend manifest 对接、信号 stride、epoch 耗尽、跨 ticket/Session 误用、并发拒绝、descriptor 篡改和失败保留 |
| [计算片段](../../testing/ascend/distributed/test_segments.py) | 23 passed | 从源 TIR 提取，shape/dtype/layout/storage/参数/annotation 不支持时拒绝 |
| [DSL 联仿边界](../../testing/ascend/distributed/test_cosim.py) | 26 passed | put/get、两次 launch、effect scope、epoch/peer 预检、非平凡 block、local copy、整数溢出及 storage alias 拒绝 |
| [worker 生命周期](../../testing/ascend/distributed/test_worker.py) | 1 passed | 无响应 worker 及独立 native 子进程组清理 |
| [应用协议测试](../../examples/distributed/test_allgather_gemm.py) | 8 passed | 多种 round/slot/seed、延迟消费者、early READY/丢 READY/丢 ACK/污染负例和不清零信号的第二次 launch |

另一次原生执行 gate、eager AST signature 与 dialect 回归为 **52 passed**，以上两组合计 **183 passed**。cannsim-helper 的 unittest discover 为 **90 passed**。额外 JIT diagnostics/pass timing 回归为 **27 passed、3 skipped**。新增代码 Ruff 格式和规则检查、两仓库的 `git diff --check` 均通过。

没有声称仓库全套测试通过：尝试扩大到旧 analysis 测试时有 9 项失败，原因是本机没有可供 `target="auto"` 选择的设备；旧 cache 测试有 4 项失败，原因是显式 CUDA target 没有固定 arch，且本机没有可用于推断 arch 的 GPU。上述环境局限与新增本机 profile 的 focused 结果分别记录。

```bash
CANNSIM_ROOT="$CANNSIM_ROOT" \
  .venv/bin/python -m pytest \
  testing/python/language/test_tilelang_language_distributed.py \
  testing/python/analysis/test_distributed_checker.py \
  testing/python/runtime/test_tilelang_distributed_session.py \
  testing/ascend/distributed examples/distributed/test_allgather_gemm.py -q

.venv/bin/python -m pytest \
  testing/python/analysis/test_distributed_native_gate.py \
  testing/python/language/test_tilelang_language_eager_ast_signature.py \
  testing/python/language/test_tilelang_language_dialect.py -q
```

## 8. 联仿边界与剩余原生目标

CPU profile 只支持两个逻辑 rank、一个静态组/通道、单 active launch、int32 信号、连续静态区域和受限结构化控制流。通信 Buffer 的部分相交切片、跨对象 alias、任意 layout、额外计算效果、Stage/Pipelined/mixed 通信调度均不在该 profile 的支持集内；零轮空工作尚未提供专门 Host 快速路径。

协议请求和完成由 CPU 模型调度；CANN 片段逐个独立运行在 device 0。单卡流水有自己的时间原点，协议 tick 无物理时间单位；当前报告没有把它们拼成统一的双卡物理流水，也没有给出带宽、时延或重叠收益。

新增分段路径已实现 SHMEM 设备通信、独占通信 scratch/event、跨进程一致性、设备 stream 和资源生命周期，但这些实现尚未经过真实双卡验收。仍未实施的目标包括单 kernel P0-A lowering、通信 effect 对 AutoSchedule/InsertSync/MultiBuffer 的完整接入、P0-B 融合，以及其他 HCOMM/UDMA transport。原生双卡压力和性能结论也仍需实机证据，不能由 CPU 模型或单卡 Simulator 结果替代。

## 9. 原生分段路径与本机验证增补

`ascend950_shmem_mte_phased_v1` 使用每 rank 独立进程和 stream，从同一源程序生成独立的 SHMEM AIV 通信 kernel 与 TileLang Cube GEMM kernel，目标固定为 `dav-3510`。通信数据由设备 MTE 搬运，CPU 只承担初始化、提交、元信息检查和独立输出校验。该路径不依赖 cannsim-helper，也不将 CPU 模型中的结果复制为远端设备结果。

| 文件 | 新增职责 |
| --- | --- |
| [共享计划](../../tilelang/ascend/distributed/plan.py) | 联仿/原生复用同一源 TIR 解释；保留通信 regions 和原始计算语句 |
| [原生编译与编排](../../tilelang/ascend/distributed/native.py) | 双 rank/epoch/区域预检、有限计划确定等待死锁检查、子 kernel 编译/缓存、launch/complete 和产物 |
| [SHMEM transport](../../tilelang/ascend/distributed/transport.py) | pure AIV put/get、原子信号、设备 wait；每 kernel 使用独占 scratch，并在自身退出前完成操作 |
| [原生 Host runtime](../../tilelang/ascend/distributed/native_runtime.py)、[SDK bridge](../../src/ascend/distributed/shmem_host_bridge.cc) | 固定版本 SHMEM 初始化、对称分配和 peer 映射、DLPack 借用视图、独立 stream、集体预检、epoch 与关闭 |
| [原生验证入口](../../examples/distributed/run_native_distributed.py) | put/get/GEMM、连续 launch、slot 复用、原始输入 oracle、源污染负例及双进程监督 |

原生执行 manifest 使用 `host-phases-shmem-v1`，记录源程序身份、SDK 编译 profile 和 SHMEM 头文件指纹。前端遗留的 `cpu-model-v1` descriptor 不上传设备；子 kernel 仅接收自身的 Buffer/标量，remote 地址由 SHMEM 的合法映射解析。头文件、库与 bridge 身份参与缓存或跨 rank 一致性检查，实际地址和 epoch 不作为可复用代码的身份。

本机实际完成以下验证：

- 最终 focused 回归 **268 passed**，包含全部 Ascend distributed 测试、Python distributed 前端/检查器/gate/Session、两个示例测试文件以及 eager AST signature/dialect 回归。测试同时配置 SHMEM、Bisheng、可选 Host bridge 和 cannsim-helper，没有 skip；这不是全仓测试通过声明。
- `run_native_distributed.py --mode build` 的 put、get、GEMM 三种用例均通过，完成真实通信/计算子 kernel 的 SDK 编译和导出，以及 Host bridge 构建。固定版本的完整 Ascend 950 SHMEM 库也已在本机编译。
- 原生 put/get 用例的同一源 IR 通过 CPU 协议模型交叉验证：多轮、有限 slot 复用、连续两次 launch、原始 peer 输入逐 bit 对照，以及全部 lease/event 收尾。它验证源协议，不证明真实互联。
- Host ABI 测试使用真实 SDK 头文件编译桥接代码，并检查 DLPack 设备类型、shape/stride、借用生命周期。仅这类本机无设备加载测试使用 driver stub，未初始化设备；实机必须使用真实驱动/runtime，不能使用 `devlib` 或 Simulator stub。
- 本机无 NPU 的真实启动入口尝试在依赖检查阶段因缺少 `torch_npu` 受监督地失败，rank 和父进程均退出 2，顶层结果为 false，未初始化设备，也未被污染负例验收误判为通过。这验证失败传播，不代表完成了设备执行。

典型回归命令如下；`SHMEM_ROOT`、`BISHENG_HOME`、`CANNSIM_ROOT` 和可选 bridge 依赖须按[原生测试指南](../testing/distributed_native.md)与本机 ABI 测试说明配置：

```bash
TILELANG_TEST_SHMEM_BRIDGE=1 python -m pytest -q \
  testing/ascend/distributed \
  testing/python/language/test_tilelang_language_distributed.py \
  testing/python/analysis/test_distributed_checker.py \
  testing/python/analysis/test_distributed_native_gate.py \
  testing/python/runtime/test_tilelang_distributed_session.py \
  examples/distributed/test_allgather_gemm.py \
  examples/distributed/test_native_distributed.py \
  testing/python/language/test_tilelang_language_eager_ast_signature.py \
  testing/python/language/test_tilelang_language_dialect.py
```

构建输出保留 `lowering.json`、`build.json`、`validation.json`、原始/兼容转换源码、TIR、包含设备代码的共享库及哈希；`device_executed=false` 是本机 build 的预期结果。所有原始产物留在本地，不随 Git 提交。

真实双卡的精度、READY/ACK 可见性、热点 slot 和实际进度仍待用户在保密机器上按[原生双卡测试指南](../testing/distributed_native.md)执行。当前分段实现没有通信计算重叠，也没有单 kernel P0-A/P0-B 融合性能结论。

## 10. 单 kernel P0-A/P0-B 实现与本机验证（2026-10-27）

新增 `ascend950_shmem_mte_p0a_v1` 和 `ascend950_shmem_mte_p0b_v1`，复用已有 Session、
对称分配、peer 映射、epoch、stream、D.jit 和关闭生命周期。默认 phased profile 保留。
融合路径把每 rank 的有限静态计划降成一个真实设备 PrimFunc，参数为对称 Buffer 与
本次 launch 的 int32 epoch；同一 launch 每 rank 只提交一个物理 kernel。

- P0-A：pure AIV，支持 put/get、受限 FP32 GM↔UB 搬运及 `vld/vadd/vsts`，
  保留顺序、显式同步、可见性/完成动作和退出 drain。向量掩码只接受 FP32 全 64 lane
  的 `pset(32, PAT_ALL)`，拒绝隐式 GM 访问、非法 layout、越界及不支持的向量语义。
- P0-B：单 mixed block，唯一通信发起者 AIV0；同 kernel 内执行手写协议 AllGather
  和固定 BF16 64×64 GEMM、FP32 输出。acquire 后才可搬入接收数据；ACK 等待接收区
  GM→L1 DMA 完成，但不等待后续 MMAD/Fixpipe，允许后续通信与 Cube 计算重叠。
  这是有界静态展开的受限流水，不支持一般动态 Stage/Pipelined 通信。
- 编译器：通信 task 具有显式 acquire/release region effects，实际经过 AutoSchedule、
  InsertSync、MultiBuffer 及后段变换；没有删除原始 TD native gate 或关闭 AutoSchedule。
  有限协议预检拒绝未排序访问、提前 slot 复用、未完成 get 源 lease、确定性等待死锁和
  epoch 相关的数据地址/计算变化。调度后和终检核对通信顺序、ABI、effects 与 scratch。
- 资源：共用不参与版本化的 4096 字节 UB scratch，通信独占 hard-event 7，普通 flag
  分配使用 0–6 与已有 mutex spill；融合路径保守禁用本地物理存储复用。根作用域的
  单次直线任务仅在完成图证明旧 WAIT 先于新 SET 时复用 flag，循环保留跨迭代约束。
  helper/SDK adapter 身份进入缓存，终检拒绝本地指针侵入实际 scratch 范围。

双 rank、单组/通道、单活动 launch、32 字节通信对齐及独立信号单元限制继续适用。
零工作计划明确拒绝，调用方在 Host 跳过；单 rank 展开最多 4096 条指令，仍须满足
本地存储、跨核 flag 和 mutex 容量。超限保持 unsupported 诊断，不以删除同步绕过。
Z3 在部分多轮例上耗尽优化预算并返回 `unknown (canceled)`；AutoSchedule 使用合法
拓扑回退，保留两核依赖和重叠机会，不宣称完成最优调度。详见[调度与资源边界](../testing/distributed_fused.md#调度与资源验证的准确边界)。

本机环境为 Linux x86_64、Python 3.10、Ascend native/stub build、CANN/Bisheng 9.1，
目标 `dav-3510`；SHMEM revision 为 `a232a76516c435cd39ecd10125fc6e14a55fc18b`。
本机没有可用 NPU/`torch_npu`。所有测试使用独立 worktree 的新 native build，未混用
原 checkout 的 Python 源码或库。原工作区已有未提交修改保持原样，未包含在本次改动中。

| 本机检查 | 实际结果 | 证据范围 |
| --- | --- | --- |
| Ascend CMake 完整构建及修改后的增量构建 | 成功 | 编译器/Host 库，不执行设备 |
| 统一 CPU 回归 | **325 passed，0 skipped，6 deselected** | 排除真实 SDK 与 opt-in bridge；前端、协议、IR、ABI、JIT、运行时及监督脚本 |
| AutoSchedule 完整目录及 MergeUBAllocations 回归 | **644 passed，31 skipped** | 31 项均缺 Ascend 运行支持；另有 9 个既有 pytest marker warning |
| 真实 SDK Host bridge DLPack 用例 | **1 passed** | metadata/借用生命周期，未初始化设备 |
| 全新 cache 的统一 SDK 编译矩阵 | **13/13 passed** | phased/P0-A/P0-B、双 rank，含 7 轮单 slot；所有 `device_executed=false` |
| P0-A 实际提取的向量计算片段 CANN Simulator | **4096 元素，0 mismatch，最大误差 0** | AIV0 正常 END，report complete；465 条指令事件，453 条带 detail |
| 同源 AllGather＋GEMM 联仿，3 轮/2 slots | **12 个片段通过，49152 元素，0 mismatch** | 原始 BF16 输入的 FP64 oracle；最大绝对误差约 2.832e-7，无未结 lease/event |
| 联仿污染负例，1 轮/2 slots | **按预期检出 64 个错误元素，退出 1** | 4 个本地计算片段均通过，独立全局 oracle 检出被污染输入；不是设备负例 |
| 无设备启动 P0-A 正例与 P0-B 预期负例 | **均被判为失败/未执行设备** | 缺 `torch_npu` 的 rank 错误传播，负例未将基础设施失败误计通过 |

首次 SDK 矩阵曾在七轮 P0-B 压力例耗尽 mutex spill（66 > 32）；修复直线任务完成图
证明下的 flag 复用后，重新构建 native 库并使用全新 cache 跑完整 13 项，全部通过。
最终 CPU/调度回归包含安全复用正例与仅有 DMA 发射顺序、没有完成依赖的拒绝负例。
SDK 编译阶段的“负例”只表示该测试程序可编译，不代表故障已在设备被检出。

本次本地产物位于 `output/distributed/fusion_local_20261027/`，不随 Git 提交：

- `final-cpu/run_r4tgo9tj/`、`final-compile/run_x0332o3l/`：统一入口 JSON、日志、pytest XML 或 SDK 产物。
- `full-schedule-regression.log`、`full-schedule-regression.xml`、`real-bridge.log`：扩展回归与 ABI 证据。
- `vector-cosim/`：实际片段提取说明、源码/设备对象哈希、独立审计及 Simulator trace。
- `cosim/p0_rank_segments_zh979273/`、`cosim-negative/p0_rank_segments_2u3ye46r/`：GEMM 联仿正负例。
- `no-device/run_sihdzm1x/`：缺运行依赖的失败传播证据。

可复现的主回归命令如下；环境准备、SDK/SHMEM 构建和服务器启动完整命令见
[融合路径验收指南](../testing/distributed_fused.md)：

```bash
python -m examples.distributed.run_p0_validation --stages cpu --output-dir output/distributed/p0-acceptance
TILELANG_CACHE_DIR="$PWD/output/distributed/p0-fresh-cache" \
  python -m examples.distributed.run_p0_validation --stages compile --keep-going \
  --sdk-profile cann-9.1 --output-dir output/distributed/p0-acceptance
python -m pytest -q -rs testing/ascend/auto_schedule \
  testing/ascend/transform/test_ascend_merge_allocations.py
TILELANG_TEST_SHMEM_BRIDGE=1 python -m pytest -q \
  testing/ascend/distributed/test_native_runtime.py -k real_bridge_dlpack
```

统一服务器入口提供超时监督、子进程退出码、各 rank 日志、独立数值与信号判定、
严格污染负例、机器可读报告和按 case 重跑；性能入口比较 Host HCCL＋本地 TileLang、
P0-A 交换＋本地 Cube 与 P0-B 融合，分别记录首次构建/执行、稳态和包含暂存/同步的
端到端耗时。本机未执行真实双卡和性能测量。**多卡正确性、可见性、进度及性能继续
保持“待实机验收”**，直到收到用户检查、脱敏的真实服务器结果。

## 11. 首次服务器性能反馈与 Host 热路径排查（2026-10-27）

用户回传了 performance 阶段的汇总：双 rank、BF16 64×64、3 轮、2 slots，
3 次 warmup 后测量 20 次。HCCL、P0-A、P0-B 均报告 `passed=true`、
`device_executed=true`、进程退出 0 且未超时。这是该性能 workload 的真实设备
数值通过证据；报告未包含 commit/provenance，也未包含默认 13 个 device case 的
完整验收结果，不能据此宣布全部多卡验收通过。

| 模式 | execution 中位数 | execution p95 | 端到端中位数 |
| --- | --- | --- | --- |
| HCCL＋本地 TileLang | 0.264 ms | 6.181 ms | 0.373 ms |
| P0-A 交换＋本地 Cube | 130.612 ms | 270.147 ms | 170.627 ms |
| P0-B 融合 | 606.210 ms | 751.887 ms | 978.151 ms |

这些是包含 Host 提交、预检与同步的 wall time，不是设备 kernel 的独立耗时。
HCCL 一次批量 AllGather 三个 tile，P0 按三轮执行 READY/ACK slot 协议；输出计算
任务相同，通信调用组织不同。HCCL 样本有明显长尾，表中 p95 按原始 20 个样本线性
插值计算。原始较慢结果保留，不以移出预检/同步或更改计时边界消除差距。

排查确认旧实现每次 prepare 和 launch 都重新调用融合 planner，每次又分别展开当前
epoch、epoch 0、epoch 1 的两个 rank；GEMM 片段提取还被描述生成和缓存键计算重复
触发。即使已经命中设备 kernel cache，这些 Host 工作仍被执行。SHMEM header
指纹还先遍历整个源码目录（包含 build/install），再过滤实际 include/src 文件。

本机保留真实 planner、头文件指纹、序列化和 ABI 检查，用测试 Bridge/Control 及
空 kernel 隔离设备与网络后，20 次测量中位数如下：P0-A prepare 29.625 ms、execution
56.971 ms、端到端 86.659 ms；P0-B 分别为 195.020、300.765、497.673 ms。
五次 P0-B cProfile 共记录 1020 次 `extract_segment`（每 launch 204 次），占该
profiling 时间约 2.287/3.745 秒；kernel 编译只在预构建时发生两次。这定位了可复现的
Host 瓶颈，不能用本机数字代替服务器各阶段的实际耗时。

后续修复缓存已验证的 scalar specialization、计划描述、GEMM 片段身份、两 rank
PrimFunc 和编译键，分别限制缓存容量。返回给调用方的计划复制所有可变 Python 容器，
build 检查成功计划的 token；失败的下一次计划不会更改前一次成功计划绑定的专化。
跨 epoch 复用必须额外经过 IR 结构证明：epoch 只能平移信号值，且当前 base 落在
每个 int32 加减中间值的安全闭区间内。epoch 相关的控制、地址、计算、未知 IR 或
区间外情况继续逐 epoch 展开核查，不用 epoch 0/1 的抽样结果推断任意 epoch。

每次 launch 仍验证 epoch 区间、regions、动态初始信号进度、Session 绑定和跨 rank
一致性；源码、manifest、helper 和 SHMEM header 内容变化的诊断保留。header 指纹
仅遍历参与原指纹的 include/src 子树，保持原路径排序和完整内容 hash。没有删除
READY/ACK、DMA 完成、资源检查或同步，也没有引入“关闭校验”的性能模式。

本轮本机回归为 **383 passed、0 skipped、6 deselected**；6 个代表用例的全新缓存
SDK 编译全部通过（P0-A put/get/vector/hot-slot，P0-B GEMM/hot-slot），其两 rank
共 **12 份生成的设备源码与上一版逐字一致**。监督/性能脚本单独回归为 30 passed。
本轮未重跑 CANN 计算片段，也未执行真实设备；设备代码未变，新的 Host 快路径仍需
服务器进行正确性回归和性能重测。

性能报告 schema 2 保留旧 execution/e2e 边界，新增 prepare、staging、launch、complete、
readback 等分段及 launch 内部规划、预检、缓存、提交、同步的嵌套归因，同时保留各
rank 原始样本、p90/p95 和逐次独立数值误差。它不把 Host 提交/同步当成纯设备时间。
运行命令和字段解释见[性能口径](../testing/distributed_fused.md#性能口径)。

本轮产物保存在 `output/distributed/performance_followup_20261027/`，不随 Git 提交：
`cpu/run_i9umbta1/`、`sdk/run_2u7i8x5m/`、`device-source-comparison.json` 与
`harness-final.log`。初次 profiling 证据在 `output/distributed/performance_host_20261027/`。

为校准前后比较，另用同一个脚本、同样的输入暂存/输出回读集合，在独立进程顺序加载
`4c557232` 的原始 Host 模块和本轮模块，各测 3 次 warmup＋20 次稳态；期间没有并行
运行编译或回归。两边都使用测试 Bridge/Control 和空 kernel，且仅预构建两次。

| 本机隔离测量中位数 | 旧版 prepare → 修复后 | 旧版 execution → 修复后 | 旧版端到端 → 修复后 |
| --- | --- | --- | --- |
| P0-A | 30.292 → 1.212 ms | 57.137 → 6.609 ms | 87.520 → 7.961 ms |
| P0-B | 195.752 → 1.698 ms | 299.164 → 7.644 ms | 494.989 → 9.591 ms |

原始样本和脚本位于本轮目录的 `paired-host/`。这组“端到端”只指隔离实验中的
prepare、输入暂存、launch/complete、Output 回读；没有真实设备、TCP 通信，亦未包含
P0-A 的额外本地 GEMM。不能将约 39 倍的 P0-B Host execution 降幅当作服务器加速比。
完整 header 内容校验和源码序列化等仍有毫秒级成本，尚未建立低延迟性能达标结论。
下一轮必须保持原服务器口径重测，并根据新增分段定位剩余成本。

## 12. 精简回传报告与保留校验的目录遍历优化（2026-10-27）

用户反馈新性能报告超过一万行，且加速比仍只有 `0.0x`。目前尚未收到该轮分段
数据，不能把剩余耗时归因于设备通信或 GEMM，也没有形成性能改善的实机结论。

新增标准库离线入口 `examples/distributed/summarize_p0.py`，可直接读取已有结果目录
或旧 schema 1/2 的完整 JSON，不导入 TileLang、不初始化设备、不重跑测试。输出
`report.md` 与 `compact.json`，只保留固定枚举、有限数值、版本和完整 commit 哈希；
命令、绝对路径、原始错误文本和逐次样本留在本地。缺失分段显示 `null`，既有失败、
退出码、超时及未执行设备状态不会因导出改变。性能比率同时列出 HCCL/当前模式和
当前模式/HCCL，避免低于 1 的比率含义不清。

统一入口及性能 worker 后续将完整证据写入 `detail.json`，默认 `summary.json`
改为 schema 3 分享摘要，并直接生成 `report.md`；验收判断仍读取完整证据，兼容
旧 worker 的完整 summary，但不接受分享摘要作为验收证据。离线导出使用原运行的
provenance，不将当前工具版本的 commit 冒充旧测量版本。对用户此前回传的旧报告，
实际导出 Markdown **59 行、2876 字节**，JSON **7392 字节**。大量样本的合成测试
确认输出尺寸不随样本数组增长。运行及反馈步骤见
[融合路径验收指南](../testing/distributed_fused.md#已有大报告的离线精简)。

同时将 SHMEM include/src 遍历改为 `os.scandir`，每次仍读取并 hash 所有头文件内容，
保持按路径分量排序、文件/目录符号链接行为及同 mtime 内容变化检测。实际 SDK 的
compiler flags 和 header digest 与 `69b9da8b` 一致。本机 205 个头文件、约 1.94 MB
的隔离测量中，完整指纹检查中位数从 3.777 ms 降为 1.457 ms；测试 Bridge/Control
及空 kernel 的 P0-A/P0-B execution 从 6.947/8.203 ms 降为 5.098/6.286 ms。
这些是 Host 隔离数据，没有 TCP、真实设备或 P0-A 的额外本地 GEMM，不能替代
服务器性能结果。本轮未移除源码、ABI、header 或跨 rank 检查，也未改变计时边界。

本轮实际验证：统一 CPU 回归 **387 passed、0 skipped、6 deselected**；全新缓存的
`p0b-gemm` 真实 SDK 编译通过，覆盖两个 rank，同时验证新报告写入路径。未重跑
完整 SDK 矩阵、CANN 计算片段或真实设备测试；生成设备程序及其同步协议没有改动。
CPU、SDK 与精简报告样本位于 `output/distributed/compact_report_20261027/`，Host
隔离测量位于 `output/distributed/performance_host_69b9da8_review/`，均不随 Git 提交。
真实多卡完整验收和性能结论仍等待相应服务器结果。
