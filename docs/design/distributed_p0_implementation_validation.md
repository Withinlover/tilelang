# P0 DSL 本机实现与端到端验证

日期：2026 年 10 月 23 日。

目标：当前仓库 Ascend 950 / AscendC；所有执行和验证均在本机完成。设计契约见 [P0 DSL 设计](distributed_p0_dsl.md)，Simulator 能力依据见[双卡仿真能力评估](cannsim_dual_rank_feasibility.md)。

## 1. 交付范围

已实现 P0 源 DSL 到本机联仿的完整数据路径：一个 TileLang PrimFunc 同时描述通信和本地 GEMM，CPU 协议模型执行它的通信操作，CANN Simulator 执行从同一函数提取并重新编译的计算片段。双 rank 的远端 GEMM 输入来自 acquire 后的 Recv，最终结果以原始全局输入作独立校验。

当前 profile 是 `ascend950_cannsim_v1`。它不提供真实跨卡传输，未认证硬件缓存一致性、互联完成域或跨卡性能。`ascend950_mapped_mte_v1` 及其他未认证 transport 会明确拒绝；含 TD 的函数不能通过普通 `tilelang.compile`、JIT/cache、`tilelang.lower` 或 Ascend pipeline 偷渡到原生设备代码。

原设计中的原生 P0-A 通信合法化和 P0-B 通信调度仍未实现。当前包含 GEMM 的联仿程序也不是一个含通信的 mixed-device kernel；它分解成 CPU 协议操作和独立单卡 Cube kernel。

## 2. 从源 DSL 到结果的证据链

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

## 3. 实现落点

| 文件 | 职责 |
| --- | --- |
| [language/distributed](../../tilelang/language/distributed/__init__.py) | 编译规格、对称/信号 Buffer、不可直接解引用的 RemoteRegion、不透明 CommEvent 和 TD 操作 |
| [eager builder](../../tilelang/language/eager/builder.py) | 参数元数据传递；函数产出前物化 descriptor 与 manifest，保持同一 canonical 签名 |
| [distributed op](../../src/op/distributed.cc) | 注册 `tl.distributed.*`，声明 opaque effect，支持新进程反序列化 |
| [distributed checker](../../tilelang/analysis/distributed_checker.py) | 校验 ABI/op 元数据、方向/权限/区域/信号访问、event 生命周期和原生执行支持边界 |
| [Host session](../../tilelang/distributed/session.py)、[ABI](../../tilelang/distributed/abi.py)、[launch](../../tilelang/distributed/launch.py) | 单 CPU 协调器的分配、预检、epoch、状态、绑定和失败报告 |
| [capabilities](../../tilelang/ascend/distributed/capabilities.py) | 只开放本机 profile，拒绝未认证 transport |
| [cosim adapter](../../tilelang/ascend/distributed/cosim.py) | 从同一 TIR 构造各 rank 计划、区域身份、模型操作及计算片段回调 |
| [segment extraction/compiler](../../tilelang/ascend/distributed/segments.py) | 验证并提取真实局部 GEMM，复用当前 Ascend lowering/codegen，明确进行受限 SDK 转换和本机编译 |
| [AllGather + GEMM 示例](../../examples/distributed/example_allgather_gemm.py) | 源 DSL、Host 绑定、独立全局 oracle、报告和 CLI |

联仿适配器按 `(rank, object, element_offset, element_count)` 识别固定连续区域。完全相同的区域共享模型状态；部分相交区域、跨对象或跨 rank 的 NumPy storage alias 会拒绝，避免分别复制后掩盖真实别名。READY/ACK 由独立 SignalRef 表达，每个 int32 信号单元占用独立 64 字节位置。

未支持的 T.Stage、PerCoreTask、其他属性作用域、非平凡 SBlock predicate/iteration/match alias、一般 local copy 转换或未知计算语义不会被静默擦除。整个 rank 计划的 peer 与发布 epoch 在协议事件执行前预检。`fence` 保守实现为 `quiet`；正常出口再 drain 一次，但不替代协议中的尾部 ACK。

## 4. Host API 与当前 profile 的差别

`D.init(process_group=None, device=None, profile="ascend950_cannsim_v1", world_size=2)` 创建一个 CPU 协调器，持有两个 rank 的独立数组。真实 process group、设备绑定和 device stream 暂不支持。BF16 数组按原始 `uint16` bits 保存，不将 NumPy 的整数乘法误当 BF16 计算。

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

descriptor 使用固定 header 和对象/rank 记录，包含 generation、launch、rank、epoch 区间和物理大小，所有地址字段为 0。它只承担 CPU profile 的 ABI 一致性检查。可复用身份包括 canonical IR 摘要和编译 profile，排除具体 epoch 和运行时地址；当前没有实现含 TD 的原生 JIT 二进制缓存或真实通信库 ABI 缓存失效流程。

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

## 6. 实际验证结果

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

最终代码重跑正例的 canonical IR SHA256 为 `7cd0ee2d586e45acc1e5efb99da8d7faf915ddb92d043bc5fb6a8f32dba39cbd`。提取的两种局部片段均编译出 SHA256 为 `92d04e122accfdd7a9419f178d823041b89d4c886b846ecb83c87a602859c763` 的同一计算 ELF；不同的输入来源和输出位置由各片段的 `dsl_context` 记录。本机记录没有可用源码 span，因此用序列化 IR 的操作/区域/参数对应关系证明提取来源，不声称已验证源码行映射。

以下为首次验收时生成的本地产物标识，未提交到 Git；在其他机器执行时会产生新的目录名。完整复现流程和外部 helper 补丁见[测试指南](../testing/distributed_p0.md)。

- 正例：总报告（`output/distributed/validation_20261023/p0_rank_segments_hz9qbbkt/report.md`）、manifest（`output/distributed/validation_20261023/p0_rank_segments_hz9qbbkt/manifest.json`）、流水索引（`output/distributed/validation_20261023/p0_rank_segments_hz9qbbkt/trace_index.json`）。
- 负例：总报告（`output/distributed/validation_20261023/p0_rank_segments_jzm0g_y4/report.md`）、manifest（`output/distributed/validation_20261023/p0_rank_segments_jzm0g_y4/manifest.json`）、流水索引（`output/distributed/validation_20261023/p0_rank_segments_jzm0g_y4/trace_index.json`）。
- 独立审计：正例审计（`output/distributed/validation_20261023/p0_rank_segments_hz9qbbkt/independent_audit.json`）、负例审计（`output/distributed/validation_20261023/p0_rank_segments_jzm0g_y4/independent_audit.json`）、审计脚本（`output/distributed/validation_20261023/independent_artifact_audit.py`）。

独立审计重新读取原始输入、片段实际输入/输出、ELF、SDK diff 和 trace，核对文件 hash、参数 ABI、每个输出 tile 的 rank/owner/round 位置，以及 AIC Core 0 的 MMAD/END/detail。全局数值使用 64 次 FP64 外积累加重新计算，独立于生产校验的矩阵乘法实现。两个审计均通过；负例审计通过表示成功验证“应失败且恰有预期通信污染”，不是把负例的数值结果判成正确。

## 7. 本机回归范围

新增功能的 focused 回归共 **131 passed**：

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

## 8. 与完整原生 P0 的差距

当前只支持两个逻辑 rank、一个静态组/通道、单 active launch、int32 信号、连续静态区域和受限结构化控制流。通信 Buffer 的部分相交切片、跨对象 alias、任意 layout、额外计算效果、Stage/Pipelined/mixed 通信调度均不在该 profile 的支持集内；零轮空工作尚未提供专门 Host 快速路径。

协议请求和完成由 CPU 模型调度；CANN 片段逐个独立运行在 device 0。单卡流水有自己的时间原点，协议 tick 无物理时间单位；当前报告没有把它们拼成统一的双卡物理流水，也没有给出带宽、时延或重叠收益。

以下仍未实施：真实 SHMEM/HCOMM/UDMA transport、设备端 acquire/release 缓存动作、通信 scratch/event 规划、通信 effect 对 AutoSchedule/InsertSync/MultiBuffer 的完整接入、跨进程一致性与设备 stream/lease 清理、含 TD 全函数的 `tvm_ffi` launch、原生双卡压力验证和性能基线。当前工作不因此要求使用保密真机环境；相关 profile 继续保持未认证，后续可在相同 DSL 契约上增加已认证的执行路径。
