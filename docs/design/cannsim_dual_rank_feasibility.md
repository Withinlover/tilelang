# cannsim-helper 双卡仿真能力评估

日期：2026 年 10 月 23 日。

从新克隆部署和测试见[分布式 P0 测试指南](../testing/distributed_p0.md)。文中的本地产物路径仅标识历史验收记录，日志、输入和流水不随 Git 分发。

约束：真机环境保密且不可提供，当前工作只在本机开展。本文评估现有 skill、实际安装的 Simulator 和可行的本地扩展，不要求连接真机。

## 1. 结论

**当前 cannsim-helper 不能执行真实双卡通信仿真，但有扩展为“双 rank 协议验证 + 单卡计算片段联仿”的价值。限制不仅在 helper，原装 CANN Simulator 的设备与通信运行时也不支持所需路径。**

| 能力 | 本机结论 | 对 P0 的作用 |
| --- | --- | --- |
| 单卡实际 `.o` 的精度与指令流水 | 已重新运行通过 | 验证每个 rank 的本地计算、局部 DMA 和同步 |
| 原装 Runtime 的 device 0 / device 1 | 仅 device 0 可用 | 不能直接启动两个模拟设备 |
| P2P / SHMEM / IPC 通信 | 相关底层接口存在空实现 | API 返回成功不代表远端访问成立 |
| 两个进程各跑一个单卡模型 | 可以作为独立片段执行方式；本次未实施联仿 | 需要新增 Host 协调和数据交换，不能自动获得卡间通信 |
| 两个逻辑 rank 的协议状态机 | 可在 CPU 上实现；前次 P0 设计已做小规模模型检查 | 验证 READY/ACK、epoch、slot 复用和 get 源租约的抽象协议 |
| 完整融合 kernel 的双 rank 互等 | 当前环境不能认证 | 不能用分段执行通过替代持续 kernel 的进度证明 |
| 卡间带宽、延迟、缓存一致性 | 当前环境不能认证 | 只能报告参数化模型假设，不能给出实机性能结论 |

因此，建议继续用现有 helper 支撑 P0 的本地研发，新增有明确边界的协议/片段验证层；不将“修改 device ID、增加 blocks、启动两个 runner”作为双卡支持方案。

## 2. 本机环境和保护范围

- skill 实际路径：SKILL.md（`$CANNSIM_ROOT/.agents/skills/cannsim-helper/SKILL.md`）。
- 工具包：cannsim-helper（`$CANNSIM_ROOT`）。
- CANN：`/usr/local/Ascend/cann-9.1.0`。
- 模型：`Ascend950PR_9589`，实际解析到 `dav_3510`。
- 默认模型配置为 32 AIC、64 AIV；这是单芯片上的核数量，不是 32/64 张卡。
- 本次没有修改 CANN 库、配置或 helper 源码。helper 中已有未提交修改保持原样；新探针和结果放在其默认 `output/` 下。

库和配置的完整 SHA256 记录在 evidence_metadata.json（`$CANNSIM_ROOT/output/probe_dual_device_20261023_z50lc_aj/evidence_metadata.json`）。本结论针对这些实际安装文件，不能泛化为所有内部型号或未来版本永久不支持。

## 3. 官方支持范围与本机实测相符

官方 [NPU Simulator 文档](https://gitcode.com/cann/ops-math/blob/master/docs/zh/debug/npu_sim.md) 的约束明确限定单卡、device 0 和 AI Core 计算类算子，不支持 MC2/HCCL；支持型号包括 950PR/950DT。已读取官方原始 Markdown，并保存带哈希的本次快照（`$CANNSIM_ROOT/output/probe_dual_device_20261023_z50lc_aj/official_npu_sim.md`）。

没有找到公开官方材料证明该安装包能够模拟跨 chip 的 peer GM 映射、SHMEM 远端信号及统一卡间时间线。这里的“不支持 MC2/HCCL”没有被直接推导为所有其他通信 API 都不支持；SHMEM/P2P 的判断还依据下述本机实现检查。

### 3.1 设备与能力探针

使用与 helper 相同的 `record_environment` 选择原装 `libruntime_camodel.so`、Simulator 库路径和配置，并记录进程实际加载的库。每个设备测试在独立进程和工作目录内执行，避免前一次初始化状态污染结果。未启用 MIX 调度修复。

| 调用/测试 | 返回值与输出 | 判断 |
| --- | --- | --- |
| `aclInit` | 0 | 初始化正常 |
| `aclrtGetDeviceCount` | 返回 0，count=1 | 只有一个可见模拟设备 |
| `aclrtSetDevice(0)` / `aclrtGetDevice` | 返回 0，当前设备 0 | device 0 正常 |
| device 0 分配、64 字节 H2D/D2H | 全部返回 0，16 个 uint32 比较无差异 | 基础内存路径正常；不是 kernel 或跨卡测试 |
| `aclrtSetDevice(1)` | 107001 | 当前安装头文件定义为 `ACL_ERROR_RT_INVALID_DEVICEID` |
| `aclrtSetDevice(9999)` | 同样 107001 | 越界设备对照 |
| `aclrtDeviceCanAccessPeer(&out, 0, 1)` | 返回 0，但 out 从预置的 -777 未改变 | 未返回有效能力结果；不能认作 P2P 支持 |

最后一项针对不存在的 peer，单独看它不能证明有效 peer 的行为。结合设备数为 1，以及下面的空实现检查，足以排除“成功返回即表示本机具备双卡 P2P”的解释。

复现材料：

- 探针源码（`$CANNSIM_ROOT/output/probe_dual_device_20261023_z50lc_aj/device_probe.cpp`）
- 结构化结果（`$CANNSIM_ROOT/output/probe_dual_device_20261023_z50lc_aj/device_probe_results.json`）
- device 0 日志与加载库（`$CANNSIM_ROOT/output/probe_dual_device_20261023_z50lc_aj/device0/probe.log`）
- device 1 日志（`$CANNSIM_ROOT/output/probe_dual_device_20261023_z50lc_aj/device1/probe.log`）

### 3.2 驱动入口不只是“尚未封装”

本机 `dav_3510/lib/libnpu_drv.so` 的反汇编结果如下：

| 符号 | 实现特征 | 含义 |
| --- | --- | --- |
| `drvGetDevNum` | 向输出写入常数 1 | 与设备枚举一致 |
| `halGetChipCount` / `halGetDeviceCountFromChip` | 输出常数 1 | Runtime 侧没有暴露第二个 chip/device |
| `halDeviceEnableP2P` / `halDeviceCanAccessPeer` | `xor eax,eax; ret` | 没有启用或查询 P2P 的实际动作 |
| `halShmemCreateHandle` / `halShmemOpenHandle` 等 | 同样直接返回 0 | 没有建立共享内存 handle/映射的实现 |
| `drvCreateIpcNotify` / `drvOpenIpcNotify` 等 | 同样直接返回 0 | 没有 IPC 通知实现 |

关键反汇编保存在 driver_symbols.txt（`$CANNSIM_ROOT/output/probe_dual_device_20261023_z50lc_aj/driver_symbols.txt`）。这些结果说明，给 helper 加上对应的函数调用还不能形成通信环境；仅修改设备数量也不能补齐映射、传输、通知和进度。

### 3.3 底层多 chip 骨架的研究价值

`libmodel_top.so` 中存在 `startModel_device`、`get_chip_num`、`get_die_num_per_chip`，并出现 `num_chip` / `num_die` 配置和 chip/die 初始化循环。这是一个潜在研究入口，但目前只证实内部有多实例初始化结构。

没有证据证明这些内部入口同时提供独立 GM 地址空间、跨模型路由、远端缓存一致性、通信队列和统一时钟。die、chip、Runtime device 与通信 rank 也不能直接等同。本次没有修改私有配置、绕过 Runtime 启动多 chip，或给 vendor 二进制打补丁。

若将来探索这条路径，最低成功标准应同时包括：两个独立且可识别的地址空间；真实的 peer 写入/读取；双方持续运行并等待通知时能推进；完整信号观察；可区分 rank 的 trace；可解释的统一时间。仅创建两个实例不算通过。这属于底层仿真平台研究，不能按 helper 小改估算。

## 4. 重新确认现有单卡能力正常

复用已经明确 ABI 的 GEMM 对象，并核对对应 IR/LLVM 签名：BF16 `A[64,64] @ B[64,64]`，FP32 输出，A/B/C 三个指针加 int32 保留参数，tiling key=0，blocks=1，覆盖完整 64×64 输出。

通过原有 `run.sh` 新跑一轮：seed=20261023、rtol=atol=1e-5、`--mix-scheduler-fix off`。这次关闭修复是为了确认原装单 block 路径，不表示多 block MIX 修复无用。

| 检查 | 本次结果 |
| --- | --- |
| 整体 manifest / 录制 / 报告 | passed=true，record_returncode=0，report_complete=true |
| 数值覆盖 | 4096/4096 元素，mismatches=0，nonfinite=0 |
| 最大绝对误差 | 3.814697265625e-6 |
| RMSE | 8.003740907904973e-7 |
| Core 0 流水 | 673 条事件，625 条带 detail；complete=true |
| END | 观察到的 AIC、AIV0、AIV1 均结束 |
| MIX 修复 | not_applied；没有声称本次验证了修复 |

流水中的 AIC/AIV0/AIV1 属于同一芯片的本地执行资源，不代表三个 rank。Core 0 的 observed span 约 3.892 µs 是模型标称时间，不是整卡或双卡实机延迟。

结果文件：

- manifest.json（`$CANNSIM_ROOT/output/run_gemm_s64_tm9l2cbd/manifest.json`）
- accuracy.json（`$CANNSIM_ROOT/output/run_gemm_s64_tm9l2cbd/accuracy.json`）
- 完整流水 trace.json.gz（`$CANNSIM_ROOT/output/run_gemm_s64_tm9l2cbd/trace.json.gz`）
- 隐藏 SET/WAIT 的流水 trace_no_set_wait.json.gz（`$CANNSIM_ROOT/output/run_gemm_s64_tm9l2cbd/trace_no_set_wait.json.gz`）

两份图来自同次执行，均可导入 Perfetto/Chrome Trace，点击指令查看 detail。这个实验确认现有 helper 可继续作为计算验证基础，没有测试任何双卡数据传输。

## 5. helper 自身还缺哪些结构

| 现有边界 | 源码位置 | 扩展时要处理的内容 |
| --- | --- | --- |
| 固定 device 0 | runtime.cpp:105（`$CANNSIM_ROOT/src/utils/runtime.cpp:105`） | 设备/context 所有权和清理；不是只增加 CLI 参数 |
| 单 stream，launch 后立即同步 | runtime.cpp:138（`$CANNSIM_ROOT/src/utils/runtime.cpp:138`） | 提交与等待分离，防止 rank 0 等待时 rank 1 尚未启动 |
| 一进程一次 trace，进程全局状态 | trace.cpp:25（`$CANNSIM_ROOT/src/utils/trace.cpp:25`）、trace.cpp:66（`$CANNSIM_ROOT/src/utils/trace.cpp:66`） | 多次 launch、并发实例不能直接复用 |
| 回调没有 device/rank 字段 | trace.cpp:14（`$CANNSIM_ROOT/src/utils/trace.cpp:14`） | 联合报告必须保留 rank/segment 身份；不能猜事件归属 |
| 单个 FP32 输出和连续行前缀 | case.h（`$CANNSIM_ROOT/src/cases/case.h`） | 通信验证需要 payload、signal、状态区的不同 dtype 和完整覆盖 |
| 一个录制子进程 | run.py（`$CANNSIM_ROOT/run.py`） | 新增 rank 协调器、失败传播和联合结果 |
| MIX 修复的全局/单 launch 假设 | tight_mix_completion.cpp（`$CANNSIM_ROOT/src/fixes/tight_mix_completion.cpp`） | 不能直接用同一进程的并发 rank；初期优先进程隔离 |

现有 `--blocks` 表示单设备逻辑 block 数，不能作为 `world_size`。分开启动两个 runner 只产生两个隔离的单卡世界，除非另建通信模型，它们不会互相看到 payload 或信号。

## 6. 推荐的本地扩展方案

### 6.1 保持三种结果分开

在现有单卡流程之外增加明确的验证模式。以下模式已在后续扩展中完成初版（见第 8 节）：

| 模式 | 执行内容 | 可以声称什么 |
| --- | --- | --- |
| `single-kernel` | 现有真实 `.o` + 单卡 Simulator | 本地数值/流水验证 |
| `protocol` | CPU 上的多 rank 通信状态机 | 给定通信契约下的协议正确性、交错与失败检查 |
| `rank-segments` | 协调器推进协议，各 rank 的本地片段交给单卡 Simulator | 分片、通信数据对应关系、片段计算和重组的功能验证 |

不要提供一个看似支持真实硬件的 `dual-npu` 模式再在内部默默替换成 Host 拷贝。报告必须写明使用的 transport model、假定的内存序、是否发生实际 kernel 执行，以及尚未验证的边界。

### 6.2 协议模型应覆盖的状态

以 P0 DSL 的 group、object、RemoteRegion、event、signal 作为输入，分别维护每个逻辑 rank 的 GM 区域和对象权限。将以下事件分离，支持可重复的延迟/重排注入：

1. put/get 发起、put 的 local/remote 完成、get 的 local 完成及远端源租约释放。
2. release 前缀、READY 送达、acquire 返回。
3. 消费开始/完成、ACK 送达、slot 再利用。
4. epoch 区间、多 launch 组内静止、异常退出。

用小规模穷举和随机调度捕获读错代、覆盖未消费 slot、get 源过早复用、等待未发起事件和无可推进状态。负例必须能触发检查，不能将实现本身用作 reference。通信时间属于模型参数，不是从单卡 trace 推导出的真实互连延迟。

模型需允许“payload 已写到目标但尚不可由某类读取观察”的抽象状态，以检查 compiler/DSL 是否要求必要 acquire；即便该抽象检查通过，也只说明程序遵守模型契约，不能证明 950 的具体缓存指令实现了它。

### 6.3 片段联仿的最小实现

协调器为两个 rank 建立分离的对象表，按通信事件显式转移数据；每个计算片段在独立 runner 进程里使用本地 device 0，保存独立输入、输出、manifest 和 trace。首版采用顺序、可复现调度，先验证功能，不追求模拟并发速度。

首版 GEMM 可复用公共单输出接口，由协议模型管理 payload 与信号，并记录 rank/segment/slot/epoch 元数据。通用融合算子仍需要多输出接口及显式的编译期切分。不能在完整 kernel 已经卡在 `wait_signal` 时，假设 Host 随时可以安全取出片上状态并恢复。

切分会把本来驻留 UB/L1 的中间值搬到 GM/Host，改变资源占用、同步和流水。因此片段联仿可以验证数学数据流，但不能据此证明原始融合二进制的进度、寄存器/片上存储生命期或性能。完整本地片段仍需要原 helper 的数值和流水检查。

联合报告可显示两个 rank 的逻辑通信事件和各自单卡流水。逻辑协议时间与模型指令 tick 应标为不同时间域；不能直接将两份 trace 的 tick=0 对齐后声称是实际通信重叠。

### 6.4 对当前 P0 设计的落实

在无法提供真机的条件下，可以完成并验收：DSL 类型与诊断、IR 的 effect 和异步生命期、ABI/cache 绑定、双 rank 抽象协议、本地计算片段的代码生成/精度/流水，以及通信适配器的源码和依赖检查。

尚不能认证：具体 950 peer 映射、实际 signal 不撕裂与 release/acquire、跨卡缓存行为、持续运行的双 rank kernel 进度，以及卡间性能。将这些结果标为“本机未验证”，不要求用户提供保密环境，也不以模型结果替代它们。

建议先实现 `protocol` 并接入 P0 IR，再做一个小型 `rank-segments` 交换加本地计算样例。底层私有多 chip 路线作为独立探索项，不作为 P0 本地开发的前置条件。

## 7. 首次能力调研完成范围

已完成 skill/源码审阅、官方说明核验、本机设备与 P2P 探针、关键 driver 空实现检查、单卡 GEMM 精度及完整 trace 复测。没有创建双卡运行环境，没有实现上述新模式，也没有修改 vendor Simulator。

当前最有价值的结论是：可以继续在本机推进 P0 的软件和协议设计，但需要显式增加验证层；现有工具或底层 Runtime 的小改不足以得到真实双卡通信仿真。

## 8. 后续扩展与实测（2026-10-23）

用户确认采用上述路线后，已在 cannsim-helper 中实现 `--mode protocol` 和 `--mode rank-segments`，
保留默认单卡入口。通信由 CPU 离散事件模型执行，各计算片段独立启动真实单卡 Simulator。

双 rank AllGather + GEMM 正例采用 3 轮、2 槽、延迟消费者，12 个真实 GEMM 片段均通过，
全局 49,152 个输出无 mismatch，最大绝对误差 `1.9371509552001953e-7`（rtol/atol 均为 `1e-5`），
24 份完整/精简流水齐全。通信 payload 污染负例的 4 个局部 GEMM 均通过，但独立全局对照发现
64 个错误元素并返回退出码 1。完整回归 84 项通过。

实现、复现命令和本次产物见 双 rank 联仿验证报告（`$CANNSIM_ROOT/docs/rank-cosimulation-validation-20261023.md`）。
该扩展尚未接入 P0 DSL parser/IR 或自动 kernel 切分；不改变本文对真实双设备通信不可认证的结论。
