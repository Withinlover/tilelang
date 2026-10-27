# Ascend 950 P0 融合路径验收

统一入口为 [run_p0_validation.py](../../examples/distributed/run_p0_validation.py)。
它分别记录 CPU 回归、真实 SDK 编译、真实双卡执行及性能，默认只运行 CPU 阶段。
**完整设备矩阵仍须对应的服务器报告验收；性能用例通过只覆盖本次被测 workload。**
编译成功、CPU 协议模型或单卡 Simulator 不能替代双卡测试。

## 支持范围

| 执行选项 | profile | 验收内容 |
| --- | --- | --- |
| `phased` | `ascend950_shmem_mte_phased_v1` | 保留的分段 SHMEM 通信、独立本地计算基线 |
| `p0a` | `ascend950_shmem_mte_p0a_v1` | 每 rank 一个 pure AIV kernel；put/get、GM↔UB、受支持的 FP32 SIMD 向量操作、显式完成/可见性和退出 drain |
| `p0b` | `ascend950_shmem_mte_p0b_v1` | 一个 mixed block、AIV0 唯一通信发起者；同 kernel 中手写 READY/ACK 的 AllGather＋BF16 GEMM、FP32 输出；64×64 tile 与下文受限 packed M/N/K 循环 |

共同限制：单机双 rank、一个静态组/通道、一个活动 Session/launch；固定 `dav-3510`、AscendC、SHMEM MTE。
数据端点显式分配为对称 GM，通信偏移和长度满足 32 字节对齐，信号 int32、每个单元隔离 64 字节。
不扩展任意 shape/dtype、一般 Pipelined/Stage 通信、多发起核、多通道、PTO 或其他 transport。
P0-B 由编译器对受支持的静态轮次计划建立本地任务依赖；每 rank 展开后最多 4096 条计划指令，
超出时报 unsupported，不意味着任意用户流水都合法。
零工作计划明确拒绝，调用方须在 Host 跳过；4096 条上限内仍须满足设备资源容量。
普通 NPU Tensor 的通用分布式 JIT 输入输出、跨算子内存池不在本次验收范围。

融合路径另有两个显式 Host 执行策略：`D.compile(..., execution_policy="strict")`
保持默认契约，每次 launch 检查源码、helper 和 SHMEM 头文件内容；`execution_policy="artifact"`
复用首次完整 build 封存的编译输入和已加载 kernel。artifact 稳态保留源码根对象、manifest、
编译配置身份以及每次 epoch、regions、progress、Session/rank/binding 和跨 rank 检查。
实际源码校验方式记录为 `source_validation=identity|serialized`：只有当前 FFI 能证明
该源码对象不可变时才使用 identity 路径，否则保留序列化校验。性能入口要求各次 launch
及两个 rank 报告同一方式，避免把不同校验开销混为同一结果。
显式再次 build、新 scalar specialization 或缓存逐出后的重建仍做完整编译输入检查。
完整 build 后，若 epoch 平移证明覆盖当前区间，artifact 可复用静态计算/区域契约，
逐次检查真实 epoch、信号历史和进度，跨 rank 比较契约及 artifact 身份，不再遍历和
序列化计算 IR。证明不足或超出区间时回退完整计划检查；READY/ACK、DMA 生命周期、
资源独占和完整区域检查在封存前完成。公开 `plan()` 与 strict 的完整检查保持不变。
磁盘文件的后续编辑不会改变已经加载的 artifact；需要应用新内容时重新构建，需要逐次
检查磁盘变动时使用 strict。该区别必须随结果记录，不能把 Host 校验成本变化称为设备优化。

统一入口对 P0-A/P0-B 的 compile/device/performance 默认选择 artifact；phased 始终 strict。
`--execution-policy strict` 可对照原路径。底层 `run_native_distributed.py` 和 `D.compile`
的默认值仍为 strict，底层脚本拒绝 phased＋artifact；性能脚本默认显式 artifact。

P0-A 向量例子在 [example_fused_exchange.py](../../examples/distributed/example_fused_exchange.py)：
本 rank FP32 tile 经 GM→UB、SIMD 倍增、UB→GM 后交换，接收端搬出后才 ACK。
独立 oracle 用未修改的原始 peer 输入乘二。P0-B 复用
[example_allgather_gemm.py](../../examples/distributed/example_allgather_gemm.py) 的源 TD 协议，
oracle 从原始全局 BF16 bits 解码后做 CPU FP64 矩阵乘法。
真正的 1024×1024×1024 dense GEMM 使用独立的[大矩阵 shape sweep](#真正的大矩阵-shape-sweep)，
原三轮小 tile 基线和完整协议验收矩阵保持不变。

## 更新、依赖与构建

服务器只需执行本地命令，不要求远程访问。先将依赖按保密环境规则准备到本地。
已有 checkout 且工作区干净时：

```bash
cd /path/to/tilelang
git remote get-url fork  # 应为 git@github.com:Withinlover/tilelang.git
git fetch fork feat/distributed-p0-dsl
git switch feat/distributed-p0-dsl
git merge --ff-only fork/feat/distributed-p0-dsl
git submodule update --init --recursive
```

未配置 `fork` 时先执行 `git remote add fork git@github.com:Withinlover/tilelang.git`。
有本地修改或分支分叉时先保留并核对，不使用 reset/force。全新 checkout 可使用：

```bash
git clone --recurse-submodules --branch feat/distributed-p0-dsl \
  git@github.com:Withinlover/tilelang.git
cd tilelang
```

需要与 Ascend 950 匹配的驱动/固件、CANN/Bisheng、PyTorch、`torch_npu`、pytest，
以及固定 SHMEM revision `a232a76516c435cd39ecd10125fc6e14a55fc18b`。
SHMEM 源码、公开 device 头文件和运行库必须来自同一次构建，保留其 Git 元数据。
准备过程详见[原生依赖说明](distributed_native.md#2-环境和离线准备)。
本机 SDK 编译使用 CANN 9.1；`cann-9.1` 是受限接口转换，不是通用 SDK 兼容承诺。

```bash
export CANN_ROOT=/path/to/cann
source "$CANN_ROOT/set_env.sh"
export BISHENG_HOME="$CANN_ROOT"
export SHMEM_ROOT=/path/to/pinned/shmem
(
  cd "$SHMEM_ROOT"
  git rev-parse HEAD
  CMAKE_BUILD_PARALLEL_LEVEL=4 bash scripts/build.sh -soc_type Ascend950
)
export SHMEM_LIBRARY="$SHMEM_ROOT/install/shmem/lib/libshmem.so"
export LD_LIBRARY_PATH="$SHMEM_ROOT/install/shmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 已有匹配的环境可直接激活；以下为新环境示例。
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt pytest
# 在此安装服务器获准的、与 CANN 配套的 torch/torch_npu wheel。
USE_ASCEND=ON USE_CUDA=OFF \
  CMAKE_ARGS="-DUSE_ASCEND=ON -DUSE_CUDA=OFF" \
  python -m pip install -e . -v --no-build-isolation
python -c 'import torch, torch_npu; print(torch.npu.device_count())'
export P0_RESULTS="$PWD/output/distributed/p0-acceptance"
```

实机进程必须加载真实 runtime/driver，不可将 `devlib` 链接桩或 Simulator stub 当成运行库。
脚本不安装依赖、不下载数据、不上传结果、不自动修改系统配置。

## 统一入口与单项重跑

```bash
# 列出全部 case，不初始化设备。
python -m examples.distributed.run_p0_validation --list

# 一次运行 CPU、真实 SDK 编译和双卡；默认失败即停。
python -m examples.distributed.run_p0_validation \
  --stages cpu,compile,device --sdk-profile cann-9.1 --execution-policy artifact \
  --devices 0 1 --timeout 900 --operation-timeout 120 \
  --output-dir "$P0_RESULTS"

# 可分阶段执行；compile 不初始化设备。
python -m examples.distributed.run_p0_validation --stages cpu --output-dir "$P0_RESULTS"
python -m examples.distributed.run_p0_validation --stages compile --output-dir "$P0_RESULTS"
python -m examples.distributed.run_p0_validation --stages device --devices 0 1 --output-dir "$P0_RESULTS"

# 单独重跑失败项。--keep-going 可让独立 case 继续执行并汇总全部失败。
python -m examples.distributed.run_p0_validation \
  --stages compile,device --cases p0b-hot-slot,p0b-negative \
  --devices 0 1 --output-dir "$P0_RESULTS" --keep-going

# 对照 strict 路径：相同 case、输入和验收要求。
python -m examples.distributed.run_p0_validation \
  --stages compile,device --execution-policy strict \
  --devices 0 1 --output-dir "$P0_RESULTS"
```

默认 case 包括 phased put/get/GEMM；P0-A put/get/向量交换；P0-B AllGather＋GEMM。
正常用例运行 3 轮、2 slots、2 次 launch；hot-slot 为 7 轮、1 slot、3 次 launch，
并在每次 launch 的全部预检之后、首个真实 kernel 提交之前延迟 rank 1 200 ms。
它测试延迟到达 peer 的进度，不声称模拟了 kernel 内任意 DMA 延迟。
unused-slot 为 1 轮、2 slots，检查未使用信号维持零值。
信号跨 launch 不清零，输出每次先写 sentinel，两个 rank 的数值和最终 READY/ACK 都必须正确。

负例只修改 rank 0 首次 launch 的单个发送元素，原始 oracle 输入保持不变。
必须满足：两个 rank 正常完成、信号正确、远端 rank 数值确实与原始参考不符、
输出与“仅指定元素被改动”的独立参考一致、后续 launch 正确。
超时、缺设备、编译失败、无输出或其他随机误差不能作为负例通过。
底层 runner 的预期负例退出 1，统一入口核对证据后将该测试记为通过并返回 0。
直接调用底层 runner 的命令示例：

```bash
python -m examples.distributed.run_native_distributed \
  --mode run --execution p0a --case vector --rounds 7 --slots 1 --launches 3 \
  --execution-policy artifact \
  --rank-delay-ms 200 --delay-rank 1 --devices 0 1 \
  --sdk-profile cann-9.1 --output-dir "$P0_RESULTS/manual-vector"
```

CPU 阶段的 pytest XML 单独记录 passed/failed/error/skip；此阶段排除 `real_sdk` 与
`real_bridge_dlpack` 用例，将真实 SDK 构建留给 compile 阶段。
缺 Ascend native build、SDK、可选 `CANNSIM_ROOT` 或显式启用的 ABI 测试依赖时可能 skip；
应核对原因，不能当作设备覆盖。compile/device 阶段缺必需依赖会失败，不能自动变成通过。
`--timeout` 为每项外层监督秒数，`--operation-timeout` 用于控制面/运行时。
两 rank 使用不同进程组，失败或超时会终止相应子进程；保存原始退出码。
首次编译过慢可以增大超时，不能据此放宽数值或协议验收。

## 已有大报告的离线精简

已有报告过大时，**无需重跑 NPU**。更新脚本后直接用标准库 Python 离线导出：

```bash
# 输入可为旧 schema 1/2 的顶层 summary.json，也可为整个 run 目录。
python3 examples/distributed/summarize_p0.py \
  "$P0_RESULTS/run_xxx/summary.json" \
  --output-dir "$P0_RESULTS/run_xxx/shareable"
cat "$P0_RESULTS/run_xxx/shareable/report.md"

# 省略 --output-dir 时，在输入所在目录写 report.md 和 compact.json。
python3 examples/distributed/summarize_p0.py "$P0_RESULTS/run_xxx"
```

模块形式 `python -m examples.distributed.summarize_p0 ...` 同样可用。该工具不导入 TileLang、
不初始化设备、不要求 CANN/torch_npu，也不重新运行编译、正确性或性能测试。
`report.md` 固定约 59 行，`compact.json` 使用固定字段及有界 case 列表；不随迭代样本数增长。
每 mode 一行核心结果，附参数、首次 build/cache-load、首次执行/E2E 的两 rank 最大值，以及
外层和 native 分段中位数。除 HCCL/候选比值外，同时报告候选/HCCL 的 `slowdown_x`，
避免用低精度的 `0.0x` 隐藏真实差距；缺失、非有限或不可表示的字段为 `null`。
新增的 `execution_policy` 记录性能模式及各 compile/device case 的实际策略；旧报告和
HCCL 为 `null`。性能 `source_validation` 来自实际 run，缺失也为 `null`；Markdown 在既有
参数表合并展示 `policy / source guard`，报告仍约 59 行。

分享版只保留白名单 commit、依赖版本、case 参数、退出码/超时、数值误差及聚合统计。
账号、主机名、绝对路径、原始命令、原始错误文本和逐次样本不复制到分享文件；错误仅保留
允许的 `error_types`。读取旧报告只检查固定 `summary.json`/`detail.json`/`provenance.json`
以及 `performance/<mode>/rank0.json`、`rank1.json`，不跟随 JSON 中记录的任意文件路径。
没有 provenance 时 commit 为 `null`，不会拿当前 checkout 的 commit 冒充原测量版本。
旧 schema 1 无分段时明确标为 `legacy_no_stage_timings`；缺少逐次检查字段不会补成零错误。

重复导出同一原始输入会覆盖自己的 `report.md`/`compact.json`，不会修改原始 summary/detail。
仅持有分享版 `compact.json` 时也可再次导出，内容会再次经过相同白名单过滤。
导出记录 `new_device_execution=false`，表示**导出动作未新增设备执行**，不是否认原报告的设备记录。
`reported_passed`、`reported_device_acceptance` 只转述原证据状态，导出器不重新授予验收通过。
分享摘要不能作为 `assess_performance` 的完整证据输入。

后续新运行默认将完整数据保存在本地 `detail.json` 和 `rank*.json`，`summary.json` 使用
同样有界的 schema 3 分享格式，并自动生成 `report.md`。统一入口先读取完整性能 detail、
执行原严格验收，再生成顶层摘要；省略详细字段不会放宽通过条件。

## 性能口径

```bash
python -m examples.distributed.run_p0_validation \
  --stages performance --benchmark-iterations 20 --timeout 1800 \
  --execution-policy artifact --devices 0 1 --sdk-profile cann-9.1 --output-dir "$P0_RESULTS"

# 只切换 Host 执行策略；保留相同数学 workload 和计时边界。
python -m examples.distributed.run_p0_validation \
  --stages performance --benchmark-iterations 20 --timeout 1800 \
  --execution-policy strict --devices 0 1 --sdk-profile cann-9.1 --output-dir "$P0_RESULTS"
```

三个模式执行相同的 3 轮 BF16 AllGather＋64×64 GEMM，输出 FP32：

- `hccl`：Host HCCL AllGather，随后本地 TileLang Cube GEMM。
- `p0a`：一个 pure AIV 交换 kernel，随后本地 TileLang Cube GEMM；计算直接借用对称 GM view。
  P0-A 本身没有 Cube，不能将此模式称为 P0-A GEMM 融合。
- `p0b`：一个含通信与 GEMM 的 mixed kernel。

这里的“相同”指相同数学结果、6 个 GEMM tile/rank 和每 peer 共 24 KiB 的逻辑有效载荷。
HCCL 一次 `all_gather` 批量传输全部 3 轮，P0 每轮交换 8 KiB 并执行 READY/ACK，
用 2 个接收 slot 复用存储；P0-A 还包含 Input→Source、Recv→Output 的本地搬运。
因此该比较衡量当前三条完整实现的开销，不能把差值全部归因于通信 primitive 或融合技术。
报告的 `workload.communication_schedule`、`payload_batches_per_peer`、`receive_slots`
明确记录此差异。不得悄悄将 HCCL 改成逐轮调用后与旧批量结果混算；需要逐轮对照时应作为单独实验保留两种基线。

每模式独立启动两个 worker；统一入口使用各 rank 独立的新 TileLang cache 目录。
记录首次 build/cache load、首次执行、3 次 warmup 后 20 个稳态样本。
artifact 的封存/建立包含在首次 build/cache load 中。所有逐次协议检查仍在原 prepare、
launch 和 complete 边界内；没有从测得的 execution/e2e 中扣除检查或等待时间。
这些是带同步的 **Host wall time**：`execution` 包含提交、分布式预检和必要同步，
`end_to_end` 进一步包含每轮输入暂存、ticket 准备和输出回读。
原始输入生成、独立 CPU oracle、Session/设备缓冲初始化分配与首次编译不计入稳态样本。
暂存和回读实际使用的 Host 临时对象、Host 复制与绑定校验保留在各自计时中。
CPU 数值校验逐次执行；任一次失败即无有效性能通过。
两个 rank 同序样本取较慢值，再报告中位数/min/max，不混用两个 rank 的局部最快值。
同时保存样本数和 p90/p95，分位数按排序后 `(count - 1) × percentile` 位置线性插值；
20 个样本的尾部分位数仍只是本次样本描述。保留各 rank 原始逐次样本，不用两个 rank
各自的中位数相减或相加估计某个阶段。
`performance_comparison` 的比值为 HCCL 中位数除以候选中位数，值大于 1 才表示相同口径更快。

完整性能 `detail.json` 的 schema 2 保留原 `execution`/`end_to_end` 的边界，并记录不改变协议与同步动作的分段：

| `stages` 字段 | Host wall time 覆盖内容 |
| --- | --- |
| `prepare` | P0 ticket 准备，包含现有计划合法性验证和跨 rank 元数据检查 |
| `staging` / `staging_sync` | 原输入暂存/搬入、必要的输入同步；包含各路径实际的 Host 检查和复制方式 |
| `launch` | P0 `D.launch` 全部调用，包含验证、缓存检查、提交和内部完成等待 |
| `complete` | P0 `D.complete`，包含 Session 完成同步及跨 rank 完成检查 |
| `collective_submit` | HCCL `all_gather` 的完整 Host 调用；可能包含 Host 等待，不能当作纯提交延迟 |
| `local_gemm_submit` | HCCL/P0-A 本地 GEMM 调用；P0-A 包含借用对称 GM view 的构造和释放 |
| `final_sync` / `readback` | 原有最终设备同步和输出回读 |

不适用的字段省略。每个 rank 的 `timing_samples` 保留含 warmup 的逐次分段，`stages` 汇总
稳态样本。`native_run` 是 `launch` 内部的嵌套归因，包括 binding validation、plan validation、plan preflight、
build/cache validation、build preflight、绑定与提交、stream sync 和 report finalize；
其中 `total` 包含其子项，且整个 `native_run` 不在 `launch` 之外额外相加。
每项的两 rank 最大值可能来自不同 rank，分段最大值或
p95 不能相加还原整体时间。`outside_e2e_alignment_seconds` 记录原本就在计时边界外的
轮次对齐及初始同步；数据生成和 CPU oracle 仍在边界外。

`build_or_cache_validation` 与 `build_preflight` 在同一 rank 上相邻且不重叠。后一项包含
控制面处理和等待另一 rank 到达，不能全部视为额外 CPU 构建工作。例如假设某轮 rank 0
构建 5 ms、等待 0 ms，rank 1 构建 3 ms、等待 2 ms；各段取 rank 最大值会得到 5 ms 和
2 ms，但两个 rank 各自的合计都只有 5 ms。该例仅说明聚合含义，并非设备测量。
只有配对中位数时不能反推同一轮的 rank 偏斜，也不能把 preflight 中位数从 execution
中位数扣除后称为 kernel 时间。每 rank 的 execution 起点在各自输入暂存之后，launch
还可能等待 peer 完成暂存；当前指标是 rank 本地 Host wall time 的配对最大值，未提供
统一全局起点或纯设备 event 延迟。需要定位偏斜时应在服务器检查本地逐 rank/逐次明细。

上述分段仍是 Host wall time。`launch` 里的计划验证、IR/hash、跨 rank 元数据检查以及
缓存身份检查都是当前 API 成本；不能把这些移出计时后称为端到端提速。
分段计时自身的少量记录开销仍留在原边界内。没有在 Torch 默认 stream 上插入 event，
因为 SHMEM 使用 Session 自有 stream；本报告不能提供纯 device kernel 延迟。

直接运行 [benchmark_p0.py](../../examples/distributed/benchmark_p0.py) 时可能命中用户已有 cache，
因此字段保留 `first_build_or_cache_load_seconds` 名称，不能无条件称为冷编译。
这些入口不提供纯设备 event 延迟，也不凭 Host 计时证明通信与 Cube 实际重叠；
需进一步判断重叠时应在服务器保存对应 profiler trace，再脱敏回传摘要。
未实测前不承诺加速比。

每次迭代（含 warmup）的独立 FP64 oracle 检查保存 `passed`、`elements`、`mismatches`、
`nonfinite`、`max_abs_error` 和容差。两 rank 的结果与分段记录一并保存在本地性能 `detail.json`，
缺 rank、缺迭代、数值失败、无效样本或超时均不通过。
`correctness_scope=benchmark_workload_only`、`full_device_acceptance=not_assessed` 表示该报告
只证明这些合成输入的本次性能 workload 正确；它没有替代 put/get、hot-slot、多 launch、
快慢 rank 和负例的完整设备矩阵。统一入口的 `device_acceptance` 不会因性能阶段通过而改变。

## 结果和验收

每次统一运行创建 `output/distributed/p0-acceptance/run_*/`，不会覆盖历史记录：

- `summary.json`：schema 3 有界分享摘要；`reported_*` 忠实转述原验收状态。
- `report.md`：约 59 行的分享报告；不含原始命令、路径或逐次样本。
- `detail.json`：完整的阶段/case 验收证据、原始进程/性能报告，仅留本地。
- `provenance.json`：Git commit、工作区状态与子模块 revision。
- `cpu/pytest.xml`、`cpu/process.log`：回归总数及 skip 证据。
- `compile/<case>/artifacts/`：源码、TIR、ABI、SDK build 产物与 `validation.json`。
- `device/<case>/artifacts/p0_native_*/`：原始合成输入、各 rank 日志、输出、`worker.json` 和全局 `manifest.json`。
- `performance/<mode>/`：完整 `detail.json`、两个 rank JSON、torchrun 各 rank stdout/stderr、独立 cache，以及有界 `summary.json`/`report.md`。
- 每个阶段的 `command.json`、`process.json`、`process.log` 保留命令、退出码和超时证据。

本地 detail 的 `device_acceptance`（分享版为 `reported_device_acceptance`）中，`not_run`
表示没有执行设备用例；`passed_requested_cases` 只对本次选择的
case 有效。首版完整验收应执行全部默认 case，两个 rank 均退出 0、物理执行标志为 true，
P0-A/P0-B 每次恰有一个物理 kernel，独立数值/信号检查通过，负例被精确识别，全部 Session 正常关闭。
性能是独立阶段；正确性通过不等于有性能收益。

常见错误：SHMEM revision/头文件改变会触发 ABI 或 cache 身份不一致；缺 peer 映射、
错误物理设备 ID、SDK profile 不匹配、torch_npu/驱动版本不匹配都应先修复环境后重跑。
TCPStore 的 loopback 与反向解析问题见[原生排障](distributed_native.md)。
`physical_kernel_count` 不等于 1、缺少某 rank 日志/输出、超时或信号不符均为失败，不能仅看顶层进程曾启动。

优先检查后回传 `report.md` 或 `compact.json`/新版 `summary.json`，无需粘贴上万行 detail。
这些报告包含 commit/SHMEM revision、依赖版本、case 参数、退出/超时、误差和聚合性能。
需要进一步诊断时再提供经检查的补充字段：SDK 版本、设备型号及逻辑拓扑、
阶段/case/shape/dtype/rounds/slots/launches、各 rank 退出码、超时标记、错误类型与首条错误、
每次 epoch 区间和最终 READY/ACK、数值误差与 mismatches、物理 kernel 数、负例检测结果、
性能口径/批量方式/样本数/中位数/min/max/p90/p95、各 rank 的 `stages` 和嵌套 `native_run`
统计、逐次误差检查、skip 数及原因。无需回传机器地址、主机名、账号、
生产数据、原始 profiler 或未检查的完整日志；路径可替换为相对 case 名。

## 调度与资源验证的准确边界

融合路径始终启用 `AutoSchedule → PrepareMultiBuffer → ResolveCore → InsertSync →
MaterializeMultiBuffer`。SHMEM task 在 AIV0 上同步完成自身的 MTE 请求，调度完成端点
是 Scalar；`wait_signal(covers=...)` 的写 effect 与接收区 GM→L1 搬运建立 RAW 边，
`signal(release_regions=...)` 的写 effect 与该搬运建立 WAR 边。ACK 因而等待 AIC MTE2
真正完成读取。它不等待后续 L1→L0、MMAD 或 Fixpipe；AIV0 可在 peer ACK 满足后发起
下一轮通信，而 AIC 继续计算已搬入本地存储的 tile。这是静态展开有限协议下的受限流水，
并非一般动态循环的通信流水调度。

本机 IR 回归使用同一 AllGather＋GEMM 的两轮、单 slot 程序，检查最终两核任务流以及
每个 ACK 对应的 MTE2→Scalar 同步边；检查 Cube/MMAD/Fixpipe 没有被错误加入 ACK 的
等待条件，并检查下一次 put 位于下一次 AIC 完成等待之前。这证明编译器允许所述重叠，
不能证明真实设备上已发生重叠或获得性能提升。仍需实机数值、进度和 trace/计时验收。

现有直线任务 Z3 优化器有固定求解预算。本机一轮样例完成求解；两轮及以上样例可因
预算耗尽返回 `unknown (canceled)`，随后 AutoSchedule 使用合法拓扑顺序回退。
诊断现在保留求解器真实状态，不把预算耗尽误报为约束 `unsat`。回退保留通信顺序、
RAW/WAR 同步和两核流分离；它没有完成最优调度搜索。不要将求解成本、估计周期或
“成功生成 mixed kernel”当作设备重叠/性能实测结果。

所有通信 task 共用一个不参与版本化的 4096 字节 UB scratch。融合 kernel 保守地禁用
本地物理存储复用，普通本地 GEMM 的同步和版本机制仍通过原有 passes。SHMEM 搬运独占
hard-event ID 7；统一 flag 分配器将普通 flag 压入 0–6，必要时使用已有 mutex spill。
固定 SHMEM 版本的这条 put/get 路径只使用传入的 MTE2↔MTE3 event，950 信号发布走
Scalar AtomicExch；此结论不自动适用于其他 SHMEM 版本。

有限直线展开可以在已有完成依赖证明旧 WAIT 已消费后复用 flag。根作用域没有下一次
循环迭代，因此不要求不存在的反向 loop-carried 边；嵌套同步提升到父作用域时按整个
子作用域的开始/结束边界证明。循环 flag 保留原有跨迭代约束。独立 DMA 的词法发射
顺序不能作为完成证明，回归明确拒绝这种复用，并验证七轮单 slot 压力例仍生成一个
kernel。此规则只复用资源编号，保留原同步动作和依赖。

4096 条静态指令上限只限制展开规模，不承诺所有该规模以内的程序都满足设备资源
容量。若报 `RewriteFlagToBuf: total spilled mutex slots (...) exceed the shared pool of 32`，
表示当前展开的不可安全复用同步超过支持范围；应减少单次展开工作量或采用 phased
执行。不能删除同步或占用保留的 event 7 来绕过诊断。其他本地存储和跨核 flag 容量
也仍受编译器约束。

编译终检在调度/多缓冲后核对通信 task ID、顺序、标量 ABI、acquire/release 区域和
scratch 身份；指针展平及存储合并后再次核对任务顺序、epoch、effect 参数数量，以及
scratch 的实际物理字节范围。其他本地指针必须可证明与该范围不相交。回归包含最终 IR
中的本地指针被改入 scratch 的负例。上述检查保护本版编译契约，不替代硬件可见性认证。

## 真正的大矩阵 shape sweep

原有三轮 64×64 基线保持不变。独立入口 `examples.distributed.benchmark_p0_shapes`
用于更大通信和计算工作量：主用例每 rank 的 `M=N=K=1024`，每 rank 持有一个
1024×1024 BF16 A 和相同的 BF16 B，AllGather 后输出 2048×1024 FP32 矩阵。
K=1024 是设备上的 16 次真实累加，不以 16 个独立 K=64 问题或增加 rounds 替代。

支持 M、N、K 各为 64 的整数倍，范围 64–1024；默认先跑 128、256，再跑 1024。
内部布局为 `A[Mt,Kt,64,64]`、`B[Nt,Kt,64,64]`（末两维是 K,N），
`Output[owner,Mt,Nt,64,64]`。P0-B 的 `td.gemm` 有限区域保留设备 M/N/K 循环，
不把 1024 的计算展开成上千条 Python/协议指令。这是显式、受检查的固定布局契约；
其他 dtype、残块、任意 stride、任意循环体不在此范围内。

三个模式都传输每 rank 一个完整的 M×K BF16 A packet，执行相同的 dense GEMM。
HCCL 调用一次整 A AllGather，再调用一次普通 TileLang grid GEMM；P0-A 调用一次
pure AIV 整 A 交换，再调用**同一个** grid GEMM。两者每次只有一个本地计算 kernel
提交，grid 为 `2×Mt×Nt`，不会通过每 tile 的 Python launch 人为抬高基线开销。
P0-B 仍只有一个 mixed block，通信由 AIV0 发起，Cube 上执行两个有界计算区域。
1024 时普通本地 GEMM 有 512 个 block，而 P0-B 是一个 block；报告明确披露这种
资源差异，不预设融合路径会快于多核基线。

P0-B 源协议为 put、local GEMM、READY/acquire、remote GEMM、ACK、drain。
P0-A 的 ACK 不表示后续 host 发起的本地 GEMM 已结束；其接收区通过 host 协议保持：
`D.complete` 后读取 Send/Recv，本地 GEMM 完成并同步后，才能进入下一次双 rank
prepare/握手，随后才允许下一次 put。不能去掉末尾同步或把下一次 prepare 提前。

使用本页前面的环境、构建和 SHMEM 配置后，按阶段运行：

```bash
# CPU 的布局、真实 K 累加参考、错误检出、计时/报告回归；不访问设备。
python -m examples.distributed.benchmark_p0_shapes \
  --stages cpu --output-dir output/distributed/p0-shapes

# 真 SDK 编译三种规模/三个模式；每项独立超时，失败继续记录其他项。
python -m examples.distributed.benchmark_p0_shapes \
  --stages compile --shapes 128,256,1024 --modes hccl,p0a,p0b \
  --sdk-profile cann-9.1 --execution-policy artifact \
  --shmem-root "$SHMEM_ROOT" --shmem-library "$SHMEM_LIBRARY" \
  --timeout 900 --output-dir output/distributed/p0-shapes

# 真实双卡：每个模式预热 3 次、测量 20 次；全部迭代都检查数值。
python -m examples.distributed.benchmark_p0_shapes \
  --stages device --shapes 128,256,1024 --modes hccl,p0a,p0b \
  --devices 0 1 --iterations 20 --warmup 3 --measure-device \
  --sdk-profile cann-9.1 --execution-policy artifact \
  --shmem-root "$SHMEM_ROOT" --shmem-library "$SHMEM_LIBRARY" \
  --timeout 900 --operation-timeout 180 --output-dir output/distributed/p0-shapes

# 单独重跑 1024 的 P0-B；strict 可与 artifact 对照 host 校验成本。
python -m examples.distributed.benchmark_p0_shapes \
  --stages device --shapes 1024 --modes p0b --execution-policy strict \
  --devices 0 1 --iterations 20 --warmup 3 --measure-device \
  --shmem-root "$SHMEM_ROOT" --shmem-library "$SHMEM_LIBRARY" \
  --timeout 900 --output-dir output/distributed/p0-shapes
```

也可使用 `--shapes 64x128x256,1024` 选择非方形前置用例。规模超过契约、SDK 编译错误、
缺设备、超时或异常退出均记为失败；编译通过没有 `device_executed`，也没有设备通过含义。
默认只执行 compile；不会默认发起真实设备测试。
本节 `cpu` 只运行新 shape benchmark 的回归；完整 CPU/设备协议矩阵仍使用前文的
`run_p0_validation --stages cpu` 和 `--stages device`，不能由这些 shape 用例替代。

每次创建新的 `output/distributed/p0-shapes/run_*/`。顶层 `report.md` 和 `summary.json`
只含有界汇总，体积随选定 shape/mode 数量变化，不随 iterations 增长。
汇总包括 commit、SHMEM revision、SDK profile 和包版本；未找到的信息为 null。
每项的 `compile|device/<MxNxK>/<mode>/` 保存完整 `detail.json`、`rank0.json`/`rank1.json`、
torchrun 各 rank stdout/stderr、进程退出/超时记录和独立冷缓存。完整命令、绝对路径、
原始错误只留本地明细。源码或资源限制导致的编译失败不会被当成 skip 或设备成功。

host execution 包括通信/计算提交、原有动态协议校验和必要同步；E2E 另外包括 dense
输入打包、本 rank A/B 暂存、ticket 准备、输出 readback 和解包，预分配设备存储仍会
产生 host 临时对象。每次的独立 FP64 dense oracle 使用原始 BF16 输入，位于计时外，
检查所有输出，容差为 `atol=2e-4, rtol=2e-5`。首次构建/缓存加载、首次 execution/E2E、
稳态两 rank 逐次最大值的 median/p95 均保留；不同阶段的 rank 最大值/中位数不能相加。

`--measure-device` 额外记录实际 stream 的 event interval。Session 预分配 ACL events，
在其 SHMEM stream 上围绕 native execute 记录；HCCL/普通 GEMM 在实际当前 Torch stream
记录。它们可能包含 host 提交造成的 stream 空闲，不应称为纯 kernel latency。
P0-A 的通信和本地 GEMM 位于不同 stream，分别列为 `communication`、`local_gemm`，
不相加宣称关键路径；HCCL 为 `collective_and_local_gemm`，P0-B 为 `fused`。
HCCL 另在 AllGather 提交后记录中间 event，分别列出 `collective` 与 `local_gemm`；
配对各 rank 的最大值再取中位数后，这两个统计值同样不能相加。
未请求或没有数据时字段为 null。event 计时不替代原 host execution/E2E，也不证明重叠。
P0-B 的一个 fused interval 不能拆分实际通信和计算占比；确认两者重叠仍需 profiler
或进一步的单 block 计算对照，不能仅凭更大的 shape 推断已实现有效重叠。
报告同时列出实际 `source_validation` 与 `launch_validation`（`static_contract` 或
`full_plan`），用于识别 host 校验路径；这些字段在两 rank 和全部迭代中必须一致。

吞吐按每 rank `4×M×N×K` useful FLOPs 和 `2×M×K` logical peer bytes 分别除以完整
execution/E2E 时间报告；后者不是物理链路带宽。性能结果只证明本次 workload 的数值，
不替代完整 put/get、slot 复用、延迟、负例设备验收。回传经检查的 `report.md` 和
`summary.json` 即可；保留原始各 rank 样本供需要时脱敏诊断。
