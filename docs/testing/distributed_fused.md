# Ascend 950 P0 融合路径验收

统一入口为 [run_p0_validation.py](../../examples/distributed/run_p0_validation.py)。
它分别记录 CPU 回归、真实 SDK 编译、真实双卡执行及性能，默认只运行 CPU 阶段。
**在收到服务器报告前，多卡正确性、可见性、进度和性能均为“待实机验收”。**
编译成功、CPU 协议模型或单卡 Simulator 不能替代双卡测试。

## 支持范围

| 执行选项 | profile | 验收内容 |
| --- | --- | --- |
| `phased` | `ascend950_shmem_mte_phased_v1` | 保留的分段 SHMEM 通信、独立本地计算基线 |
| `p0a` | `ascend950_shmem_mte_p0a_v1` | 每 rank 一个 pure AIV kernel；put/get、GM↔UB、受支持的 FP32 SIMD 向量操作、显式完成/可见性和退出 drain |
| `p0b` | `ascend950_shmem_mte_p0b_v1` | 一个 mixed block、AIV0 唯一通信发起者；同 kernel 中手写 READY/ACK 的 AllGather＋BF16 64×64 GEMM、FP32 输出 |

共同限制：单机双 rank、一个静态组/通道、一个活动 Session/launch；固定 `dav-3510`、AscendC、SHMEM MTE。
数据端点显式分配为对称 GM，通信偏移和长度满足 32 字节对齐，信号 int32、每个单元隔离 64 字节。
不扩展任意 shape/dtype、一般 Pipelined/Stage 通信、多发起核、多通道、PTO 或其他 transport。
P0-B 由编译器对受支持的静态轮次计划建立本地任务依赖；每 rank 展开后最多 4096 条计划指令，
超出时报 unsupported，不意味着任意用户流水都合法。
零工作计划明确拒绝，调用方须在 Host 跳过；4096 条上限内仍须满足设备资源容量。
普通 NPU Tensor 的通用分布式 JIT 输入输出、跨算子内存池不在本次验收范围。

P0-A 向量例子在 [example_fused_exchange.py](../../examples/distributed/example_fused_exchange.py)：
本 rank FP32 tile 经 GM→UB、SIMD 倍增、UB→GM 后交换，接收端搬出后才 ACK。
独立 oracle 用未修改的原始 peer 输入乘二。P0-B 复用
[example_allgather_gemm.py](../../examples/distributed/example_allgather_gemm.py) 的源 TD 协议，
oracle 从原始全局 BF16 bits 解码后做 CPU FP64 矩阵乘法。

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
  --stages cpu,compile,device --sdk-profile cann-9.1 \
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

## 性能口径

```bash
python -m examples.distributed.run_p0_validation \
  --stages performance --benchmark-iterations 20 --timeout 1800 \
  --devices 0 1 --sdk-profile cann-9.1 --output-dir "$P0_RESULTS"
```

三个模式执行相同的 3 轮 BF16 AllGather＋64×64 GEMM，输出 FP32：

- `hccl`：Host HCCL AllGather，随后本地 TileLang Cube GEMM。
- `p0a`：一个 pure AIV 交换 kernel，随后本地 TileLang Cube GEMM；计算直接借用对称 GM view。
  P0-A 本身没有 Cube，不能将此模式称为 P0-A GEMM 融合。
- `p0b`：一个含通信与 GEMM 的 mixed kernel。

每模式独立启动两个 worker；统一入口使用各 rank 独立的新 TileLang cache 目录。
记录首次 build/cache load、首次执行、3 次 warmup 后 20 个稳态样本。
这些是带同步的 **Host wall time**：`execution` 包含提交、分布式预检和必要同步，
`end_to_end` 进一步包含每轮输入暂存、ticket 准备和输出回读。
原始输入生成、独立 CPU oracle、初始化/分配与首次编译不计入稳态样本。
CPU 数值校验逐次执行；任一次失败即无有效性能通过。
两个 rank 同序样本取较慢值，再报告中位数/min/max，不混用两个 rank 的局部最快值。
`performance_comparison` 的比值为 HCCL 中位数除以候选中位数，值大于 1 才表示相同口径更快。

直接运行 [benchmark_p0.py](../../examples/distributed/benchmark_p0.py) 时可能命中用户已有 cache，
因此字段保留 `first_build_or_cache_load_seconds` 名称，不能无条件称为冷编译。
这些入口不提供纯设备 event 延迟，也不凭 Host 计时证明通信与 Cube 实际重叠；
需进一步判断重叠时应在服务器保存对应 profiler trace，再脱敏回传摘要。
未实测前不承诺加速比。

## 结果和验收

每次统一运行创建 `output/distributed/p0-acceptance/run_*/`，不会覆盖历史记录：

- `summary.json`：机器可读阶段/case 状态、退出码、超时、数值与性能摘要。
- `provenance.json`：Git commit、工作区状态与子模块 revision。
- `cpu/pytest.xml`、`cpu/process.log`：回归总数及 skip 证据。
- `compile/<case>/artifacts/`：源码、TIR、ABI、SDK build 产物与 `validation.json`。
- `device/<case>/artifacts/p0_native_*/`：原始合成输入、各 rank 日志、输出、`worker.json` 和全局 `manifest.json`。
- `performance/<mode>/`：两个 rank JSON、torchrun 各 rank stdout/stderr、独立 cache 和性能汇总。
- 每个阶段的 `command.json`、`process.json`、`process.log` 保留命令、退出码和超时证据。

`device_acceptance=not_run` 表示没有执行设备用例；`passed_requested_cases` 只对本次选择的
case 有效。首版完整验收应执行全部默认 case，两个 rank 均退出 0、物理执行标志为 true，
P0-A/P0-B 每次恰有一个物理 kernel，独立数值/信号检查通过，负例被精确识别，全部 Session 正常关闭。
性能是独立阶段；正确性通过不等于有性能收益。

常见错误：SHMEM revision/头文件改变会触发 ABI 或 cache 身份不一致；缺 peer 映射、
错误物理设备 ID、SDK profile 不匹配、torch_npu/驱动版本不匹配都应先修复环境后重跑。
TCPStore 的 loopback 与反向解析问题见[原生排障](distributed_native.md)。
`physical_kernel_count` 不等于 1、缺少某 rank 日志/输出、超时或信号不符均为失败，不能仅看顶层进程曾启动。

请检查和脱敏后回传：commit/SHMEM revision、SDK 与 torch/torch_npu 版本、设备型号及逻辑拓扑、
阶段/case/shape/dtype/rounds/slots/launches、各 rank 退出码、超时标记、错误类型与首条错误、
每次 epoch 区间和最终 READY/ACK、数值误差与 mismatches、物理 kernel 数、负例检测结果、
性能口径/样本数/中位数/min/max、skip 数及原因。无需回传机器地址、主机名、账号、
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
