# Ascend 950 原生双卡 P0 DSL 验证

本指南对应 `feat/distributed-p0-dsl` 分支的新增 profile：`ascend950_shmem_mte_phased_v1`。
入口是 [run_native_distributed.py](../../examples/distributed/run_native_distributed.py)。
它从同一个 TD PrimFunc 生成两 rank 的顺序计划，将 TD 通信编译成真实 SHMEM MTE AIV kernel，
将原始 `T.copy/T.gemm` 语句提取为普通 TileLang Cube kernel，经既有 `tvm_ffi` 执行。
通信数据不经过 CPU 协议模型、NumPy 或 HCCL；CPU 只负责初始化、提交、元信息一致性检查和结果校验。

**本机已验证源码、原生编译和 Host ABI；真实双卡执行及互联可见性尚待实机验收。**
选择该 profile 就是明确选择这条待实机验证的路径，不需要删除原生入口限制或关闭语义检查。
原始含 TD 的函数仍不能直接交给 `tilelang.compile`；使用 `D.compile(..., profile=D.NATIVE_PROFILE)`。

## 1. 支持范围与执行方式

| 项目 | 本次实现 |
| --- | --- |
| 平台 | 单机两张不同的 Ascend 950；每 rank 独立进程、独立 NPU stream |
| 通信 | SHMEM 建立合法 peer 映射；AIV 发起 MTE put/get、原子信号 SET、设备端 wait |
| 完成 | 每个通信 kernel 在退出前完成自身操作；`copy_async` 可采用规范允许的更强同步完成 |
| 缓冲协议 | 64 字节隔离的 int32 READY/ACK；epoch 连续分配，跨 launch 不清零 |
| 本地计算 | 连续 GM copy；原始 BF16 64×64 GEMM，FP32 输出 |
| 控制流 | `T.Kernel(1)`、可由 rank/Host 标量确定的 serial/条件；一个组和通道 |
| 未支持 | 多机、超过两 rank、任意非连续区域、任意计算、PTO、Stage/Pipelined 通信、多发起核、graph replay |
| 融合 | 通信与计算分 kernel 顺序执行；未实现原设计的单 kernel P0-A 和 P0-B 自动融合/重叠 |

通信长度及起始字节偏移须为 32 的倍数。P0 信号仍为正 int32；950 的公开原子交换接口使用
uint32 位视图，限定 `1..2^31-1` 后，SET 与 GE 比较和原 DSL 完全一致，无隐式截断。
所有通信 kernel 都自行 drain，后一个 kernel 不依赖前一个被调度到同一物理 AIV。
同 stream 的执行顺序连接计算完成、发布、获取和 ACK；程序退出后还要全组确认才释放内存。

实现依赖以下固定版本的公开契约：

- [SHMEM completion 定义](https://gitcode.com/cann/shmem/blob/a232a76516c435cd39ecd10125fc6e14a55fc18b/include/device/gm2gm/shmem_device_mo.h)：quiet 完成此前对称内存操作并保证目的 PE 可见。
- [MTE 实现](https://gitcode.com/cann/shmem/blob/a232a76516c435cd39ecd10125fc6e14a55fc18b/src/device/gm2gm/engine/shmem_device_mte.hpp)：显式 UB scratch、事件和 MTE quiet；本实现使用独占 4096 字节 scratch 和事件 0。
- [原子完成说明](https://gitcode.com/cann/shmem/blob/a232a76516c435cd39ecd10125fc6e14a55fc18b/docs/api/atomic_api_sync_async_comparison.md)：MTE 原子操作之后执行 quiet。

这些契约是实现依据，不替代双卡热缓存、slot 复用和实际进度测试。接收端的标准 Cube GM→L1
读取不走 scalar/SIMT/NDDMA 数据缓存；不将这个结论推广到当前未支持的读取路径。

SHMEM 使用默认实例 0：Host 初始化在 950 上建立共享的设备元数据映射并写入 peer 状态，
各独立 ELF 的 `aclshmemi_get_state()` 从该映射读取状态，无需额外 kernel 参数或逐 kernel 注册。
本实现没有启用 SHMEM 的 `MULTI_INSTANCE` 模式，每进程只允许一个存活的原生 Session。

## 2. 环境和离线准备

准备当前分支与全部子模块、支持实际 950 型号的驱动/固件、CANN/Bisheng、PyTorch/`torch_npu`，
以及下述固定 SHMEM 源码。普通构建依赖参照 [Ascend 安装](../../tilelang/ascend/README.md)。
本机编译验证使用 CANN 9.1，目标明确固定为 `dav-3510`；实机首次建议使用相匹配的 SDK。
`--sdk-profile cann-9.1` 仅转换审查过的 GEMM 和 SHMEM 头文件/接口，不是任意 CANN 版本兼容层。
`--sdk-profile native` 使用仓库原始生成代码，须先确认它与目标机 SDK 的编译检查通过。

有网准备阶段获取依赖；保密环境使用获准的离线源码和 wheelhouse。SHMEM 需要保留 Git 元数据，
因为 Host bridge 检查精确 revision 及 `include/`、`src/` 的工作区状态。必须用同一份源码构建
运行库；库内容哈希用于缓存与两 rank 一致性检查，本身不能证明任意外部 `.so` 的源码来源。

```bash
git clone --recurse-submodules --branch feat/distributed-p0-dsl \
  git@github.com:Withinlover/tilelang.git
git clone https://gitcode.com/cann/shmem.git
git -C shmem checkout --detach a232a76516c435cd39ecd10125fc6e14a55fc18b

# Ascend950 的 SHMEM 构建依赖。先备齐这个目录，离线构建脚本就不会下载它。
mkdir -p shmem/3rdparty
git clone --branch v3.11.3 --depth 1 https://github.com/nlohmann/json.git shmem/3rdparty/json
```

在目标机器按其安装方式加载 **真实 CANN runtime** 的环境脚本，确认 `ASCEND_HOME_PATH` 和
`PATH` 指向同一套 SDK；不要用 Simulator 的 runtime/driver stub 代替实机库。
随后在 SHMEM 源码根目录编译 950 库：

```bash
export SHMEM_ROOT="$(pwd)/shmem"  # 根据实际离线导入位置设置
(
  cd "$SHMEM_ROOT"
  CMAKE_BUILD_PARALLEL_LEVEL=4 bash scripts/build.sh -soc_type Ascend950
)
export SHMEM_LIBRARY="$SHMEM_ROOT/install/shmem/lib/libshmem.so"
test -f "$SHMEM_LIBRARY"
export LD_LIBRARY_PATH="$SHMEM_ROOT/install/shmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

保留完整 install 树及依赖库；
`SHMEM_ROOT` 指向源码根目录，不能只指向 `lib/` 或不含 device 实现头文件的安装目录。
此 profile 初始化时明确选择 MTE，不启用 RDMA/UDMA 作为隐式回退。

在 TileLang 根目录构建：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
# torch/torch_npu 按目标机 CANN 配套版本预先安装，离线环境使用本地 wheelhouse。
USE_ASCEND=ON USE_CUDA=OFF \
  CMAKE_ARGS="-DUSE_ASCEND=ON -DUSE_CUDA=OFF" \
  python -m pip install -e . -v --no-build-isolation
python -c 'import torch, torch_npu; print(torch.npu.device_count()); print(torch.npu.get_device_name(0))'
export P0_NATIVE_OUTPUT="$PWD/output/distributed/native"
mkdir -p "$P0_NATIVE_OUTPUT"
git rev-parse HEAD > "$P0_NATIVE_OUTPUT/tilelang_commit.txt"
git submodule status --recursive > "$P0_NATIVE_OUTPUT/submodules.txt"
git -C "$SHMEM_ROOT" rev-parse HEAD > "$P0_NATIVE_OUTPUT/shmem_commit.txt"
sha256sum "$SHMEM_LIBRARY" > "$P0_NATIVE_OUTPUT/shmem_library.sha256"
```

至少两张不同的物理卡可用，且它们之间 SHMEM MTE peer mapping 可建立。设备编号是当前进程
可见的 NPU 索引；初始化还检查实际物理 ID，避免可见编号不同却落在同一卡。
初始化/元信息通信只使用 loopback 两个独立端口；测试入口不上传日志或数据。

## 3. 本机可先执行的检查

源码生成不需要 SHMEM、CANN 或 NPU，也不需要 cannsim-helper：

```bash
python -m examples.distributed.run_native_distributed \
  --mode lower --case gemm --rounds 3 --slots 2 \
  --output-dir "$P0_NATIVE_OUTPUT/gemm_lower"
python -m pytest -q \
  testing/ascend/distributed/test_native_plan.py \
  testing/ascend/distributed/test_native_runtime.py \
  testing/ascend/distributed/test_native_transport.py \
  examples/distributed/test_native_distributed.py
```

需要构建了 Ascend 后端的 TileLang。本机没有 SDK 时，可选 SDK 测试会 skip，必须与已执行测试区分。
准备好 SHMEM 和 CANN 后，编译完整 native 路径但不启动设备：

```bash
for P0_CASE in put get gemm; do
  python -m examples.distributed.run_native_distributed \
    --mode build --case "$P0_CASE" --sdk-profile cann-9.1 \
    --rounds 3 --slots 2 --shmem-root "$SHMEM_ROOT" \
    --shmem-library "$SHMEM_LIBRARY" \
    --output-dir "$P0_NATIVE_OUTPUT/${P0_CASE}_build"
done
```

`lowering.json` 保存两 rank 的计划和源码，`build.json` 保存真实子 kernel 编译产物及哈希，
`validation.json` 保存 Host bridge 身份。`device_executed=false` 是预期值；编译成功不是实机通过。
bridge 使用真实 SDK 头文件构建 C ABI，Python 不手写 SHMEM C++ struct 布局。
头文件内容、SHMEM 库、bridge 源码及工具链身份参与缓存/跨 rank 一致性检查。

已配置真实 SDK/运行库时，还可显式启用 Host bridge 的编译和 DLPack ABI 测试：

```bash
TILELANG_TEST_SHMEM_BRIDGE=1 python -m pytest -q \
  testing/ascend/distributed/test_native_runtime.py
```

该项只检查桥接 ABI，不初始化 NPU。普通无 SDK 测试会明确跳过它；不能把 skip 记录成通过。

## 4. 真实双卡运行

在 TileLang 根目录运行，入口自动创建两 rank 子进程，不需另套 `torchrun`：

```bash
# 先分别验证 put、get，再验证 GEMM；使用同一 Session 连续两次 launch。
for P0_CASE in put get gemm; do
  python -m examples.distributed.run_native_distributed \
    --mode run --case "$P0_CASE" --devices 0 1 \
    --rounds 3 --slots 2 --launches 2 --seed 0 \
    --sdk-profile cann-9.1 --shmem-root "$SHMEM_ROOT" \
    --shmem-library "$SHMEM_LIBRARY" \
    --operation-timeout 60 --timeout 600 \
    --output-dir "$P0_NATIVE_OUTPUT"
done
```

终端打印本次独立结果目录 `p0_native_<case>_*`。默认动态选择两个 loopback 端口；需要固定端口时
同时传 `--bootstrap-port` 和 `--control-port`，二者必须不同且未被占用。
`--operation-timeout` 控制初始化/Host 控制面/stream 同步的等待上限，`--timeout` 控制整个父进程
测试期限（含编译）。一个 rank 失败或超时，监督进程会终止其余 worker；超时不表示设备请求已成功取消。
失败 Session 禁止继续复用或主动释放对端可能仍访问的内存。

三个用例的实际语义：

- `put`：本地准备 Source → `copy_async` put → local wait → fence → release signal → remote wait；
  对端 acquire 后复制到输出，再 ACK。验证分离的 copy/signal、双完成点和接收槽复用。
- `get`：每轮改写有限 Source slot 前等待上代 ACK；发布 SourceReady，对端 acquire 后执行 get，
  本地完成并消费后 ACK。验证源租约、真实拉取以及源槽/接收槽复用。
- `gemm`：复用现有 [AllGather + GEMM 源 DSL](../../examples/distributed/example_allgather_gemm.py)，
  两 rank 发送不同 A tile，分别对本地和远端 A 执行原始 GEMM。输出按 owner 存放，B 为复制权重。
  3 轮时每 rank 每次 launch 执行 6 个 Cube GEMM，两次 launch 合计 24 个。

## 5. 正例与负例验收

正例要求退出码 0，顶层 `manifest.json` 同时满足：

- `passed=true`、`report_complete=true`，两个 rank 退出码均为 0；两 worker `completed=true`、`closed=true`。
- 每次 launch 的 execution `passed/device_executed/kernel_executed` 为 true；完整 phase 记录与计划相符。
- 全部 READY/ACK 等于预期 epoch：默认第 1 次 launch 为 `[3, 2]`，第 2 次为 `[6, 5]`。
- `put/get` 输出与原始 peer BF16 位逐项相同；GEMM 全局 FP64 oracle 的错误数、非有限值均为 0，
  固定 `rtol=1e-5, atol=1e-5`，不因实机结果临时放宽。

参考值来自父进程保存的原始输入 `launch*_inputs.npz`，不会从通信后的接收数据反推。
GEMM 默认两次 launch 共检查 98,304 个 FP32 输出。未写输出先填 NaN，交换输出先填无效 BF16，
避免未完成的写入被零初始化掩盖。

再运行源数据污染负例：

```bash
if python -m examples.distributed.run_native_distributed \
  --mode run --case gemm --devices 0 1 --rounds 3 --slots 2 --launches 2 \
  --fault corrupt_send --sdk-profile cann-9.1 \
  --shmem-root "$SHMEM_ROOT" --shmem-library "$SHMEM_LIBRARY" \
  --output-dir "$P0_NATIVE_OUTPUT"; then
  P0_NEGATIVE_STATUS=0
else
  P0_NEGATIVE_STATUS=$?
fi
test "$P0_NEGATIVE_STATUS" -eq 1
```

它只改 rank 0 第一次 launch 的一个输入元素，保留原始参考输入，rank 1 必须通过真实通信消费
这个被污染的 operand。预期 `passed=false`、`report_complete=true`、`expected_negative_detected=true`，
还必须 `fault_reference_passed=true`：父进程独立构造仅含指定输入污染的参考值，核对两 rank 的全部
输出与之相符且没有 NaN/Inf；后续未污染 launch 必须恢复正确。全 NaN、任意其他输出损坏、缺失
phase 或请求污染却未实际污染都不算成功负例。退出码 2 表示构建、初始化、执行、超时或其他验收失败。
该注入验证远端数据路径及独立 oracle，不宣称模拟了互联链路位翻转。

## 6. 扩展实机矩阵与结果留存

基础用例通过后，按相同命令逐项改变参数：

| 测试 | 参数 | 重点 |
| --- | --- | --- |
| 单槽强制复用 | `--slots 1 --rounds 7 --launches 3` | 每轮 ACK/credit、热槽不读旧值 |
| 不满槽/恰满槽 | `--slots 2 --rounds 1` / `--rounds 2` | 未使用槽保持 0、正确尾部 drain |
| 较长运行 | `--slots 2 --rounds 31 --launches 10` | seed 随 launch 改变，epoch 不重置 |
| 交换物理卡 | `--devices 1 0` | rank 与物理设备编号分离 |
| 新会话 | 重复启动入口 | 正常 close 后重新初始化，无旧映射/信号泄漏 |
| 污染对照 | `--fault corrupt_send` | rank 1 远端输出失败，完整协议仍结束 |

每次保留完整目录，尤其是原始输入、每 rank 输出、`worker.json`、rank 日志、计划、生成/转换后的
源码和编译产物哈希。另记录驱动、固件、CANN、PyTorch/torch_npu、SHMEM、TileLang commit、
物理设备型号及 peer 拓扑；所有原始资料留在保密环境。

检查器可拒绝已展开有限计划中的确定性等待死锁、非法区域、错 epoch 和元信息不一致，
不能静态证明所有跨 rank 协议都没有数据竞争。此版本不输出融合加速比或通信带宽认证。
双卡精度、内存可见性和进度的最终结论，以本节实机结果为准。

## 7. 设备初始化与 bridge 加载排障

若运行在 `torch.npu.set_device()` 阶段失败，例如错误码 `507033` 并伴随 `TsdOpen failed`，
先使用独立入口 [diagnose_native_runtime.py](../../examples/distributed/diagnose_native_runtime.py)
比较普通 PyTorch 单卡计算与 bridge 加载顺序。`507033` 表示设备启动失败；同一堆栈后续出现的
“无有效 context”不足以证明程序漏调 `set_device`，也不能仅凭这条堆栈断言是环境或 bridge 的问题。
此时原生 Session 尚未完成设备绑定，不能据此判断 SHMEM 通信或 GEMM 是否正确。

在发生故障的同一账号、容器、终端和 Python 环境中，从仓库根目录运行：

```bash
python examples/distributed/diagnose_native_runtime.py \
  --devices 0 1 --shmem-root "$SHMEM_ROOT" \
  --shmem-library "$SHMEM_LIBRARY" \
  --timeout 180 --output-dir output/distributed/diagnostics
```

`--devices` 使用当前进程可见的设备索引；可只传一张卡。`--shmem-library` 可省略，库选择规则与
原生入口相同。保持发生故障时的运行库配置，脚本不会修改 `LD_LIBRARY_PATH` 或重置设备。
每个设备的每种模式都会在全新的子进程中执行，各探针串行运行，默认包含以下三组：

| 模式 | 执行顺序 | 用途 |
| --- | --- | --- |
| `plain` | 只导入 `torch`/`torch_npu`，绑定设备，执行微型加法并校验结果 | 检查普通单卡计算能否启动，不导入 TileLang |
| `bridge-first` | 导入 TileLang、构造 `ShmemBridge`，再绑定设备并校验加法 | 复现原生入口在设备绑定前加载 bridge 的顺序 |
| `torch-first` | 先绑定设备并校验加法，再加载 bridge，再校验加法 | 比较设备初始化先于 bridge 加载的情况 |

两种 bridge 模式在加法校验之后还会执行 `bridge_device_identity` 阶段：再次调用
`torch.npu.set_device()`，通过 bridge 查询 SoC 并确认其为 Ascend 950，再查询物理设备映射。
该阶段单独记录，失败不能等同于 bridge 加载失败或加法计算失败。

这些探针不创建分布式 Session，不执行 SHMEM 初始化或双卡通信。TileLang 导入期间可能间接
导入 PyTorch 或 `torch_npu`，因此 `bridge-first` 比较的是设备初始化与 bridge 加载的先后，
不保证 bridge 先于所有 Python 模块导入。报告记录执行阶段和已加载的相关运行库路径。

也可先只跑纯单卡基线；这一模式不要求构建 TileLang 或配置 SHMEM：

```bash
python examples/distributed/diagnose_native_runtime.py \
  --devices 0 1 --modes plain --timeout 180 \
  --output-dir output/distributed/diagnostics_plain
```

`--timeout` 是每个探针的期限，包含导入、bridge 构建和设备操作。某个探针失败后仍继续其余探针；
超时会终止该探针的进程组，避免编译子进程继续运行。只要有探针失败或超时，脚本最终返回非零。
这种进程清理不保证已提交的设备请求被成功取消。

每次运行会在 `--output-dir` 下新建 `run-*` 子目录，不覆盖之前的结果。终端会打印本次目录，其中包括：

- `report.json`：完整报告，含执行阶段、版本、选定环境变量、运行库路径与错误信息。
- `summary.json`：用于人工审查和脱敏的摘要，不含完整路径、环境变量或 traceback。
- 每个探针的 `.log` 和 `.json`：保留独立进程的原始输出与阶段记录，便于本地定位首次失败。

原始报告和日志留在保密环境。需要反馈时，先人工检查 `summary.json`，再按环境要求脱敏；
摘要仍可能包含版本或设备相关信息，不应未经审查直接对外发送。脚本不会自动上传结果。

按同一设备的三组结果判断下一步：

| 结果 | 下一步 |
| --- | --- |
| `plain` 在导入、设备绑定或加法阶段失败 | 先检查对应 Python/`torch_npu`、驱动、固件、CANN、设备可见性及容器配置；以该进程 Ascend 日志中的首次错误为依据 |
| `plain` 通过，bridge 模式在构建或加载阶段失败 | 检查固定 SHMEM revision、头文件、编译器、库文件及其动态链接依赖；此结果还未比较到设备初始化 |
| `plain` 与 `torch-first` 通过，只有 `bridge-first` 在设备绑定或计算阶段失败 | 优先比较各阶段实际加载的运行库及路径，进一步调查加载顺序；这只是定位线索，不是根因证明 |
| `plain` 通过，两种 bridge 模式均在设备绑定或计算阶段失败 | 根据首次失败阶段及库路径继续定位 bridge 与运行环境的相互影响 |
| 加法已通过，失败阶段为 `bridge_device_identity` | 本地查看错误详情，区分再次绑定设备、Ascend 950 型号检查或物理设备映射查询失败；不能归为加法精度失败 |
| 三组均通过 | 单卡基础与两种加载顺序检查通过，再回到第 4 节执行真实双卡测试；这不证明 SHMEM 或双卡 GEMM 已通过 |

不要把 SDK 的 `devlib` 链接桩库或 Simulator 运行库用于真实设备进程。仓库构建选项
`TILELANG_USE_ASCEND_STUBS` 对应的是延迟加载真实 ACL 的转发层，不能仅凭该选项名称认定
正在使用 Simulator；排障时以报告中实际加载的库为准。
