# Ascend 分布式 JIT：共享对称堆与算子调用

`tilelang.distributed.jit` 将原生执行包装成可调用算子，默认仍选择 phased profile。每个 rank
初始化一次 Session，Session 建立固定容量的 SHMEM 对称堆；每个绑定的算子特化
从堆中取得自己的输入、输出、scratch 和信号空间。后续调用复用这些空间。

该入口复用普通 TileLang 子 kernel 的编译和 `tvm_ffi` 执行路径。
本页默认示例的通信和计算在同一 stream 上分 kernel 顺序执行。
新增单 kernel 路径可显式设置 `D.jit(..., profile=D.P0A_PROFILE)` 或
`D.P0B_PROFILE`，并绑定相同 profile 的 Session；输入/输出与生命周期约定不变。
当前仍限于同机两张 Ascend 950。环境准备见[原生双卡测试指南](distributed_native.md)，
新路径的受限算子范围、SDK 编译与真实设备验收见[融合路径验收](distributed_fused.md)。

## 调用接口

已有返回 TD PrimFunc 的 Python 工厂函数可以直接用 `D.jit` 包装：

```python
import tilelang.distributed as D
from examples.distributed.example_allgather_gemm import allgather_gemm


@D.jit(
    inputs=("Send", "B"),
    outputs=("Output",),
    epochs="rounds",
    profile=D.NATIVE_PROFILE,
    compile_profile="cann-9.1-gemm64",
    transport_compile_profile="cann-9.1-shmem",
)
def gemm(rounds=3, slots=2):
    return allgather_gemm(rounds=rounds, slots=slots)


# 每个 rank 运行相同代码；RANK / LOCAL_RANK 可由 torchrun 设置。
with D.init(
    profile=D.NATIVE_PROFILE,
    bootstrap="tcp://127.0.0.1:29610",
    control_port=29611,
    shmem_root="/path/to/pinned/shmem",
    heap_bytes=64 * 1024 * 1024,
) as session:
    op_a = gemm(rounds=3, slots=2).bind(session)
    op_b = gemm(rounds=1, slots=2).bind(session)
    y1 = op_a(a1, b)
    y2 = op_b(a2, b)
    y3 = op_a(a3, b)
```

代码中的 `a1`、`a3` 是本 rank 的 `(3, 64, 64)` 输入，`a2` 是
`(1, 64, 64)` 输入，`b` 是 `(64, 64)` 输入。BF16 输入用 NumPy `uint16`
数组保存原始 BF16 位模式；直接把浮点数转成 `uint16` 并不是 BF16 编码。
示例中的 `make_inputs` 展示了这种表示。返回值为独立的 NumPy `float32`
数组，形状分别是 `(3, 2, 64, 64)` 和 `(1, 2, 64, 64)`。

`inputs` 和 `outputs` 使用 TD buffer 的名字，明确指定调用参数及返回顺序；
`remote_access` 仍然只描述远程访问权限。`epochs="rounds"` 从工厂参数中
取得本次调用消耗的 epoch 数，也可以直接指定正整数。单个输出直接返回数组，
多个输出按声明顺序返回 tuple，无输出返回 `None`。输入输出名称可以重叠，表示
显式的 in/out buffer；额外的运行时整数参数通过 `op(..., scalars={"n": 1})`
传入，`epoch0` 由框架管理。其余未暴露的 buffer 和信号由绑定自动管理。
当前输入输出仅支持 NumPy，经主机和设备间拷贝；不接收普通 NPU Tensor，
也不提供 D2D 或零拷贝输出。

调用工厂得到程序特化与元数据，`.bind(session)` 完成 collective 绑定和资源
分配。首次执行时，通信及计算子 kernel 通过现有 `tilelang.compile` /
`tvm_ffi` 路径编译或读取缓存；重复执行复用已准备的子 kernel。
编译缓存不保存 Session 内的设备指针、stream 和 ticket。
SHMEM 源码根目录从 Session 传入绑定，不必在每个 JIT 工厂上重复指定。
相同工厂的默认参数、位置参数和同值关键字参数共享特化；不同静态参数产生不同
特化。程序源码可通过 `kernel.prim_func` 只读查看。相同执行契约重复绑定同一
Session 会复用 workspace；需要同一程序的独立状态时，可显式使用双方一致的
`.bind(session, key="another-instance")`。

### 固定构建输入的融合执行

P0-A/P0-B 可显式设置 `D.jit(..., execution_policy="artifact")`；直接调用
`D.compile` 使用同名参数。默认 `strict` 保留每次执行对源码及设备构建文件的完整
检查，phased profile 当前只支持 `strict`。

`artifact` 在首次构建时完整验证并固定源码快照、ABI、编译配置和已加载的两个 rank
kernel。后续调用复用这些已编译对象，不再读取磁盘头文件来重新确认已完成的构建。
这意味着运行期间编辑 SDK/helper 文件不会替换现有产物；需要采用新文件时，应重新
编译并创建相应绑定。显式 `.build()`、新运行时 scalar 专化或被驱逐后重建仍做完整
文件内容检查，检测到与原构建约定不符时要求重建程序。

每次调用仍验证源码快照身份、manifest、编译配置、Session 所有权、实际 scalar、
epoch 区间、信号进度及跨 rank 一致性。普通只读 TIR 使用经过证明的身份检查；
包含可变或未知 metadata 的源码继续逐次比较完整序列化内容。未证明不可变的节点
不会仅凭 root 地址跳过检查。构建模式记录在 manifest 和运行报告，两个 rank 必须
一致；它不改变 READY/ACK、DMA 完成、退出 drain 或同步要求。

性能验收命令、两种模式对照及首次构建与稳态口径见
[融合路径验收指南](distributed_fused.md#性能口径)。

## 对称堆与生命周期

- 对称堆在进程绑定设备后、初始化 Session 时建立，不需要在启动 Python 前申请。
  默认容量为 **64 MiB / rank**，可用 `heap_bytes` 显式调整；两 rank 必须一致。
- 不同算子或 shape 的绑定持有独立 workspace，同一个已绑定算子的连续调用复用
  自己的空间。第一版不进行跨算子的内存池复用，不自动驱逐旧特化，也不自动扩容；
  绑定越多，累计占用越大。容量不足时需重建具有更大堆的 Session。
- Ready/Ack 由绑定持有，epoch 在 Session 中持续递增。重复调用无需手工清零信号、
  申请 ticket 或推进 epoch。
- 算子返回前完成必要的运行确认和输出回读。返回的 NumPy 数组不引用对称堆，
  后续调用不会覆盖 `y1`，Session 正常关闭后也可继续使用这些数组。
- 所有 rank 必须以相同顺序创建绑定和调用算子，使用一致的特化参数及编译配置。
  例如双方都执行 A → B → A；不能让 rank 0 调 A 时 rank 1 调 B。
  本地输入的内容可以不同。当前只支持串行调用。
- `with D.init(...)` 正常退出时协调关闭 Session。异常退出将 Session 标记失败，
  保留可能仍被远端访问的资源；失败后的 Session 不再用于调用或回收复用，需重启
  worker 进程。每个进程同一时间只允许一个原生 SHMEM Session。

## 双 rank 示例

[example_jit.py](../../examples/distributed/example_jit.py) 演示同一 Session
内 A（3 轮）→ B（1 轮）→ A（3 轮），第三次使用不同输入，并检查第一次返回的
结果在后续调用后保持不变。两个特化都使用 2 个通信槽。

先按[原生指南](distributed_native.md) 配置与构建经过固定版本检查的 CANN / SHMEM，
构建 TileLang，并设置 `SHMEM_ROOT` 与 `SHMEM_LIBRARY`。从仓库根目录运行：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module examples.distributed.example_jit \
  --sdk-profile cann-9.1 \
  --shmem-root "$SHMEM_ROOT" --shmem-library "$SHMEM_LIBRARY" \
  --bootstrap-port 29610 --control-port 29611 --heap-mib 64
```

launcher 仅启动两个独立 worker 并设置环境变量，不调用
`torch.distributed.init_process_group`。设备默认取 `LOCAL_RANK` 对应的进程可见
NPU 索引。SHMEM bootstrap 和元数据控制通道使用两个不同且空闲的端口，
均不要与 launcher 的 rendezvous 端口复用。

`--sdk-profile cann-9.1` 选择现有受限的 GEMM64 和 SHMEM 兼容转换；
`--sdk-profile native` 使用原始生成代码，应先确认目标 SDK 的编译检查通过。
这些选项不扩大算子支持范围。

日常 JIT 调用默认不导出逐次诊断文件。需要排查时可以添加
`--output-dir output/distributed/jit`，示例会给 `D.jit` 传入 `debug=True`
和每 rank 独立的输出目录。

每个 worker 使用不可变原始输入在 CPU 上进行 FP64 matmul，独立校验本 rank 的
完整输出，并打印包含 `rank`、`local_checks_passed`、三次调用误差和
`first_output_preserved` 的 JSON。任一精度或输出生命周期检查失败都会抛出异常，
让 launcher 失败退出。需要同时检查两个 rank 的成功记录和 launcher 的退出码。
此示例不收集全局精度报告，也不进行性能认证；需要污染负例和完整报告时仍使用
[run_native_distributed.py](../../examples/distributed/run_native_distributed.py)。

此处提供的是实机验收入口，文档和无设备测试通过不代表双卡运行已验证。

无设备的 JIT 和资源生命周期回归需要已用 `USE_ASCEND=ON` 构建 TileLang，
但不需要可用 NPU；可运行：

```bash
python -m pytest -q testing/ascend/distributed/test_native_jit.py \
  testing/ascend/distributed/test_native_runtime.py \
  testing/ascend/distributed/test_native_plan.py
```

其中双 rank 测试使用 CPU 模拟设备边界，保留真实的程序规划、参数绑定及控制面
collective；它们不代替真实 SHMEM 传输与内存可见性验证。

## 一条命令服务器验收

在已经用 `USE_ASCEND=ON` 构建 TileLang、安装 pytest / torch 并配置 CANN / SHMEM 的 Linux 双卡服务器上，
运行 [run_jit_tests.sh](../../examples/distributed/run_jit_tests.sh)：

```bash
SHMEM_ROOT=/path/to/pinned/shmem \
SHMEM_LIBRARY=/path/to/libshmem.so \
PYTHON_BIN=python3 OUTPUT_DIR=output/distributed/jit-validation \
  ./examples/distributed/run_jit_tests.sh \
    --sdk-profile cann-9.1 --bootstrap-port 29610 --control-port 29611 --heap-mib 64
```

`SHMEM_ROOT` 必需；已导出的 `SHMEM_LIBRARY` 会自动继承，也可省略并使用运行时默认查找。
`PYTHON_BIN` 默认为 `python3`，可以指定已配置环境中的 Python 可执行文件。
脚本复用现有依赖，不自动安装或重建。参数原样传入 `example_jit.py`，但输出目录由脚本
统一管理，需用 `OUTPUT_DIR` 指定根目录；每次运行会创建一个唯一子目录，避免覆盖旧结果。
只查看帮助可运行 `./examples/distributed/run_jit_tests.sh --help`，不会初始化设备。

脚本先运行 JIT、native runtime 和 native plan 三组无设备回归，通过后启动两个真实
NPU worker 执行 A → B → A。两个阶段各自受 `TIMEOUT_SECONDS` 限制，默认 900 秒；
需要 GNU `timeout`，首次编译较慢时可以提高该值。CPU 回归失败时不会启动设备阶段。
错误码保持原值，超时通常为 124，强制终止时可能为 137；日志管道不会吞掉失败。

输出目录保留 `git_commit.txt`、`git_submodules.txt`、`git_status.txt`、`commands.txt`、
`cpu-tests.log`、`torchrun.log`、`EXIT_CODE` 和 `summary.txt`。成功的设备 worker
在 `device/rank0/result.json` 与 `device/rank1/result.json` 保存本 rank 的精度、
旧输出保持和 Session 关闭检查。异常仍保留在 `torchrun.log` 中并导致非零退出；
缺少任一 rank 的结果文件也会令脚本失败。请连同整个目录回收验证证据；两个 rank 的
独立报告不会被标成全局精度报告，也不证明性能或通信计算重叠能力。
