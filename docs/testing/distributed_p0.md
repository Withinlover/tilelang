# P0 分布式 DSL 测试指南

本指南适用于 `feat/distributed-p0-dsl` 分支的当前实现。执行前记录实际 commit；不要用分支名代替可复现版本。接口设计见 [P0 DSL 设计](../design/distributed_p0_dsl.md)，已完成的验证见 [实现记录](../design/distributed_p0_implementation_validation.md)，本指南保留本机协议与 CANN 联仿流程，入口程序是 [双 rank AllGather + GEMM](../../examples/distributed/example_allgather_gemm.py)。**真实双卡的环境、构建、启动和验收命令见新增的 [Ascend 950 原生双卡测试指南](distributed_native.md)。**

## 1. 当前能验证什么

本机联仿 profile `ascend950_cannsim_v1` 由一个 CPU 协调器持有两个逻辑 rank，执行从同一个 TileLang PrimFunc 得到的通信协议；计算可以用 NumPy，也可以由 CANN Simulator 执行从该 PrimFunc 提取并新编译的单卡 GEMM。默认 3 轮、2 个接收槽覆盖槽复用、READY/ACK 和 epoch 递增。

新增的 `ascend950_shmem_mte_phased_v1` 将同一 TD 源程序编译成真实 SHMEM 通信 kernel 和独立 TileLang 计算 kernel，由每 rank 的 Host/stream 顺序执行。该路径不依赖 cannsim-helper；本机已完成源码、原生构建和 Host ABI 检查，真实双卡正确性与可见性仍待用户实机验收。它不是原设计的单 kernel P0-A，也没有实现 P0-B 融合调度。

`ascend950_cannsim_v1` 仍不接受设备绑定或 device stream。普通 `tilelang.compile`、JIT/cache 和 lowering 入口仍拒绝原始含 TD 的函数；原生分段路径通过显式 `D.compile(..., profile=D.NATIVE_PROFILE)` 消解 TD，再让子 kernel 复用既有编译器，不能通过删除 gate 或关闭语义检查启用。

| 层次 | 是否需要 NPU | 是否需要 CANN | 是否需要扩展 helper | 证明范围 |
| --- | --- | --- | --- | --- |
| 前端、共享检查、Host、native gate | 否 | 否 | 否 | DSL/ABI/生命周期，以及原始 TD 不被普通 native 入口误执行 |
| 局部片段提取和 Ascend 源码 lowering | 否 | 否 | 否 | 编译器保留受支持的真实计算语句；需要构建 Ascend 后端 |
| 双 rank 协议模式 | 否 | 否 | 是 | CPU 模型中的协议、数据流、故障及独立全局精度 |
| CANN 计算片段联仿 | 否 | 是，当前固定 9.1 | 是 | 上述协议加真实单卡 Simulator 计算/流水 |
| 原生分段源码、构建与 Host ABI | 否 | 构建时需要 | 否 | SHMEM 通信/计算子 kernel 的生成、真实编译和 ABI；不证明设备执行 |
| 原生双卡验证 | 是 | 是，另需固定 SHMEM | 否 | 按[原生测试指南](distributed_native.md)在两张 950 上验收；本机未执行 |

所有命令在 Bash 中、仓库根目录运行，除非另有说明。输出目录由 `P0_OUTPUT_DIR` 指定；原始输入、输出、IR、生成代码和 trace 都可能包含业务信息，应保留在获准的本地目录。

## 2. 新克隆与构建

需要 Linux、Python 3.10 及以上、C/C++ 编译器、Git 子模块、CMake 3.26 及以上和 Ninja。CANN 联仿当前实测平台为 Linux x86_64。只做本地语义检查时不需要 CUDA、真实 NPU 或 `torch_npu`，但必须构建本分支的 native 库，注册新增 `tl.distributed.*` op。

```bash
git clone --recurse-submodules --branch feat/distributed-p0-dsl \
  git@github.com:Withinlover/tilelang.git
cd tilelang
export P0_REPO="$PWD"
export P0_OUTPUT_DIR="$P0_REPO/output/distributed/testing"
mkdir -p "$P0_OUTPUT_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install pytest
CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_ASCEND=ON -DTILELANG_USE_ASCEND_STUBS=ON" \
  python -m pip install -e . -v --no-build-isolation

git rev-parse HEAD > "$P0_OUTPUT_DIR/tilelang_commit.txt"
git submodule status --recursive > "$P0_OUTPUT_DIR/submodules.txt"
python -c 'import tilelang; print(tilelang.__file__); print(tilelang.__version__)'
```

本地 stub 配置允许没有 CANN 的机器加载 Ascend 编译器；它不提供 NPU runtime。CANN Simulator 在 helper 的独立进程中运行，不把 stub 当作真实设备实现。实机安装需要匹配的驱动、CANN、PyTorch/`torch_npu`、固定 SHMEM 和真实 runtime，参照 [原生测试指南](distributed_native.md)及 [Ascend 后端安装](../../tilelang/ascend/README.md)。

切换 commit、改变 TVM 子模块或修改 C++ 后必须重建；已配置的开发构建可执行 `cmake --build build --parallel`。如果安装器选择了其他构建目录，以配置日志为准。Python 从仓库根目录导入时通常加载当前 checkout 和 `build` 下的 native 库，不能用旧 wheel 的成功导入证明新 C++ 已生效。不要覆盖子模块中已有的未提交改动。

## 3. 无 helper、无设备的测试

以下测试显式覆盖源码前端、manifest、opaque op、event 生命周期、64 字节信号布局、Session/ticket/epoch、原生执行拒绝，不依赖 GPU 自动探测。

```bash
python -m pytest -q \
  testing/python/language/test_tilelang_language_distributed.py \
  testing/python/analysis/test_distributed_checker.py \
  testing/python/runtime/test_tilelang_distributed_session.py \
  testing/python/analysis/test_distributed_native_gate.py \
  --junitxml="$P0_OUTPUT_DIR/frontend_host_gate.xml"

python -m pytest -q \
  testing/python/language/test_tilelang_language_eager_ast_signature.py \
  testing/python/language/test_tilelang_language_dialect.py \
  testing/ascend/distributed/test_segments.py \
  testing/ascend/distributed/test_worker.py \
  --junitxml="$P0_OUTPUT_DIR/compiler_worker.xml"
```

`test_segments.py` 需要已启用的 Ascend lowering/codegen，但不调用 Bisheng 或 CANN。`test_worker.py` 仅在 Linux 上启动可控的假 worker/native 进程，验证无响应时的进程组清理；它不是设备执行测试。

本机和实机都可以运行下面的 native gate smoke：

```bash
python -m pytest -q \
  testing/python/analysis/test_distributed_native_gate.py \
  -k 'native_public_entries_gate_before_target_resolution and ascend'
```

预期四个普通公开入口在后端解析、编译或设备提交前拒绝原始 TD 函数，由测试捕获后显示通过。这里的“通过”表示入口限制生效，**不是原生通信已经执行**。显式原生 profile 使用另一条受检查的分段路径；不要移除 gate 或关闭语义检查。

不建议以整套 `testing/python` 作为无设备机器的准入命令。部分既有测试需要实际 GPU 或固定 CUDA 架构；应将环境不满足、skip 和真正的语义回归分别报告。

## 4. 可选的扩展 cannsim-helper

协议模式和 CANN 联仿依赖外部扩展版本的 cannsim-helper。原来的单卡工具版本或仅安装 Agent skill 都不够。本分支随附 [companion 补丁与安装说明](../../tools/cannsim/README.md)，将所需功能加到固定的 helper 基线；不要求这些扩展已在 helper 远端发布。

从 TileLang 根目录为 helper 创建一个新 checkout，再应用随分支交付的补丁：

```bash
git clone git@gitcode.com:Withinlover/cannsim-helper.git ../cannsim-helper
git -C ../cannsim-helper checkout --detach 2eb1ce5a79a76cc3f3e1c0eb4b04c6e9f6f694cf
python tools/cannsim/apply_companion.py --helper-root ../cannsim-helper --check
python tools/cannsim/apply_companion.py --helper-root ../cannsim-helper
python tools/cannsim/apply_companion.py --helper-root ../cannsim-helper --verify
export CANNSIM_ROOT="$(realpath ../cannsim-helper)"

# 选择本机已安装的 Python 3.11–3.13；默认示例使用 3.12。
export P0_HELPER_PYTHON="${P0_HELPER_PYTHON:-python3.12}"
(
  cd "$CANNSIM_ROOT"
  PYTHON="$P0_HELPER_PYTHON" ./setup_env.sh
)
```

应用脚本只接受精确基线和干净工作区，拒绝覆盖已有改动。已经应用过补丁的目录按 companion README 的检查方式处理，不重复强制应用。`CANNSIM_ROOT` 必须指向这个实际源码根目录，包含 `cosim/protocol.py`、`cosim/segments.py`、`cosim/worker.py` 和匹配的 Host/报告代码；不可混用不同 checkout 中的 `cosim` Python 包。当前 companion 仅提供纯 Cube 片段所需的 `off` 路径，不包含或认证其他 FA/MIX 修复。

记录基线、companion 版本和实际工作区状态：

```bash
: "${CANNSIM_ROOT:?请设置为匹配的扩展 cannsim-helper 源码目录}"
test -f "$CANNSIM_ROOT/cosim/protocol.py"
test -f "$CANNSIM_ROOT/cosim/segments.py"
test -f "$CANNSIM_ROOT/cosim/worker.py"
git -C "$CANNSIM_ROOT" rev-parse HEAD > "$P0_OUTPUT_DIR/helper_commit.txt"
git -C "$CANNSIM_ROOT" status --short > "$P0_OUTPUT_DIR/helper_worktree.txt"
cp tools/cannsim/manifest.json "$P0_OUTPUT_DIR/helper_companion_manifest.json"
sha256sum tools/cannsim/p0_companion.patch > "$P0_OUTPUT_DIR/helper_companion_sha256.txt"
```

应用 companion 后 helper 的 HEAD 仍是基线，仅 commit 不能完整标识实际版本；必须同时保留 companion manifest/补丁 hash。若还有额外本地补丁，另在受控环境内保存补丁或文件 hash。当前 helper 的 CANN worker 使用其自己的 `.venv/bin/python`，支持 Python 3.11–3.13，依赖 NumPy 2.2.6；本仓库解释器可以是 Python 3.10。初始化环境不需要 CANN；不要直接复制另一台机器的虚拟环境。

纯协议模式在 TileLang 解释器中加载 helper 的模型，使用 NumPy 进行本地 GEMM。它不会运行 CANN。

```bash
# 可选：先检查 helper 模型自身的协议测试，不需要 CANN。
(
  cd "$CANNSIM_ROOT"
  .venv/bin/python -m unittest discover -s tests -p test_cosim_protocol.py -v
)

# 再检查本仓 DSL 到该模型的实际接入。
python -m pytest -q \
  testing/ascend/distributed/test_cosim.py \
  examples/distributed/test_allgather_gemm.py \
  --junitxml="$P0_OUTPUT_DIR/protocol.xml"

python examples/distributed/example_allgather_gemm.py \
  --mode protocol --rounds 3 --slots 2 --seed 0 \
  --helper-root "$CANNSIM_ROOT" --output-dir "$P0_OUTPUT_DIR/protocol"
```

预期退出码为 0，顶层 `manifest.json` 中 `passed=true`、`kernel_executed=false`、全局误差元素数为 0。协议测试还覆盖发送/拉取、两次 launch 不清零信号、延迟消费者、early READY、丢 READY、丢 ACK 和 payload 污染。未提供 helper 时部分测试会 skip；必须检查 pytest 摘要，不能把 skip 计成模型验证通过。

## 5. Ascend 950 本机 CANN 联仿

本节只使用一个 Simulator device，仍然没有实际跨卡 DMA。需要完整的 CANN 9.1 Simulator、`Ascend950PR_9589` 模型、Bisheng、支持 CCE 的 `ld.lld`、C++17 Host 编译器，以及已准备好的 helper 环境。当前受限编译 profile 是 `cann-9.1-gemm64`，仅支持 BF16 64×64 输入、FP32 输出的指定 Cube GEMM 片段；其他 SDK 版本不能仅凭目录名替换后宣称兼容。

按安装环境加载 CANN 的设置脚本，令 `PATH`、动态库路径和 Toolkit 变量保持一致。下面假定 `CANN_ROOT` 已由本机环境设置；如果设置脚本不在 Toolkit 根目录，请使用安装说明中的实际脚本。

```bash
: "${CANN_ROOT:?请设置为已安装的 CANN 9.1 Toolkit 根目录}"
source "$CANN_ROOT/set_env.sh"
export ASCEND_TOOLKIT_HOME="$CANN_ROOT"
export ASCEND_HOME_PATH="$CANN_ROOT"
test -x "$CANNSIM_ROOT/.venv/bin/python"

python - <<'PY'
from tilelang.contrib.bisheng import find_bisheng_path, find_ld_lld_path
print("bisheng:", find_bisheng_path())
print("ld.lld:", find_ld_lld_path())
PY

python examples/distributed/example_allgather_gemm.py \
  --mode rank-segments --rounds 3 --slots 2 --seed 0 \
  --helper-root "$CANNSIM_ROOT" --output-dir "$P0_OUTPUT_DIR/positive" \
  --segment-timeout 120
```

正例预期退出码为 0：两 rank 各 3 轮，每轮执行本地和远端 GEMM，共 12 个 CANN 片段。`--consumer-delay` 是协议逻辑 tick，`--segment-timeout` 是单片段 Host 墙钟秒，不能用于比较设备延迟。

随后运行污染负例。使用 `if` 保存预期非零退出码，避免开启 `set -e` 的终端提前终止：

```bash
if python examples/distributed/example_allgather_gemm.py \
  --mode rank-segments --rounds 1 --slots 2 --seed 0 \
  --fault corrupt_payload --helper-root "$CANNSIM_ROOT" \
  --output-dir "$P0_OUTPUT_DIR/negative" --segment-timeout 120; then
  P0_NEGATIVE_STATUS=0
else
  P0_NEGATIVE_STATUS=$?
fi
test "$P0_NEGATIVE_STATUS" -eq 1
```

负例应执行 4 个局部 GEMM，并由独立的全局 oracle 拒绝结果。**非零退出码本身不是负例通过**：编译失败、丢失报告或超时也可能非零，必须核对下一节的故障、完整性和精度字段。

## 6. 报告验收

每次命令生成独立的 `p0_protocol_*` 或 `p0_rank_segments_*` 子目录；终端结果中的 `run_dir` 是本次目录。保留整目录，不只保留终端的 PASS/FAIL。

| 验收项 | 协议正例 | CANN 正例，3 轮 | CANN 污染负例，1 轮 |
| --- | --- | --- | --- |
| 顶层 `passed` | true | true | false，符合预期 |
| `kernel_executed` | false | true | true |
| `report_complete` | true | true | true |
| `protocol.passed` | true | true | true，污染不破坏握手 |
| `protocol.fault_injected` | false | false | true |
| CANN `segments_completed/expected` | 0/0 | 12/12 | 4/4 |
| 每个局部片段 `passed` | 不适用 | 全部 true | 全部 true |
| 全局输出元素 | 49,152 | 49,152 | 16,384 |
| 全局错误元素 / 非有限值 | 0 / 0 | 0 / 0 | 固定 seed=0 时 64 / 0 |
| 完整/去 SET-WAIT 流水 | 不适用 | 共 24 份 | 共 8 份 |
| 未完成 event / open lease | 均空 | 均空 | 均空 |

全局精度容差为 `rtol=1e-5, atol=1e-5`，参考只使用持久化的原始 BF16 全局输入，按 FP64 计算。正例实测最大绝对误差约 `2.29e-7`；这是已观察结果，不是要求所有工具链都逐位相同。负例只污染接收的一个 BF16 元素，局部 GEMM 相对实际收到的输入仍正确，而全局 oracle 检出一行 64 个错误。

重点保留这些证据：

- `manifest.json`、`accuracy.json`、`report.md`：顶层合取结果、形状、故障、片段覆盖、误差。
- `program.json`、`program.tir`、`abi.json`：源 DSL 的规范化 TIR、程序身份、CPU descriptor 契约。
- `protocol_trace.json`、`oracle_inputs.npz`、`outputs.npy`：协议事件、未受通信污染的原始输入、完整输出。
- `compiled_*/`：提取后的 IR、生成源码、SDK 转换后源码和 diff、编译命令/日志、新编译 ELF 及 SHA256。
- 各片段实际输入/输出、manifest、`pipeline.json` 和两份 trace，以及顶层 `trace_index.json`：核对输入来自正确的 rank/round/owner，AIC Core 0 有 MMAD/END 和 detail。

不能用局部 GEMM 精度代替全局精度，也不能用 trace 文件存在代替 END/完整性检查。各 CANN trace 有独立时间原点，协议 tick 没有物理时间单位；当前不能拼成真实双卡时间线，也不能据此报告通信带宽或融合加速比。

## 7. 环境记录与保密环境离线复现

测试前后记录同一份环境清单。至少包括以下内容；仅记录与测试有关的字段，不需要导出完整环境变量或凭据。

| 类别 | 必须记录 |
| --- | --- |
| 源码 | TileLang commit、子模块 SHA、是否有未提交改动；helper commit/补丁/hash |
| 构建 | Python/NumPy/PyTorch 版本，构建 flags，native 库来源；CMake/编译器版本 |
| 仿真 | CANN Toolkit、Bisheng/linker 版本，Simulator/model 版本或 hash、编译 profile、SOC/arch |
| 用例 | mode、profile、rounds、slots、seed、故障、consumer delay、timeout、容差 |
| 结果 | 退出码、所有验收字段、日志、IR/ELF/输入输出 hash、trace 完整性 |
| 实机附加项 | 驱动/固件、设备具体型号、逻辑 rank 到物理卡映射、peer 拓扑、transport/ABI 版本 |

没有网络的保密环境应先通过获准渠道准备源码与全部子模块、匹配的 helper、适合目标平台的依赖 wheelhouse、CANN 安装介质和版本清单。不要把另一台机器的 `.venv` 或未经确认的编译对象当作可复现环境。安装后重新构建并运行第 3 节，再逐层执行协议和联仿。

离线 pip 安装可通过 `PIP_NO_INDEX=1` 与 `PIP_FIND_LINKS="$P0_WHEELHOUSE"` 使用已准备的 wheelhouse；缺少依赖应作为环境准备失败报告，不自动联网补齐。生成的 IR、输入/输出、地址信息、日志和流水全部留在保密环境内，按当地规则审阅后再决定是否导出脱敏摘要。测试命令不包含上传、遥测或推送产物的步骤。

## 8. 原生双卡入口与后续验证矩阵

原生分段 profile 的环境准备、固定 SHMEM 版本、源码/编译检查、真实启动器和正反例验收，见 [Ascend 950 原生双卡测试指南](distributed_native.md)。它已经实现设备通信子 kernel、对称内存与 peer 映射、Host 多进程一致性预检、独立 stream、超时监督及正常关闭协议；不能把这些本机编译结果写成实机通过。

下表保留完整原生能力的验收矩阵。其中 put/get、slot 复用、多次 launch 和分段 GEMM 已有测试入口；热点缓存压力、更广 dtype/shape、单 kernel P0-A、P0-B 融合和性能结论仍需对应扩展或实机证据。每台机器仍须核对具体 950 硬件/软件/拓扑元组，确认 peer 路径和完成/可见性契约匹配。

满足各层先决条件后，按顺序验证；上一层未通过时不将下一层结果作为认证。

| 层次 | 测试与变化 | 验收指标 |
| --- | --- | --- |
| 平台和单卡基线 | 检查两张卡、peer 路径、SDK/driver；每张卡单独运行普通 TileLang GEMM | 设备映射和版本记录完整；全量数值正确；无错卡或隐式 CPU fallback |
| 真实传输最小实验 | 两个方向的 put/get；local/remote wait；payload 到达与 READY 分开观察；热点缓存 | 仅在契约规定的完成点复用/消费；零撕裂、旧数据、错代；生成代码保留发布/获取动作 |
| 信号和 slot | READY/ACK，S=1/2；N=0、N<S、N=S、N>S；两侧交替延迟、不同合法消息大小 | 零覆写未消费 slot、零提前 ACK；尾部 drain；N=0 提供明确空工作路径后方可执行 |
| 多次 launch 与 ABI | 不清零信号连续运行；新 Session/进程；epoch 接近上限；参数/拓扑不一致 | epoch 单调且提交前拒绝回绕；无旧地址；不一致参数在任何 rank 提交前拒绝 |
| 故障和退出 | 丢通知、对端停止、超时、注册/编译失败；重复 create/close | 有界失败、保留 rank/epoch/等待上下文；无后台设备提交、悬挂进程或资源泄漏 |
| 端到端通信 + GEMM，分段 profile | 由 Host 编排 SHMEM 通信 kernel 与独立本地 GEMM；原始全局输入 oracle、多个 seed、实际消费远端数据、通信污染负例 | 所有输出和 dtype 容差通过，零漏写/非有限值；故障对照被全局校验拒绝；明确此层不是含通信的融合 kernel |
| P0-B 融合 | AIV 通信→Cube acquire、Cube 读取完成→AIV ACK；Stage/流水、UB/L1 压力与多缓冲 | 调度/资源检查通过；零提前覆盖/死锁；融合 kernel 的所有 rank 输出正确 |
| 压力与性能 | 在已认证拓扑内扩大迭代/消息规模；与 Host 集合通信编排基线对比 | 记录 seed/次数/最长等待，正确性零异常；分别报告最慢 rank、端到端时间、初始化成本、峰值资源 |

实机数值容差应按 kernel 的 dtype/算法提前冻结，不能为通过结果临时放宽。压力轮数、超时和资源阈值按已认证平台确定并写入用例配置，不承诺一个通用固定值。性能基线需使用相同输入/输出和精度条件；没有性能提升可以报告正确性成立，但不能声称融合优化成功。真实多卡内存可见性、进度和 P0-B 调度必须有对应实机证据，当前 CPU 协议模型与单卡 Simulator 结果不能替代。
