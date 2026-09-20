# 服务器验证：P0/P1 第 01 轮

本轮假设：固定来源的 MEGA 可以在 910B 编译、加载并按原版 slot ABI 正确更新状态。
同时采集原生模型连续两次 spec 调用，为下一轮物理 ABI 改造提供证据。
本地没有 NPU；本文所有设备步骤当前为 `pending_remote`。

## 1. 固定候选与依赖

在独立实验 checkout/容器中执行，使用已有个人 fork remote。分支发布后才能从服务器 fetch；未发布时先由本地发布候选。
不要在正在提供服务的 editable 目录切换提交。以下为 Linux Bash：

```bash
set -euo pipefail
test -z "$(git status --porcelain)"
git fetch origin codex/mega-gdn-910b
candidate=$(git rev-parse FETCH_HEAD)
git switch --detach "$candidate"
git submodule sync
git submodule update --init -- refs/vllm csrc/third_party/catlass csrc/third_party/pto-isa
git submodule status --recursive
test "$(git -C refs/vllm rev-parse HEAD)" = "$(cat .github/vllm-main-verified.commit)"
test "$(git -C csrc/third_party/pto-isa rev-parse HEAD)" = 781ac0eb2e4d31e0bc2ab953bca9f9036bec7528
git submodule foreach 'test -z "$(git status --porcelain)"'
round_dir=$(mktemp -d /tmp/mega-p0p1-r01.XXXXXX)
printf '%s\n' "$candidate" > "$round_dir/candidate.txt"
git submodule status --recursive > "$round_dir/submodules.txt"
npu-smi info > "$round_dir/npu-smi.txt" 2>&1
python --version > "$round_dir/python.txt" 2>&1
python -m pip install -r requirements-lint.txt
bash format.sh ci 2>&1 | tee "$round_dir/format-ci.log"
test -z "$(git status --porcelain --untracked-files=no)"
```

`refs/xllm` 和 `refs/xllm-ops` 只供源码参考，构建不读取它们。需要浏览时显式初始化，避免全树 recursive 拉取引擎依赖。
服务器是 Linux，PTO 可完整检出；本地 Windows 为避免 PTO 文档大小写同名冲突，仅对该子仓 sparse-checkout `include/`。

## 2. 构建并记录真实加载产物

先 source 该服务器已经安装的 CANN 环境脚本，记录镜像 digest、驱动、固件、CANN toolkit/compiler 版本。
`SOC_VERSION` 填实际 910B 型号（例如 ascend910b1，但不要直接照抄未知型号）。
使用与此快照兼容的 Python 3.12 环境，首次安装可按本仓 source 安装文档配置包索引。

```bash
: "${SOC_VERSION:?Set SOC_VERSION to this server's actual 910B SoC}"
case "$SOC_VERSION" in ascend910b*) ;; *) echo 'This round requires 910B'; exit 1 ;; esac
VLLM_TARGET_DEVICE=empty python -m pip install -v -e refs/vllm \
  --extra-index-url https://download.pytorch.org/whl/cpu/ \
  2>&1 | tee "$round_dir/install-vllm.log"
COMPILE_CUSTOM_KERNELS=1 python -m pip install -v -e . \
  --extra-index-url https://download.pytorch.org/whl/cpu/ \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  2>&1 | tee "$round_dir/build-ascend.log"
python -m pip freeze > "$round_dir/pip-freeze.txt"
find vllm_ascend -type f \( -name '*.so' -o -name '*.o' -o -name '*.json' \) \
  -path '*_cann_ops_custom*' -print > "$round_dir/cann-artifacts.txt"
find vllm_ascend -type f -name '*.so' -exec sha256sum {} + > "$round_dir/shared-libraries.sha256"
```

安装默认使用构建隔离；只有完整预装 `pyproject.toml` 构建依赖时才增加 `--no-build-isolation`。
不要将 `COMPILE_CUSTOM_KERNELS=0` 的成功安装计为本轮构建成功。
保留完整日志中 `ascend_mega_gdn_mtp_decode` 的编译/打包信息及生成的 `aclnnAscendMegaGdnMtpDecode` 产物。
旧 extension 的相同路径不代表新构建，须将 candidate、构建日志、文件 hash 和加载信息一起回传。

## 3. 独立 slot ABI 验证

```bash
python -m tools.mega_gdn.validate \
  --expected-commit "$candidate" --output "$round_dir/smoke" \
  2>&1 | tee "$round_dir/smoke.log"
# smoke 成功后再跑 S=2..17；失败立即保留本轮产物。
python -m tools.mega_gdn.validate --matrix \
  --expected-commit "$candidate" --output "$round_dir/matrix" \
  2>&1 | tee "$round_dir/matrix.log"
```

smoke 共 16 个 case：S=2/3、B=2、NK=2/NV=4、两种矩阵布局、原位/共享只读源、accepted=1/S。
matrix 共 128 个 case，覆盖 S=2..17。这里没有覆盖所有 head 几何和 B/N 边界，不能扩写成完整支持矩阵。
检查 Conv、所有 SSM checkpoint、norm 输出、未触及块、状态存储地址、Meta 形状、mutation schema、非连续状态拒绝。
NPU 不可用、非 910B、加载失败、checkout 不匹配或数值失败都返回非零，不把 skip 当 PASS。

`results.json` 记录运行 SHA、依赖、实际 extension 路径/hash、OPP 路径、shape/stride、误差和结果。
`last-inputs.pt`、`last-case.json` 在 MEGA launch 前写入；数值失败增加 `failure.pt`。
设备异常后立即停止本进程；不要在已经写过的 state 上跑 fallback。
这些测试显式同步和 CPU dump，不用于性能测量。

## 4. 原生模型 P0 采集

使用服务器已跑通的模型/版本和 speculative_config，准备 `engine-args.json` 与 `prompts.json`。
engine args 传给 `vllm.LLM`：本轮要求 `enforce_eager=true`、TP=PP=1、`async_scheduling=false`、`enable_prefix_caching=false`，无 KV connector；
保留模型支持的真实投机配置。不要凭示例猜测 method 名称。prompts 是短字符串列表。
`--layer` 必须是实际 GDN 层 prefix；不同模型层号/命名可能不同。

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m tools.mega_gdn.capture_native \
  --engine-args /path/to/engine-args.json --prompts /path/to/prompts.json \
  --layer model.layers.0.linear_attn --calls 2 --max-tokens 32 \
  --output "$round_dir/native" 2>&1 | tee "$round_dir/native.log"
```

仅此命令的进程会临时包装被选层的纯 spec 调用；生产源码没有默认日志、自动 dump 或新环境变量。
采集 qkv/a/b/z、权重、accepted、query 长度、物理表、实际 Conv/SSM stride、被引用块的前后值、core 与 norm 输出。
负 padding 保留在表中，但不用于 gather。少于所需完整 spec 调用时脚本失败，不写成 baseline PASS。
数据搬移会改变调度时序，结果只用于状态和数值分析。此轮脚本不支持多进程/TP/PP 采集。

## 5. 回传与下一步

回传整个 `round_dir`（包括失败日志）、模型及 revision、真实 SoC、镜像 digest 和 CANN/驱动/编译器信息。
prompt 和状态数据可能包含业务输入，采集使用固定测试文本即可。

- 编译失败：优先解决 PTO/CANN/注册/链接的第一个错误。
- slot ABI 数值失败：按 Conv → SSM checkpoint → norm 查首个分歧，不改变容差掩盖差异。
- P0/P1 通过：从物理表、state stride、接纳后整理证据实施 P2，随后对齐 FP32 gate 和 Q/K 舍入点。
- P2 状态与数值通过后才增加 P3 模型开关与 core+norm 路由；graph、prefix COW、async 各自需要额外证据。

当前不是生产启用版，也没有性能结论。
