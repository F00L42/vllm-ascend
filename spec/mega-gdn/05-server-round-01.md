# 服务器 Agent 编译与验证 Runbook：MEGA P0/P1 第 01 轮

本文是服务器执行入口。目标是验证个人分支的 **910B 独立算子**，并在具备模型配置时采集原生 GDN 状态。
当前模型没有调用 MEGA；smoke/matrix 通过也不能写成“模型迁移完成”或“已有加速”。

## 0. 执行规则与成功标准

按 1 → 7 顺序执行。第 8 步需要额外模型资料；第 9 步无论成功或失败都执行。
每步只有满足写明的成功条件才能继续。一次只运行一个命令块，检查退出码和输出，不要一次粘贴整篇。

- 在 Linux、已分配的 Ascend 910B 开发容器/服务器内运行 Bash；不要在 Windows 运行本 runbook。
- 只操作本文新建的 checkout/venv。保留根目录 `AGENTS.md`，不要在生产服务的 editable 目录构建。
- 不 rebase、不更新子仓 main、不执行 `submodule update --remote`，不改变任何源码、容差或测试数量。
- 不通过关闭 custom kernels、跳过失败用例、换系统包、复制旧 `.so` 或扩大容差获得 PASS。
- 失败后停止后续编译/测试，跳到第 9 步。只读检查允许继续；修复源码、变更版本或重跑需进入明确的新一轮。
- 不自动重启 NPU、杀其他服务、安装驱动/CANN、使用 sudo；缺这些条件时报告 `BLOCKED_ENV`。
- 本文 shell 变量是当前实验参数，不是新增的产品环境变量。不要修改 `vllm_ascend/envs.py`。

| 阶段 | 允许填写的结果 |
| --- | --- |
| P1 构建/加载 | 完整构建退出 0，加载路径与本 checkout 一致，指定 ACLNN 符号存在 |
| P1 smoke | 进程退出 0，`results.json.status=passed`，恰好 16 个 case 全部 passed |
| P1 matrix | 进程退出 0，恰好 128 个 case 全部 passed，commit 与 smoke 相同 |
| P0 原生采集 | `captured_pending_analysis` 且有两份完整调用 dump；采集成功不代表数值已分析 |
| 未运行/条件不足 | `NOT_RUN` / `BLOCKED_ENV` / `BLOCKED_MODEL_INPUT`，不得写 PASS |

可直接给服务器 Agent 的任务：

```text
阅读并严格执行 spec/mega-gdn/05-server-round-01.md。
从 F00L42/vllm-ascend 的 codex/mega-gdn-910b 固定一次完整 SHA，
在新 checkout 与新 venv 中编译并依次完成加载、16-case smoke、128-case matrix。
没有已确认模型配置时，完成独立算子验证，将 P0 标为 BLOCKED_MODEL_INPUT。
不修改代码、依赖 pin、容差，不绕过失败。任何失败停止后续步骤，收集完整日志和结果。
最后返回候选 SHA、每阶段状态、首个错误、证据包绝对路径和 SHA-256；不得宣称模型迁移完成。
```

## 1. 新建实验目录与可恢复会话

先检查 `uname -s` 为 Linux，`git`、`python3.12`、`npu-smi`、`gcc`、`g++`、`tar`、`sha256sum` 可用。
确认被分配的设备是 910B 且允许做实验。不要仅凭机器目录名判断硬件。
`python3.12` 缺失时可以使用**已确认的 Python 3.12 绝对路径**替换本节中的该命令；不要自行更换 Python 大版本。

```bash
set -euo pipefail
test "$(uname -s)" = Linux
for command_name in git python3.12 npu-smi gcc g++ tar sha256sum; do
    command -v "$command_name"
done
npu-smi info
python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12); print(sys.version)'
mkdir -p "$HOME/mega-gdn-runs"
run_dir=$(mktemp -d "$HOME/mega-gdn-runs/round-01.XXXXXX")
repo_dir="$run_dir/src"
mkdir "$run_dir/logs" "$run_dir/results"
{
    printf 'run_dir=%q\n' "$run_dir"
    printf 'repo_dir=%q\n' "$repo_dir"
    cat <<'SESSION'
set -euo pipefail
run_logged() {
    local name=$1
    shift
    local rc=0
    local -a statuses
    printf '%s\n' "$name" > "$run_dir/current-stage.txt"
    if "$@" 2>&1 | tee "$run_dir/logs/$name.log"; then
        rc=0
    else
        statuses=("${PIPESTATUS[@]}")
        rc=${statuses[0]}
        if (( rc == 0 )); then rc=${statuses[1]}; fi
    fi
    printf '%s\n' "$rc" > "$run_dir/results/$name.exit"
    if (( rc != 0 )); then
        printf 'STOP: stage=%s exit=%s; collect evidence with step 9.\n' "$name" "$rc" |
            tee "$run_dir/FAILED.txt"
        return "$rc"
    fi
}
require_done() {
    if [[ ! -f "$run_dir/DONE-$1" ]]; then
        printf 'STOP: step %s has not completed.\n' "$1" >&2
        return 1
    fi
}
if [[ -f "$run_dir/runtime.sh" ]]; then source "$run_dir/runtime.sh"; fi
if [[ -d "$repo_dir" ]]; then cd "$repo_dir"; fi
if [[ -f "$run_dir/candidate.txt" ]]; then candidate=$(cat "$run_dir/candidate.txt"); fi
SESSION
} > "$run_dir/session.sh"
printf 'SESSION_FILE=%s/session.sh\n' "$run_dir"
printf 'RUN_DIR=%s\n' "$run_dir"
```

**成功条件：** 输出唯一 `SESSION_FILE` 和 `RUN_DIR`。把两个绝对路径记入任务记录。
后续所有 `/ABSOLUTE_RUN_DIR` 替换为这里输出的真实路径，不能照抄占位符。
每个命令块都先 source session；即使 Agent 的终端工具每次启动新 shell，也能恢复 cwd、venv、CANN 和 candidate。
不要反复执行本节创建多套目录。第一次预检就失败时，记录命令、退出码和完整错误；尚无证据目录可打包。

## 2. 拉取个人分支并锁定所有必需提交

下面只 clone 新目录。若 HTTPS 认证/网络失败，保留错误并报告；不要替换成另一个仓库或绕过证书验证。

```bash
source /ABSOLUTE_RUN_DIR/session.sh
expected_candidate=''  # 若交接指定了 40 位 SHA，填入此处；否则只在本步解析一次分支。
run_logged 02-clone git clone --no-checkout --single-branch \
    --branch codex/mega-gdn-910b https://github.com/F00L42/vllm-ascend.git "$repo_dir"
cd "$repo_dir"
candidate=$(git rev-parse refs/remotes/origin/codex/mega-gdn-910b)
printf '%s\n' "$candidate" > "$run_dir/candidate.txt"
if [[ -n "$expected_candidate" ]]; then
    [[ "$expected_candidate" =~ ^[0-9a-f]{40}$ ]]
    test "$candidate" = "$expected_candidate"
fi
run_logged 02-checkout git switch --detach "$candidate"
run_logged 02-submodules git submodule update --init -- \
    refs/vllm csrc/third_party/catlass csrc/third_party/pto-isa
run_logged 02-pins python3.12 - "$candidate" <<'PY'
import pathlib, subprocess, sys

def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()

assert git('rev-parse', 'HEAD') == sys.argv[1]
assert git('status', '--porcelain') == '', 'checkout must be clean'
for path in ('refs/vllm', 'csrc/third_party/catlass', 'csrc/third_party/pto-isa'):
    expected = git('rev-parse', f'HEAD:{path}')
    actual = git('-C', path, 'rev-parse', 'HEAD')
    assert actual == expected, (path, expected, actual)
    assert git('-C', path, 'status', '--porcelain') == '', path
    print(path, actual)
assert git('-C', 'refs/vllm', 'rev-parse', 'HEAD') == pathlib.Path('.github/vllm-main-verified.commit').read_text().strip()
assert git('-C', 'csrc/third_party/pto-isa', 'rev-parse', 'HEAD') == '781ac0eb2e4d31e0bc2ab953bca9f9036bec7528'
print('PIN_CHECK_PASS')
PY
git submodule status --recursive > "$run_dir/submodules.txt"
git show -s --format=fuller HEAD > "$run_dir/candidate-info.txt"
touch "$run_dir/DONE-02"
```

**成功条件：** `PIN_CHECK_PASS`，完整 candidate 已保存，checkout 干净。
如果交接消息提供了指定 SHA，还必须核对 `candidate.txt` 与其完全一致；不一致就停止并报告分支已移动。
`refs/xllm`、`refs/xllm-ops` 未初始化是正常的，它们不参与构建；CATLASS/PTO/vLLM 未初始化则不能继续。
此后不要再次 fetch、更换 commit、rebase 或追随任何 main。执行的是 candidate，而不是一个会移动的分支名。

## 3. 配置本次工具链与独立 Python 环境

先从服务器既有部署资料确认这两个值：

| 参数 | 必须填什么 |
| --- | --- |
| `cann_env` | 已安装且准备验证的 CANN toolkit 的 `set_env.sh` **绝对路径** |
| `soc_version` | 实际 910B 子型号对应值，例如 `ascend910b1`；例子不是对本机型号的判断 |

记录容器镜像 tag+digest、操作系统、驱动/固件版本、CANN toolkit/compiler 版本与来源。
可以读取现有版本文件和部署记录；未知项明确写 `unknown`，不能编造。找不到可确认的 CANN 环境或 SoC 时停止并报告 `BLOCKED_ENV`。
使用已分配的可见设备；`npu:0` 指当前进程的第一个可见 NPU，不等于机器物理卡 0。

```bash
source /ABSOLUTE_RUN_DIR/session.sh
require_done 02
cann_env='/REPLACE_WITH_CONFIRMED_CANN_PATH/set_env.sh'
soc_version='REPLACE_WITH_CONFIRMED_SOC_VERSION'
test -f "$cann_env"
case "$soc_version" in ascend910b*) ;; *) echo 'STOP: actual 910B SoC required'; exit 1 ;; esac
run_logged 03-venv python3.12 -m venv "$run_dir/venv"
{
    printf 'set +u\n'
    printf 'source %q\n' "$cann_env"
    printf 'source %q\n' "$run_dir/venv/bin/activate"
    printf 'set -u\n'
    printf 'export SOC_VERSION=%q\n' "$soc_version"
} > "$run_dir/runtime.sh"
source "$run_dir/runtime.sh"
run_logged 03-environment bash -euo pipefail -c 'uname -a; cat /etc/os-release; python --version; gcc --version; g++ --version'
run_logged 03-device npu-smi info
python -c 'import pathlib, sys; assert sys.version_info[:2] == (3, 12); assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve(); print(sys.executable)' "$run_dir/venv"
printf 'SOC_VERSION=%s\nCANN_ENV=%s\n' "$SOC_VERSION" "$cann_env" > "$run_dir/toolchain.txt"
touch "$run_dir/DONE-03"
```

**成功条件：** Python 路径位于本次 `run_dir/venv`，CANN 脚本加载成功，SoC 来自实际环境。
在 `toolchain.txt` 补充上述镜像、驱动/固件、CANN 版本信息；不要保存整个 `env` 输出或认证信息。
`set +u` 仅用于兼容 CANN 环境脚本，随后恢复严格模式；它不忽略 CANN 脚本的失败退出码。

## 4. 安装、完整编译和本地检查

按顺序执行 vLLM → vLLM Ascend → 测试工具。沿用本仓安装文档的包索引，默认保留构建隔离。
不要自行加 `--no-deps`、`--no-build-isolation`、`COMPILE_CUSTOM_KERNELS=0`，也不要安装另一个 vLLM wheel 覆盖源码版本。

```bash
source /ABSOLUTE_RUN_DIR/session.sh
require_done 03
run_logged 04-install-vllm env VLLM_TARGET_DEVICE=empty \
    python -m pip install -v -e refs/vllm --extra-index-url https://download.pytorch.org/whl/cpu/
run_logged 04-build-ascend env COMPILE_CUSTOM_KERNELS=1 \
    python -m pip install -v -e . \
    --extra-index-url https://download.pytorch.org/whl/cpu/ \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
run_logged 04-test-tools python -m pip install -r requirements-lint.txt pytest==8.3.5 cmake ninja
run_logged 04-pip-check python -m pip check
python -m pip freeze > "$run_dir/pip-freeze.txt"
run_logged 04-format bash format.sh ci
run_logged 04-cpu python -m pytest --confcutdir=tests/ut/ops/mega_gdn \
    --basetemp="$run_dir/pytest-tmp" --junitxml="$run_dir/results/cpu.xml" tests/ut/ops/mega_gdn -q
run_logged 04-cpu-check python - "$run_dir/results/cpu.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
assert len(list(root.iter('testcase'))) == 29
assert not any(node.tag in ('failure', 'error', 'skipped') for node in root.iter())
print('CPU_PASS: 29/29; skipped=0')
PY
run_logged 04-symbolic-meta python tools/check_symbolic_meta.py
test -z "$(git status --porcelain --untracked-files=no)"
touch "$run_dir/DONE-04"
```

**成功条件：** 每个 `04-*.exit` 为 0，CPU 输出 `29 passed` 且没有 skipped，父仓和已初始化子仓无改动。
`format.sh ci` 可能自动修改文件；如果因此退出非零或产生 diff，保留 diff 并停止，不提交修复或忽略该检查。
`pip install` 成功本身不能证明算子已编译。日志必须包含 `ascend_mega_gdn_mtp_decode` 构建/打包记录，下一步还要检查真实加载。
新 checkout 避免复用旧构建目录；本轮失败不要执行 `git clean -fdx` 或盲删全局缓存。

## 5. 检查包路径、ACLNN 符号与构建产物

这一步不能省略。要排除“import 到另一套 editable 包”及“旧 `.so` 被加载”的情况。

```bash
source /ABSOLUTE_RUN_DIR/session.sh
require_done 04
run_logged 05-load python - "$repo_dir" "$run_dir" <<'PY'
import ctypes, hashlib, importlib, json, os, pathlib, sys
import torch
import torch_npu
import vllm
import vllm_ascend
from vllm_ascend.utils import enable_custom_op

root, run = map(lambda x: pathlib.Path(x).resolve(), sys.argv[1:])
assert torch.npu.is_available(), 'NPU unavailable'
device = torch.npu.get_device_name(0)
assert '910B' in device.upper(), device
assert pathlib.Path(vllm.__file__).resolve().is_relative_to(root / 'refs/vllm'), vllm.__file__
assert pathlib.Path(vllm_ascend.__file__).resolve().is_relative_to(root), vllm_ascend.__file__
assert enable_custom_op(), 'custom operator load failed'
extension = importlib.import_module('vllm_ascend.vllm_ascend_C')
extension_path = pathlib.Path(extension.__file__).resolve()
assert extension_path.is_relative_to(root), str(extension_path)
vendor = root / 'vllm_ascend/_cann_ops_custom/vendors/custom_transformer'
library_path = vendor / 'op_api/lib/libcust_opapi.so'
assert library_path.is_file(), str(library_path)
library = ctypes.CDLL(str(library_path))
for symbol in ('aclnnAscendMegaGdnMtpDecodeGetWorkspaceSize', 'aclnnAscendMegaGdnMtpDecode'):
    assert getattr(library, symbol), symbol
schema = str(torch.ops._C_ascend.npu_mega_gdn_mtp_decode.default._schema)
record = {
    'python': sys.executable, 'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
    'device': device, 'vllm': vllm.__file__, 'vllm_ascend': vllm_ascend.__file__,
    'extension': str(extension_path), 'opapi': str(library_path), 'schema': schema,
    'ASCEND_CUSTOM_OPP_PATH': os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
    'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (extension_path, library_path)},
}
(run / 'load-info.json').write_text(json.dumps(record, indent=2))
(run / 'loaded-maps.txt').write_text(pathlib.Path('/proc/self/maps').read_text())
print(json.dumps(record, indent=2))
print('LOAD_CHECK_PASS')
PY
find vllm_ascend/_cann_ops_custom -type f -print0 | sort -z | \
    xargs -0 -r sha256sum > "$run_dir/cann-artifacts.sha256"
grep -n 'ascend_mega_gdn_mtp_decode' "$run_dir/logs/04-build-ascend.log" > "$run_dir/mega-build-lines.txt"
touch "$run_dir/DONE-05"
```

**成功条件：** `LOAD_CHECK_PASS`，路径指向本次 checkout，两个符号均存在，`mega-build-lines.txt` 非空。
此时只证明加载门槛，真正调用 kernel 由后续测试证明。任何找不到符号/错路径都停止，不手动复制或软链接旧库。

## 6. 先运行 16-case smoke

```bash
source /ABSOLUTE_RUN_DIR/session.sh
require_done 05
run_logged 06-smoke python -m tools.mega_gdn.validate \
    --expected-commit "$candidate" --output "$run_dir/results/smoke"
run_logged 06-check python - "$run_dir/results/smoke/results.json" "$candidate" <<'PY'
import json, pathlib, sys
result = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert result['status'] == 'passed', result.get('error')
assert result['manifest']['parent_commit'] == sys.argv[2]
assert len(result['cases']) == 16
assert all(case['status'] == 'passed' for case in result['cases'])
print('SMOKE_PASS: 16/16; commit=' + sys.argv[2])
PY
touch "$run_dir/DONE-06"
```

**成功条件：** `SMOKE_PASS: 16/16`，运行与检查进程都退出 0。否则不运行 matrix。
覆盖 S=2/3、B=2、NK=2/NV=4、两种矩阵布局、原位/共享只读源、accepted=1/S。
验证还包括 Meta 形状、mutation schema、非连续 state 拒绝、Conv/SSM/norm 数值、未触及块与 state 地址。
Meta 检查成功不等于图模式通过；这些带同步/dump 的测试不是性能测试。

## 7. smoke 通过后运行 128-case matrix

```bash
source /ABSOLUTE_RUN_DIR/session.sh
require_done 06
run_logged 07-matrix python -m tools.mega_gdn.validate --matrix \
    --expected-commit "$candidate" --output "$run_dir/results/matrix"
run_logged 07-check python - "$run_dir/results/matrix/results.json" "$candidate" <<'PY'
import json, pathlib, sys
result = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert result['status'] == 'passed', result.get('error')
assert result['manifest']['parent_commit'] == sys.argv[2]
assert len(result['cases']) == 128
assert all(case['status'] == 'passed' for case in result['cases'])
print('MATRIX_PASS: 128/128; commit=' + sys.argv[2])
PY
touch "$run_dir/DONE-07"
```

**成功条件：** `MATRIX_PASS: 128/128`。覆盖 S=2..17，其他几何同 smoke。
没有覆盖所有 B/N/head 几何，不能写成“所有形状支持”。out 容差是原版独立算子的测试标准，不是 vLLM 模型等价标准。
失败时保留 `results.json`、`last-inputs.pt`、`last-case.json`、存在时的 `failure.pt`。
若设备错误发生在 MEGA launch 前的原生 Conv，可能还没有 `last-inputs.pt`；如实报告文件不存在，不能伪造或补跑掩盖失败。
不要在同一已被修改的 state 上运行 fallback；脚本非零退出后结束该测试进程。

## 8. 可选 P0 原生模型采集：只使用已确认的输入

只有第 7 步通过，且拿到以下资料才执行本节：本地模型路径及 revision、服务器已跑通的 `engine-args.json`、测试 `prompts.json`、准确 GDN 层 prefix。
缺任意一项，在 `run_dir/P0-status.txt` 写 `BLOCKED_MODEL_INPUT` 和缺项，然后执行第 9 步；**不要为找模型阻塞独立算子结果回传**。
不得随意下载模型、猜 speculative method、把 `model.layers.0.linear_attn` 示例当实际 prefix，或为适配显存擅自改变 TP。

引擎 JSON 传给 `vllm.LLM`：要求 `enforce_eager=true`、TP=PP=1、`async_scheduling=false`、`enable_prefix_caching=false`，无 KV connector，保留该模型已确认的 `speculative_config`。
prompts 是固定测试文本的 JSON 字符串列表。脚本仅采集被选层的纯 spec 调用，不启用 MEGA 模型路由。

```bash
source /ABSOLUTE_RUN_DIR/session.sh
require_done 07
engine_args='/REPLACE_WITH_CONFIRMED_PATH/engine-args.json'
prompts='/REPLACE_WITH_CONFIRMED_PATH/prompts.json'
layer_prefix='REPLACE_WITH_CONFIRMED_GDN_LAYER_PREFIX'
test -f "$engine_args"
test -f "$prompts"
[[ "$layer_prefix" != REPLACE_* ]]
cp "$engine_args" "$run_dir/engine-args.json"
cp "$prompts" "$run_dir/prompts.json"
printf '%s\n' "$layer_prefix" > "$run_dir/layer-prefix.txt"
run_logged 08-native env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python -m tools.mega_gdn.capture_native \
    --engine-args "$run_dir/engine-args.json" --prompts "$run_dir/prompts.json" \
    --layer "$layer_prefix" --calls 2 --max-tokens 32 --output "$run_dir/results/native"
run_logged 08-check python - "$run_dir/results/native" "$candidate" <<'PY'
import json, pathlib, sys
folder = pathlib.Path(sys.argv[1])
result = json.loads((folder / 'manifest.json').read_text())
assert result['status'] == 'captured_pending_analysis'
assert result['manifest']['parent_commit'] == sys.argv[2]
for name in ('call-000.pt', 'call-001.pt', 'outputs.json'):
    assert (folder / name).is_file(), name
print('NATIVE_CAPTURED_PENDING_ANALYSIS')
PY
printf 'captured_pending_analysis\n' > "$run_dir/P0-status.txt"
touch "$run_dir/DONE-08"
```

采集 qkv/a/b/z、权重、accepted、物理表、Conv/SSM stride、被引用块前后值、core/norm 输出。
只有 `*-before.pt` 而没有两份完整 `call-000.pt`/`call-001.pt` 时不能算采集成功。
本节会产生 CPU 同步，仅供状态/数值分析，不报告吞吐收益。

## 9. 无论成功或失败：收集证据并交付

本节故意不 source session/runtime，避免 CANN/venv 的初始化故障再次阻断日志回收。
填真实 run_dir，先将阶段结果写入 `SUMMARY.md`：candidate、每步状态、首次失败命令/退出码、错误原文片段、缺项、未执行步骤。
保留完整日志，摘要不能代替日志。模型配置中若含认证字段，回传前移除认证字段并说明，不公开上传证据包。

```bash
set -euo pipefail
run_dir='/ABSOLUTE_RUN_DIR'
test -d "$run_dir/logs"
repo_dir="$run_dir/src"
if [[ -d "$repo_dir/.git" ]]; then
    git -C "$repo_dir" status --short > "$run_dir/final-git-status.txt"
    git -C "$repo_dir" diff > "$run_dir/final-source.diff"
    git -C "$repo_dir" submodule status --recursive > "$run_dir/final-submodules.txt"
fi
test -s "$run_dir/SUMMARY.md"
archive="${run_dir}.tar.gz"
test ! -e "$archive"
tar -czf "$archive" --exclude='./src' --exclude='./venv' --exclude='./pytest-tmp' -C "$run_dir" .
sha256sum "$archive" | tee "${archive}.sha256"
printf 'EVIDENCE_ARCHIVE=%s\n' "$archive"
```

返回格式（填实际值）：

```text
candidate: <完整 40 位 SHA>
SoC / CANN / torch / torch_npu: <实际值；未知注明 unknown>
构建 / 包加载: PASS | FAIL | NOT_RUN
CPU / format: <实际结果>
smoke: PASS 16/16 | FAIL <首个 case> | NOT_RUN
matrix: PASS 128/128 | FAIL <首个 case> | NOT_RUN
P0: captured_pending_analysis | BLOCKED_MODEL_INPUT | FAIL | NOT_RUN
首次失败: <步骤、命令、退出码、首个有效错误；成功则 none>
源码是否被修改: <git status/diff 结果>
证据包: <绝对路径>，sha256: <值>
结论: <仅陈述本轮独立 ABI 和采集范围；模型接入/性能未验证>
```

## 10. 失败定位表：允许检查什么

| 现象 | 只读检查 / 应回传证据 | 不要做什么 |
| --- | --- | --- |
| clone/submodule 失败 | URL、失败退出码、Git 完整日志、缺失 gitlink | 改 pin、换仓库、追 main、禁用证书校验 |
| 缺 CANN/PTO 头或编译器 | 首个编译错误、实际 include/toolkit 路径、PTO SHA | 删除算子、关闭 custom kernels、吞掉编译错误 |
| pip 依赖冲突 | resolver 原文、pip freeze、当前 Python、包索引来源 | 随意升降 torch/torch_npu/vLLM、忽略 pip check |
| 找不到 ACLNN 符号/错包路径 | build log、load log、OPP、library hash、loaded maps | 拷贝旧 `.so`、软链另一个 OPP、把 import 成功当 kernel 成功 |
| 数值不符 | 首个 case、JSON 误差、inputs、actual/expected dump | 扩容差、跳过 case、改 dtype、把原版 ABI 当物理 ABI |
| NPU 非法访问/超时/OOM | 最后完整日志、退出码、npu-smi；标明 dump 是否生成 | 在污染 state 上 fallback、自动重启设备、反复重跑 |
| P0 无完整 dump | 层 prefix、已确认模型配置、是否命中纯 spec、进程放置 | 猜 method/层名、把 before-only 文件当完整采集 |

远端结果回传后，由开发端处理首个错误或推进 P2 物理状态 ABI。参见 [当前契约](04-current-contract.md) 和 [第 01 轮本地报告](rounds/round-01.md)。
