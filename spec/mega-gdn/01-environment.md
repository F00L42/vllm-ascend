# 环境初始化方案

设计日期：2026-09-18。下列是初始化方案及当时快照；2026-09-20 已完成本地 I1–I3，实际状态见 [开发入口](START_HERE.md) 和 [源码 pins](source-manifest.json)。除特别说明外，命令使用 Git Bash/Linux Bash，从父仓根目录运行。

## 1. 当前仓库事实

| 项目 | 本轮读取到的值 |
| --- | --- |
| 当前分支 | `main`，整理资料前工作树干净 |
| 父仓 HEAD / 本地 upstream/main | `aff1b74b66467a7805cde69ef0728b7e32c0f990` |
| origin | `git@github.com:F00L42/vllm-ascend.git`，个人 fork |
| upstream | `git@github.com:vllm-project/vllm-ascend.git` |
| CI 验证的 vLLM main 提交 | `.github/vllm-main-verified.commit`：`84030bbe3d74d99bad477a3d2e37a973ccd8865c` |
| 已有 submodule | `csrc/third_party/catlass`，尚未初始化 |
| CATLASS gitlink | `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |

这里的 upstream/main 是本地 remote-tracking ref，不宣称已经 fetch 到官方当日最新。执行初始化时重新 fetch 并记下实际 SHA。

当前 `requirements.txt` 固定了 torch 2.10.0、torch-npu 2.10.0.post4、triton-ascend 3.2.2；它们属于上述父仓快照，不应永久抄写成未来服务器版本要求。服务器 Python 优先与本仓 A2 CI 的 3.12 对齐；CANN/驱动/PTO 组合仍需实机确认。

## 2. 参考源码和构建依赖

建议在个人开发分支增加以下布局，现有上游目录保留：

```text
vllm-ascend/
  AGENTS.md                         # 上游项目规范
  refs/
    vllm/                           # submodule；状态契约与运行时源码
    xllm/                           # submodule；模型与调度参考
    xllm-ops/                       # submodule；Mega 源码/reference
  spec/                             # 本轮所有资料、后续决策和小型报告
  csrc/attention/ascend_mega_gdn_mtp_decode/ # P1 实现，尚未接模型
  csrc/third_party/                  # 已有 CATLASS；必要的 PTO 构建依赖另行明确
```

`refs/` 主要用于本地开发和实验复现。正式算子实现、注册、测试和必要构建依赖应随父仓交付，不能让产品构建隐式依赖一整套 xLLM 引擎。首个上游 PR 应能脱离个人 `refs/` 布局构建。

已核对官方仓库：[vLLM](https://github.com/vllm-project/vllm)、[xLLM](https://github.com/xLLM-AI/xllm)、[xLLM-ops](https://github.com/xLLM-AI/xllm-ops)。后两者主分支均为 `main`。

## 3. 建分支和添加 submodule

先检查当前改动。本轮 `spec/` 是未提交资料；切换到由当前基线新建的分支后，可作为独立文档提交保留。不要直接在 `main` 开发算子。

```bash
git status --short --branch
git remote -v
git fetch upstream main
git switch -c codex/mega-gdn-910b upstream/main

# 本轮资料单独提交，后续上游 PR 容易排除。
git add spec
git commit -s -m "docs(mega-gdn): preserve migration references and workflow"

git submodule add -b main https://github.com/vllm-project/vllm.git refs/vllm
git submodule add -b main https://github.com/xLLM-AI/xllm.git refs/xllm
git submodule add -b main https://github.com/xLLM-AI/xllm-ops.git refs/xllm-ops

# 优先使用父仓 CI 已验证的配套 vLLM，避免把框架漂移混入算子实验。
vllm_ref=$(tr -d '[:space:]' < .github/vllm-main-verified.commit)
git -C refs/vllm fetch origin "$vllm_ref"
git -C refs/vllm checkout --detach "$vllm_ref"

git submodule status
git add .gitmodules refs/vllm refs/xllm refs/xllm-ops
git commit -s -m "chore(mega-gdn): pin source reference submodules"
```

命令逐段执行并检查退出码；已有同名分支或目录时先检查并复用，不强制覆盖。每次提交前按仓库规范完成适用检查；提交不是 NPU 验证通过的证明。

`-b main` 记录远端更新目标，父仓实际保存的是具体提交 gitlink。正常复现使用 `git submodule update --init`，而 `update --remote` 会选择远端分支的新提交；参见 [Git submodule 文档](https://git-scm.com/docs/git-submodule)。

本地每天可以 `fetch` 三仓 main 阅读最新差异；当前实验仍以锁定组合为准。如果明确要测试四仓最新 main，将 vLLM 最新头作为**另一个兼容性实验组合**，记录其与 CI 配套提交的差异，不与已验证组合混写。

更新参考版本时，在干净子仓中 fetch、检查差异、detach 到选定 SHA，再提交父仓 gitlink。不要用 `git submodule foreach git pull`：它既不适合 detached HEAD，也会让三仓独立漂移。xLLM 与 xLLM-ops 的最新头不自动构成已验证配套版本；P0 需要核对实际使用关系。

## 4. 嵌套依赖和源码迁入

先查看子仓 `.gitmodules`，只初始化本阶段需要的依赖。导入资料记录 Mega 使用 `xllm-ops/third_party/pto-isa`，路径和 SHA 要在当前检出版本重新确认。

```bash
git -C refs/xllm-ops config -f .gitmodules --get-regexp 'submodule\..*\.(path|url)'
git submodule update --init csrc/third_party/catlass

# 仅在上一步确认此路径存在且本轮需要 PTO 时执行：
git -C refs/xllm-ops submodule update --init --recursive third_party/pto-isa
git submodule status --recursive
```

原基线的 CATLASS 缺失处理会递归初始化所有子仓。本轮已限制到 `csrc/third_party/catlass`，并为 910B 增加独立 PTO 初始化，构建不拉取三个参考工程。

到 P1 时，建议把所需 Mega Host/Device 文件有选择地迁入父仓，在迁入记录中保留来源 repo、完整 SHA、文件清单、版权头、许可证及改动说明。PTO 采用明确固定的构建依赖，例如父仓 `csrc/third_party/pto-isa` gitlink（拟新增，需核对兼容版本），避免生产构建从 `refs/xllm-ops` 偷取头文件。依赖 pin 和构建脚本随实现一并提交。

原则上三个参考子仓只读。确需修改 vLLM 时，先在个人 vLLM fork 提交并发布，再使父仓 `.gitmodules`/gitlink 指向服务器能取得的提交；或使用带基准 SHA 与失败校验的显式补丁机制。子仓未提交修改、仅存在本地的 commit 都不能通过父仓推送交付。

## 5. 本地开发环境

Windows 上保留当前工作区，Git 操作可用 Git Bash。涉及完整 Bash hooks、Linux 依赖的检查优先在 WSL2 独立 Linux 检出中执行；两个环境通过提交同步，不在 Windows/WSL 交替复用同一个虚拟环境。初期 Windows 足够做源码阅读、文档和标准库 CPU 契约测试。

本地初始化最小环境示例：

```bash
uv venv --python 3.12 .venv
# Git Bash / Windows：
source .venv/Scripts/activate
# Linux / WSL 对应 source .venv/bin/activate
uv pip install -r requirements-lint.txt
uv pip install 'pytest>=6,<9'
pre-commit install
bash format.sh ci
```

这里暂不安装 `requirements-dev.txt` 全部内容：它递归包含 NPU 依赖。本地 CPU 契约测试应尽量隔离设备 import；实际项目 UT 所需依赖按测试补齐，缺环境标 `skipped_env`。不能仅设 `COMPILE_CUSTOM_KERNELS=0` 就假设 Windows 能安装整个 Ascend 栈。

当前 `setup.py` 的 `extras_require` 为空；开发依赖来自 `requirements-dev.txt`，不要把 `pip install -e '.[dev]'` 当作这个版本已经定义的 dev extra。检查入口以实际文件为准。

`format.sh ci` 会运行可能修复文件的 pre-commit hooks；结束后检查 diff。本轮已尝试该命令，但本机缺 `pre-commit`，尚未通过完整仓库检查。

## 6. 服务器环境门槛

服务器使用独立实验目录/镜像或虚拟环境，固定基础镜像 digest 与工具链；不要在正在服务的 editable 源码目录 rebase 或更换二进制。记录 `npu-smi info`、Python、编译器、CANN、驱动、torch/torch_npu/triton-ascend、PTO 和模型 revision。

先在同版本原生路径完成 import 和最小运行，再增加 Mega。vLLM 用 `VLLM_TARGET_DEVICE=empty` 安装源码；Ascend 构建保持 `COMPILE_CUSTOM_KERNELS=1`，`SOC_VERSION` 使用实际 910B 型号。`--no-build-isolation` 仅在构建依赖已按该提交的 `pyproject.toml` 配齐时使用。

依据本地 [安装说明](../../docs/source/getting_started/installation/install_vllm_ascend.inc.md)、[A2 容器说明](../../docs/source/getting_started/installation/cann_image/atlas-a2.inc.md)、[自定义 ACLNN 文档](../../docs/source/developer_guide/Design_Documents/add_custom_aclnn_op.md) 生成当轮命令。不要仅因历史 OpDef 声明 `ascend910b` 就认定实际 CANN/PTO 组合可用。
