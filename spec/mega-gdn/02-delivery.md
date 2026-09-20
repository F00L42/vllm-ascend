# 提交、每日更新与服务器验证

本页是拟采用的工作流；示例命令没有在本轮执行。命令使用 Bash，具体轮次、路径和版本在发布时填写。

## 1. 分支与提交组织

| 对象 | 用途 | 更新方式 |
| --- | --- | --- |
| `upstream/main` | 官方集成基线 | 每日 fetch |
| 本地 `main` | 便于对照的上游镜像 | 仅快进，不承载个人功能 |
| `codex/mega-gdn-910b` | 个人持续开发与远端实验分支 | 每日 rebase 到 upstream/main |
| `mega-910b/P0-r01` 等 tag | 一轮服务器候选快照 | 创建后固定，不移动、不覆盖 |
| `codex/mega-gdn-upstream` | 最终上游 PR 分支 | 从新 upstream/main 挑选可独立构建的实现提交 |

建议独立提交：资料导入、参考 submodule、构建与算子来源、状态 ABI 与测试、模型路由与 fallback、诊断、性能优化。每个功能提交包含对应测试，并用 `git commit -s` sign-off。新开关集中登记于 `vllm_ascend/envs.py`；现有环境变量、patch 和 runner 审查要求继续适用。

## 2. 每日 rebase

在工作树及子仓均干净、上一轮候选已固定后执行。下面以远端分支已经存在为前提；首次推送使用 `git push -u origin codex/mega-gdn-910b`。

```bash
set -euo pipefail
git switch codex/mega-gdn-910b
test -z "$(git status --porcelain)"
git submodule foreach --recursive 'test -z "$(git status --porcelain)"'

git fetch upstream main
git fetch origin
lease=$(git rev-parse refs/remotes/origin/codex/mega-gdn-910b)
# 若远端有本地未包含的提交，先审查并整合，不能覆盖他人更新。
git merge-base --is-ancestor "$lease" HEAD
old_base=$(git merge-base HEAD upstream/main)
backup="codex/backup/mega-gdn-$(date +%Y%m%d-%H%M%S)"
git branch "$backup" HEAD
git rebase upstream/main

# 按本轮已初始化依赖同步 gitlink，不追远端 main。
git submodule sync
git submodule update --init refs/vllm refs/xllm refs/xllm-ops csrc/third_party/catlass
git range-diff "$old_base..$backup" "upstream/main..HEAD"
```

有冲突时逐个判断新 API/行为，必要时 `git rebase --abort` 返回备份；不对冲突整目录选 ours/theirs。检查 GDN core/norm、metadata、accepted/postprocess、真实 stride、图/异步依赖及构建注册是否发生语义变化。父仓更新后，重新读取 `.github/vllm-main-verified.commit`，决定是否在单独提交中更新配套 vLLM gitlink。

完成本地测试、`bash format.sh ci` 并审查修复 diff 后，才推送改写过的个人分支。显式 lease 应使用上面已记录且已审查的远端 SHA：

```bash
git push --force-with-lease="refs/heads/codex/mega-gdn-910b:$lease" \
  origin HEAD:refs/heads/codex/mega-gdn-910b
```

在同一 Bash 会话中保留 `lease`；若会话已结束，使用此前记录的完整 SHA，不以新的后台 fetch 结果替代审查过的值。

若推送被拒绝，重新 fetch 并理解远端变化，不能改用 `--force`。显式 expected SHA 可避免后台 fetch 改变普通 lease 的参照值；见 [Git push 文档](https://git-scm.com/docs/git-push)。不要对 upstream/main 或他人共用分支执行上述推送。

每天更新基线不等于每天自动修改所有参考 gitlink。把“查看上游变化”和“升级实验组合”分开记录，保证失败可归因。旧测试报告保留旧 SHA，不能用于给 rebase 后的新 SHA 标记 PASS。

## 3. 固定一轮候选

每轮报告先写明确假设、测试命令、预期结果和需要的回传资料。尚未存在的测试或脚本标为待实现，发布候选时必须换成真实入口。

```bash
set -euo pipefail
test -z "$(git status --porcelain)"
git submodule foreach --recursive 'test -z "$(git status --porcelain)"'
git submodule status --recursive
candidate=$(git rev-parse HEAD)
round_tag=mega-910b/P0-r01  # 每轮使用新名字
git tag -a "$round_tag" "$candidate" -m "P0-r01: native GDN state baseline"
git push origin "refs/tags/$round_tag"
git rev-parse "$round_tag^{commit}"
```

把最后输出的完整 SHA 与 tag 一起交给服务器。tag 固定旧提交，可在开发分支 rebase 后继续复现。实验候选可以是待远端验证状态；明确说明本地检查范围和未执行的 NPU 门槛，不把它当成可合并上游的验收结果。

发布前核对所需 submodule 状态：`+` 表示检出提交与 gitlink 不同，`U` 表示冲突，均需解决；`-` 表示未初始化，只有在本轮明确不需要的参考嵌套依赖上才允许，并在清单中注明。需要构建或采集 SHA 的依赖必须已初始化、无未提交改动且与 gitlink 一致。

不要在一个提交内试图写入“该提交自身的 SHA”：会产生自引用。父仓 gitlink 是依赖 pin 的权威来源；跟踪的基线文件记录 upstream 基点，运行时 manifest 再采集候选实际 HEAD、各子仓/嵌套依赖 SHA 与 dirty 状态。

## 4. 服务器按 tag 和 SHA 建独立目录

服务器维护一个只用于 fetch/worktree 管理的个人 fork 克隆，每轮使用新目录。以下路径为示例，先替换为实际路径；不修改正在运行的服务目录。

```bash
set -euo pipefail
# 首次执行；目录必须尚不存在：
git clone --no-checkout git@github.com:F00L42/vllm-ascend.git /workspace/mega-source
```

后续每轮从 `/workspace/mega-source` 执行：

```bash
set -euo pipefail
round_tag=mega-910b/P0-r01
expected_sha=REPLACE_WITH_FULL_CANDIDATE_SHA
round_dir=/workspace/mega-rounds/P0-r01/vllm-ascend

git fetch origin "refs/tags/$round_tag:refs/tags/$round_tag"
actual_sha=$(git rev-parse "$round_tag^{commit}")
test "$actual_sha" = "$expected_sha"
git worktree add --detach "$round_dir" "$actual_sha"
cd "$round_dir"
git submodule sync
git submodule update --init refs/vllm refs/xllm refs/xllm-ops csrc/third_party/catlass
git submodule status --recursive
```

再按本轮依赖清单显式初始化需要的嵌套依赖/PTO，并核对完整 SHA。真实构建依赖若已迁到父仓 `csrc/third_party/pto-isa`，应初始化该位置。服务器不执行 `git pull` 来合并已 rebase 的开发分支，也不使用 `submodule update --remote`。

若路径或 tag 已存在，检查其用途和 SHA，复用或选择新轮次路径；不要强行重置或删除旧实验。回滚使用上一轮已验证 tag 对应的目录和环境，重新启动测试进程，不复用失败候选已修改的状态池。

## 5. 编译与 editable 安装

先在隔离环境中准备该候选版本的系统依赖、CANN 环境和 Python 构建依赖，并记录基础镜像 digest。按 `pyproject.toml` 与 `requirements-dev.txt` 安装依赖后，再执行下列安装骨架：

```bash
set -euo pipefail
# 位于本轮 vllm-ascend 根目录，已激活本轮 Python 环境及 CANN 环境。
run_dir=/workspace/mega-results/P0-r01
mkdir -p "$run_dir"

VLLM_TARGET_DEVICE=empty python -m pip install -v -e refs/vllm \
  --no-build-isolation --no-deps 2>&1 | tee "$run_dir/vllm-install.log"

export SOC_VERSION=REPLACE_WITH_ACTUAL_910B_SOC
export COMPILE_CUSTOM_KERNELS=1
python -m pip install -v -e . --no-build-isolation --no-deps \
  2>&1 | tee "$run_dir/ascend-build.log"
python -m pip check 2>&1 | tee "$run_dir/pip-check.log"
```

`--no-deps` 的前提是依赖已完整安装且版本匹配；它避免 editable 安装悄悄替换已选定的核心包，但不能修复缺依赖。先对照双方 requirements/build-system，解决冲突；`pip check` 失败就停止。部署材料应记录具体环境准备命令，而不是直接复制带占位符的骨架。

`set -o pipefail` 保留编译失败状态；即使失败，也回传已经生成的环境记录和完整日志。修改 C++/PTO/tiling/ABI 后必须重编译并启动新进程；Python editable 不会自动更新原生二进制。

加载核验至少包含：`sys.executable`、vllm/vllm_ascend/torch/torch_npu 的版本与 `__file__`，真实加载扩展及 OPP 路径、二进制 hash、SoC、构建参数、算子真实 smoke。只检查 `torch.ops` 名称存在不足以证明新 kernel 被调用。禁止使用旧 ABI 的缓存产物来跳过本轮构建。

## 6. 验证与回传闭环

顺序：环境和来源 → 构建 → 加载 smoke → 算子/状态 UT → 最小模型路径 → 当前声明支持的 E2E → 独立性能轮。P0 首先运行原生 GDN 基线；P1 独立算子通过后才进入模型接入。

每轮回传 `manifest.json`、构建/加载日志、测试结果与退出码、`summary.md`；状态问题追加定向事件和最小 pre/post-state dump。报告分别记录 executed/pass、failed、skipped_env、pending_remote，以及实际候选命中次数、fallback 原因和验证范围。具体矩阵见 [服务器验证参考](../imported/2026-09-18-mega-migration/04_服务器验证与Telemetry.md)。

小型摘要放入 `spec/mega-gdn/rounds/<round_id>/`（后续创建）；大日志、trace、NPZ、wheel 和模型文件放独立产物目录，报告只存路径/hash。若确需在仓库内暂存大产物，使用 `spec/artifacts/` 并仅在本地 `.git/info/exclude` 排除；不添加全局忽略规则。

本地收到结果后先确认 tag、完整 SHA、依赖、加载路径和实际路由，再找首个失败层。诊断与性能运行分开：telemetry 默认关闭，性能轮不使用开启 dump 的数据；不因日志新增热路径 `.item()`、`.cpu()` 或同步。

## 7. 向上游提交

完成对应范围的 910B 测试和仓库检查后，从最新 upstream/main 建独立 PR 分支，挑选生产实现及必要依赖/测试提交。个人参考 submodule、历史资料导入和实验记录不随 PR 提交；必要的正式使用说明移入上游已有 docs 结构。

挑选后重新检查构建不依赖 `refs/`，跑受影响测试与 `bash format.sh ci`。新分支提交必须保留 sign-off。推送至个人 `origin` 后向官方仓库发 PR，采用 `[Type][Module] Description` 标题和现有 PR 模板；更新描述时保留自动生成的 `- vLLM version:`、`- vLLM main:` 行。
