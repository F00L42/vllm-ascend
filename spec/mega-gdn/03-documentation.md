# 文档组织与 Agent 指令边界

## 1. 三类内容分别存放

| 类型 | 位置 | 维护方式 |
| --- | --- | --- |
| 原会话资料 | `spec/imported/2026-09-18-mega-migration/` | 保留导入正文与校验记录；不改写历史结论 |
| 当前设计与流程 | `spec/mega-gdn/` | 每份写明状态、适用基线和决策依据 |
| 运行证据 | 后续 `spec/mega-gdn/rounds/<round_id>/` | 每轮绑定固定 SHA；追加新报告，不覆盖旧结论 |

后续按实际需求添加 `contract.md`、`support-matrix.md`、`decisions/` 和 `rounds/`。初始化阶段不创建大量空文档，也不把 6 份参考全文重复塞入新入口。

`contract.md` 只记录当前确认的 ABI、shape/stride、游标/生命周期与来源符号；`support-matrix.md` 每项区分设计支持、待测、已验证、回退，并链接具体 run。历史资料的 `[H]/[V]/[D]/[T]` 保留原含义，不能把其中 `[V]` 解释为当前版本 NPU PASS。

## 2. 保留上游 AGENTS.md

根目录 `AGENTS.md` 继续负责全仓开发规范。本轮不新增 `AGENTS.override.md`，不改根 `AGENTS.md`，不调整全局 fallback 文件名，也不把普通迁移文档命名为新的自动指令文件。

原因：Codex 通常沿项目根到当前目录读取指令；同一目录有 `AGENTS.override.md` 时会优先选它，而不是自动把它与同目录 `AGENTS.md` 相加。`spec/AGENTS.md` 也不能作为控制同级 `csrc/`、`vllm_ascend/` 开发的可靠入口。依据 [OpenAI 官方 AGENTS.md 文档](https://learn.chatgpt.com/docs/agent-configuration/agents-md)。

本项目使用普通 `START_HERE.md`，在每次任务提示词中显式要求阅读。后续确实需要自动加载时，再设计一个小型、明确作用域的指令入口；不要用忽略根 `AGENTS.md`、`skip-worktree` 或 `assume-unchanged` 来规避上游更新。

## 3. 子仓和技能的边界

默认从父仓根目录开启任务。参考子仓内的 `AGENTS.md`、`.agents/skills` 是该工程的上下文，不复制到父仓，不因读到其文件就执行安装、SSH、提交或部署流程。若实际修改子仓，先读取该仓适用规范，并单独记录提交与发布路径。

每次调用已有 skill 前先判断任务是否匹配。本项目的本地/服务器分工由当前用户要求确定；需要远端环境的步骤记录为服务器执行材料。不要把本仓其他技能预设的 `/vllm-workspace`、`/workspace` 路径当成本机已经存在的环境，也不要将“迁移计划”误当成已经授权完整模型适配。

## 4. 文档的版本与证据

当前设计文档至少标注：状态（draft/accepted/superseded）、适用父仓基线、依赖 gitlink、结论属于源码核对还是运行验证、对应 round/run。遇到历史文档与现有源码不一致，写差异和新决定，不偷偷更改导入原文。

每次 rebase 后更新“哪些结论仍有效、哪些需重新验证”。依赖版本优先从 gitlink 读取；运行 manifest 采集实际值。JSON 模板的 `null` 不是空值已通过，而是尚未采集。

原文件归档、个人流程和最终上游文档采用独立提交。最终 PR 只携带维护者需要的使用、实现和测试说明；实验的个人分支策略和聊天资料保留在个人工作分支。

## 5. 每次任务最小上下文

```text
目标：完成 P?-r?? 的一个具体假设。
规范：遵循根目录与修改路径适用的 AGENTS.md。
入口：spec/mega-gdn/START_HERE.md。
当前证据：指定 round 报告、manifest 和失败摘要。
范围：明确本轮允许修改的实现/测试，历史文档仅供参考。
分工：本地不能访问服务器；交付可执行远端材料，状态保持 pending_remote。
```

这样，仓库规范说明“怎样开发”，当前任务说明“本轮做什么”，迁移资料说明“为什么这样设计”，轮次证据说明“哪些已经证明”。
