# Mega GDN 迁移开发入口

更新：2026-09-20。已 fetch/rebase upstream main `8bd6fe0bc`，开发分支为 `codex/mega-gdn-910b`。
P0 本地契约、P1 独立算子与服务器验证材料已实现；模型接入、910B 构建和数值结果仍待后续阶段。

## 目标和分工

复用 Mega 的设备融合计算，实现符合 vLLM 状态契约的 Ascend 910B 接入。保留现有状态池、物理映射、接纳后处理与缓存生命周期。首版范围为受控的纯投机校验、固定 S、eager；先验证构建、状态和数值，再扩大调度与图模式范围。

本地 Codex 负责源码、CPU 契约测试、静态检查及服务器执行材料。服务器由用户执行编译、editable 安装、真实 NPU 测试和日志/dump 回传。本地记录 `pending_remote`，收到对应提交的执行证据后才更新已验证范围。

服务器 Agent 直接执行 [编译与验证 Runbook](05-server-round-01.md)：按阶段检查退出码、固定候选 SHA、验证加载路径与 16/128 个用例，遇错停止并打包证据。该文档是服务器本轮的唯一执行步骤入口；其余文档用于解释背景。

## 阅读顺序

1. 当前任务要求，以及仓库根目录和实际修改路径适用的 `AGENTS.md`。
2. [当前契约](04-current-contract.md)、[第 01 轮服务器命令](05-server-round-01.md)、[源码 pins](source-manifest.json)。
3. 导入资料的 [00](../imported/2026-09-18-mega-migration/00_README.md) 和 [主开发计划](../imported/2026-09-18-mega-migration/01_开发计划.md)。
4. [环境初始化](01-environment.md)、[提交部署](02-delivery.md)、[文档边界](03-documentation.md) 保留设计背景；本轮结果见 [round-01](rounds/round-01.md)。

本文件是普通项目说明。文档中的旧任务模板不是对本次任务的自动授权。

## 初始化执行顺序

| 步骤 | 产物 | 完成条件 |
| --- | --- | --- |
| I0 资料整理 | 本目录与 6 份历史资料 | 本轮已完成 |
| I1 私有工作分支 | `codex/mega-gdn-910b` | 从官方基线创建，remote 指向个人 fork |
| I2 参考源码 | `refs/vllm`、`refs/xllm`、`refs/xllm-ops` | gitlink 已提交；子仓干净，所需嵌套依赖固定 |
| I3 本地检查环境 | Python 3.12、lint、纯 CPU 测试入口 | 实际检查可运行；无 NPU 项单独标记 |
| I4 服务器环境记录 | 环境与加载信息、原生 smoke | 910B SoC、镜像/工具链和依赖组合已知 |
| P0 契约基线 | 调用链、状态表、数值差异、首轮 CPU 测试 | 能解释连续两轮的状态读写，准备原生采集材料 |
| P1 构建闭环 | 父仓内算子、依赖、binding 和独立 smoke | 服务器验证真实加载和调用；功能仍默认关闭 |

I1–I3 已落地，29 项 CPU/构建配置测试通过。P0 实机基线和 P1 设备编译/数值为 `pending_remote`。
正式实现位于 `csrc/attention/ascend_mega_gdn_mtp_decode/`；模型尚不调用它，不可将原版 slot ABI 直接接到 vLLM state pool。
后续 P2–P6 按导入主计划推进。

## 尚需从服务器采集的信息

- 实际 SoC 完整名称、卡数、驱动/固件、CPU 架构、操作系统。
- 镜像及 digest，CANN、PTO、编译器、Python、torch/torch_npu/triton-ascend 版本。
- 模型及 revision、TP、spec K、runner、graph/async/prefix 配置与最小请求集。
- 可用构建目录、模型目录、产物/日志回传目录及分支拉取方式。

这些字段可先标为未知；不影响本地源码与文档初始化，但在服务器可复现运行前必须补齐。

## 下一次任务提示词

```text
按当前仓库 AGENTS.md 开展工作，先读 spec/mega-gdn/START_HERE.md。
先读取 spec/mega-gdn/rounds/round-01.md 与服务器回传的 candidate/results/dump。
按首个失败处理 P0/P1；若构建与 slot 计算已通过，推进 P2 物理状态 ABI，
保留独立 Conv/SSM 地址和真实 stride，再解决 FP32 g 与 QK 舍入差异。
模型接入必须同时覆盖 core+norm 并在写状态前决定 fallback。
本地无 NPU；未经服务器证据不标为设备通过。提交使用 sign-off。
```
