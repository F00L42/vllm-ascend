# 算子迁移资料与初始化方案

更新：2026-09-20。当前目标：迁移 MegaGdnMtpDecode 至 Ascend 910B；已交付首轮本地 P0/P1 实现。

本目录是项目资料区。它不替代仓库根目录及实际修改路径上的 `AGENTS.md`，也不自动启用迁移开发或服务器操作。

## 本轮交付

- 已读取指定会话的全部 6 份文档，保存为 Markdown；见下方资料索引。
- 已 fetch/rebase 官方 main `8bd6fe0bc`，创建个人分支并锁定三个参考 submodule 和 PTO。
- 已移入独立算子 Host/Device，新增 PyTorch binding/Meta、CPU 契约与数值参考、服务器验证和原生采集脚本。
- 根目录 `AGENTS.md` 保持上游内容；当前进度以开发入口和轮次报告为准。P2 模型状态适配及设备验证尚未完成。

## 从哪里开始

| 文档 | 用途 |
| --- | --- |
| [开发入口](mega-gdn/START_HERE.md) | 下一次任务的阅读入口、现状与首轮工作 |
| [环境初始化方案](mega-gdn/01-environment.md) | 三个参考 submodule、版本选择、本地和服务器环境 |
| [提交与部署流程](mega-gdn/02-delivery.md) | 每日 rebase、固定候选 tag、服务器编译安装、结果回传 |
| [文档与 Agent 边界](mega-gdn/03-documentation.md) | 历史资料、当前决策、测试证据及仓库指令的区分 |
| [当前状态契约](mega-gdn/04-current-contract.md) | 当前源码链、原版 slot ABI、已知数值差异 |
| [第 01 轮服务器验证](mega-gdn/05-server-round-01.md) | 构建、独立算子 smoke/matrix、原生状态采集与回传 |
| [第 01 轮报告](mega-gdn/rounds/round-01.md) | 本轮实际检查、限制和下一步 |
| [轮次模板](mega-gdn/templates/round.md) | 每轮开发、服务器执行与判定 |
| [运行清单模板](mega-gdn/templates/run-manifest.example.json) | 环境、版本、加载产物和测试状态；`null` 表示待采集 |

## 会话资料索引

来源：[生成迁移文档](https://chatgpt.com/c/6aace1e2-da84-83ed-a2ad-e7fe157a2e6f)。

| 文件 | 内容 |
| --- | --- |
| [00_README.md](imported/2026-09-18-mega-migration/00_README.md) | 阅读顺序、约束、证据口径 |
| [01_开发计划.md](imported/2026-09-18-mega-migration/01_开发计划.md) | P0–P6 开发主线、验收门槛、收益范围 |
| [02_状态与数值契约.md](imported/2026-09-18-mega-migration/02_状态与数值契约.md) | SSM/Conv、checkpoint、游标、数值与所有权 |
| [03_构建接入与源码导航.md](imported/2026-09-18-mega-migration/03_构建接入与源码导航.md) | PTO/CANN、注册、core+norm、历史源码依据 |
| [04_服务器验证与Telemetry.md](imported/2026-09-18-mega-migration/04_服务器验证与Telemetry.md) | 测试矩阵、诊断、可重放 dump、性能 |
| [05_本地Codex执行手册.md](imported/2026-09-18-mega-migration/05_本地Codex执行手册.md) | 任务说明、每日更新、轮次报告 |

导入方法和完整性边界见 [IMPORT.md](imported/2026-09-18-mega-migration/IMPORT.md)。原文中的“本次复核”属于原会话的历史证据，不能解释成本地本轮已复核或服务器已验证。

## 资料导入时检查（2026-09-18）

- `markdownlint-cli@0.45.0`：13 份 Markdown 通过。
- 文档引用：32 个本地链接存在；2 份 JSON 可解析。
- 导入校验：6 份正文和链接转换回读一致；12 个 Markdown/HTML 文件 SHA-256 校验通过。
- `bash format.sh ci`：已尝试，因本机未安装 `pre-commit` 未能运行完整检查。
- 本轮为文档工作，没有执行 CANN 编译、NPU UT、E2E 或性能测试。

2026-09-20 的实现检查另记于第 01 轮报告；历史资料和导入 hash 不改写。
