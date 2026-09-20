# MEGA P0/P1 第 01 轮本地交付

日期：2026-09-20。状态：`pending_remote`；本地实现和检查已完成，未运行 CANN/NPU。

## 版本与范围

- `main` 从 `aff1b74b66467a7805cde69ef0728b7e32c0f990` fetch/rebase 到官方 `8bd6fe0bc519e2beb9930687ac3e89f2245aa70d`，无冲突。
- 在此基线创建 `codex/mega-gdn-910b`；精确依赖和来源 blob hash 见 [source-manifest.json](../source-manifest.json)。
- `refs/vllm` 固定为本仓 CI 验证提交；xLLM/xLLM-ops 为本轮获取的 main 快照。PTO 使用 xLLM-ops 的确切 gitlink。
- GDN/metadata/postprocess 在本次上游更新中没有发生需要重做接口的变化；构建脚本基于更新后内容修改。
- 上游 `AGENTS.md` 未变。历史 6 份导入资料保留，新增字节属性避免跨平台换行破坏 archive hash。

## 实现

- 父仓移入 MEGA MTP Host/proto/tiling/kernel 和单 token MEGA 的公共 PTO 头；保留许可证与来源。
- 唯一 CANN 名称 `AscendMegaGdnMtpDecode`，仅在 910B op 列表构建。PTO 是父仓独立依赖，构建不读取 `refs/`。
- PyTorch 独立入口、原位状态 schema、shape/dtype/device/contiguity/base-format/alias 检查和符号 Meta 输出。
- CPU 状态地址模型、原版数值 oracle、非连续块和跨轮回放测试。
- 显式 NPU smoke/matrix runner，采集实际加载路径、hash、shape/stride、误差及失败复现输入；另提供原生层定向诊断脚本。
- CATLASS 缺失初始化限定到其自身子仓，避免拉入三个参考项目的所有依赖。

## 实际本地检查

环境：Windows、Python 3.12.13、torch 2.10.0+cpu、pytest 8.3.5、Ruff 0.14.0、CMake 4.4.3、Ninja 1.13.2。

| 检查 | 结果 / 边界 |
| --- | --- |
| CPU pytest | 29 passed，包括状态访问/冲突、游标、batch 重排、连续两轮、FLA/non-FLA 等价、采集 stride/PAD |
| CMake 配置（含在 29 项内） | 使用父仓真实注册 macro，校验 op 名/注册目录和缺 PTO 时明确失败；不编译 CANN |
| Ruff check/format | 通过 |
| codespell | 修正一处迁入源码注释后通过 |
| Markdown、typos、clang-format、actionlint | 全仓格式命令对应项目通过；后续新增文档另行检查 |
| Python 目录/导入/长函数检查、symbolic Meta | 通过 |
| `bash -n csrc/build_aclnn.sh`、ShellCheck 0.11.0 | 通过；本地脚本换行已规范为 LF |
| Gitleaks 8.30.1 | 通过，staged changes 无泄漏；使用官方 Windows 版本从 Git Bash 执行仓库原脚本 |
| logger 检查 | 从 Git Bash 显式执行原脚本通过 |
| 导入资料 | 12 个正文/HTML hash 一致 |
| `bash format.sh ci` 整体入口 | 本机未整体通过：部分 hook 的 `/bin/bash` 绝对路径在 Windows 无法解析；ShellCheck 初始未安装。已补工具并对修改脚本执行 ShellCheck，对 Gitleaks/logger 执行原脚本；Linux 入口仍需服务器复核 |
| 910B 编译、真实 dispatch/Meta、数值、原生模型、E2E、性能 | 均为 `pending_remote` |

可复现本地测试（PowerShell，workspace 根目录）：

```powershell
.venv/Scripts/python.exe -m pytest --confcutdir=tests/ut/ops/mega_gdn `
  --basetemp=.pytest_cache/mega-local-check tests/ut/ops/mega_gdn -q
.venv/Scripts/ruff.exe check tools/mega_gdn tests/ut/ops/mega_gdn
.venv/Scripts/python.exe tools/check_symbolic_meta.py
```

`--basetemp` 用于避开本机系统临时目录的 ACL 问题；只使用仓库内专用临时目录，不指向已有数据。

## 判定和下一轮

本轮可以交付独立构建候选，不能宣布 MEGA 模型迁移完成。原版 slot×S 与 vLLM 物理表不兼容，FP32 gate、Q/K 舍入和真实 stride 仍未改造。
模型没有路由到 MEGA；现有 core+norm 路径保持原样。不存在已经验证的 fallback、图模式、prefix COW 或 async 支持。

按 [服务器第 01 轮命令](../05-server-round-01.md) 回传完整结果。
先关闭 P0/P1 的编译、加载、状态证据缺口，再实施 P2 物理 ABI 和 P3 core+norm 模型路由。
每次远端实验用 clean checkout 的完整 HEAD 标识；本报告不把易变分支名当执行证据。
