# SystemOne 共享前缀：阶段一执行记录

日期：2026-10-15。依据：[共享前缀时延设计](systemone_shared_prefix_latency.md) 第 4 节与 P1。

阶段一的 endpoint 编排代码、CPU 回归测试和远程 HTTP 测量脚本已完成。
本地没有安装 vLLM、torch-npu，也没有可用 NPU；实际混合状态复用、数值一致性、容量和 `T20/T1 <= 1.5` 验收均为 `pending_remote`。
因此，本次完成的是实现与本地验证，尚不满足原设计中“冷缓存实际复用通过”的 P1 远程完成依据。

## 1. 实现基线与修改范围

本地分支：`codex/systemone-token-readout`。开始实施时 HEAD：`a21ec34a9`；logits 读出功能基线：`ecd7d1263f6f2ca33d8a92abdecf2c76e9b12006`。
设计文档内容保持原样，随第一阶段实现和验证记录一起纳入版本控制。交付分支为上述分支；未创建 PR。

| 文件 | 已完成内容 |
|---|---|
| `vllm_ascend/entrypoints/systemone/config.py` | 新增冻结、严格类型的 `SharedPrefixConfig`，位于现有 `systemone_config` 内 |
| `vllm_ascend/entrypoints/systemone/shared_prefix.py` | 能力检查、完整 token LCP、保守有效终点 C 和可定位回退原因 |
| `vllm_ascend/entrypoints/systemone/serving.py` | 内部预填、完成屏障、原 prompt 并发执行、取消清理和父请求日志 |
| `tests/ut/entrypoints/test_systemone_config.py` | 新配置默认值、严格类型、未知字段和不可变性 |
| `tests/ut/entrypoints/test_systemone_shared_prefix.py` | 能力矩阵与 token 边界规划 |
| `tests/ut/entrypoints/test_plumb_serving.py` | 真实服务源码的 CPU 编排、故障、取消和 SDK 合同回归 |
| `benchmarks/scripts/benchmark_systemone_shared_prefix.py` | 逻辑冷缓存的 1/20 题配对 HTTP 测量与失败统计 |
| `tests/ut/entrypoints/test_systemone_shared_prefix_benchmark.py` | 测量隔离、失败计入和响应完整性检查 |

未增加环境变量、模型文件、runner 行为或引擎 patch。复用原生 `generate()`、APC、缓存管理与状态复制。
后续复用分析单列在 [state KV 复用分析](systemone_state_kv_reuse.md)。

## 2. 已实现执行流程

1. 继续用原 `plan_request()` 渲染、分词并检查所有完整 prompt；公共前缀规划在同一个线程池任务内执行。
2. 单问题直接执行；默认关闭或能力不足时直接回退，不扫描长前缀。
3. 能力可用时计算所有完整 token 序列的 LCP，令 `limit=min(LCP, min_prompt_length-1)`，选择 `C=floor(limit/B_verified)*B_verified`。
4. 若没有有效边界或 C 小于配置阈值，保持原始完整 prompt 并发执行。
5. 否则提交首题 `token_ids[:C]` 作为内部 producer，不追加 token。
6. producer 使用 `max_tokens=1`、`temperature=0`、`ignore_eos=True`、`detokenize=False`、`FINAL_ONLY`，不请求 logprobs。
7. 消费到正常 `finished`，继续耗尽并关闭生成器；此后才创建和启动所有子问题。
8. 每个子问题继续提交原始完整 token IDs，保留候选 token、温度、priority、trace 与同一个 `cache_salt`。
9. 子问题各自拥有生成器；出错或取消时，取消并等待全部任务清理后才退出。
   异常退出额外调用 public `engine.abort()`，覆盖原生 `generate()` 在注册完成但 collector 尚未返回时的取消窗口。

每个父组的引擎请求 ID 加入独立 UUID；客户端重复提供 `X-Request-Id` 时，public abort 不会误取消另一组。
正常完成不额外 abort。abort 本身失败时保留原错误并记录异常；实际引擎故障下的资源释放仍需服务器验证。

预填未返回完成信号、异常结束或失败时，不启动子问题。预填完成后实际缓存命中不足，仅记录并返回真实执行结果，不重复预填。
结果保持输入问题顺序；不返回部分答案或伪造概率。
业务 `usage.input_tokens` 仍为完整 prompt 长度之和，`usage.output_tokens=0`；内部 producer 不计入业务 usage。

## 3. 开关与能力边界

默认配置不启用新路径。示例仅适用于**已在实际部署验证每个 2048-token producer 边界都能保存完整混合状态**的情况：

```json
{
  "systemone_config": {
    "mode": "answer_tokens",
    "shared_prefix": {
      "enabled": true,
      "verified_boundary_tokens": 2048,
      "min_shared_tokens": 4096
    }
  }
}
```

| 字段 | 默认值 | 含义 |
|---|---|---|
| `enabled` | `false` | 每个模型实例固定的显式开关 |
| `verified_boundary_tokens` | `null` | 操作者已验证的 producer 终点间隔；必须是正整数 |
| `min_shared_tokens` | `0` | C 的最小阈值；必须是非负整数，可按实测收益调整 |

2048 和 4096 是示例值，本地没有验证其 NPU 效果。配置字段存在、验证开关开启，都不能证明设备保存或命中了状态。
`min_shared_tokens` 只作保守筛选，不预测冷/热缓存收益；热请求多一次 producer 可能变慢，需独立比较。

新路径检查以下条件，不满足则保留原功能并记录回退原因：

- APC 和 chunked prefill 开启，Mamba 模式为 `align`。
- DP=1，eager，同步调度；拒绝外部/混合 DP 路由、PCP/DCP 大于 1 和 KV transfer。
- 已知实际 `cache_config.block_size` 和 `mamba_block_size`；验证间隔必须同时为两者的整数倍。
- 默认稠密留存。目标 vLLM `v0.27.1` 从中央 `vllm.envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 读取留存设置，必须为 `None`；若新版本在 CacheConfig 暴露该设置，同样检查。

保留原服务对 V1、`raw_logprobs`、无量化/投机解码、关闭 `enable_reduce_sample` 等限制。
`prefix_match_unit` 不用来推断状态保存边界。Ascend 初始化可能上调实际物理块大小，必须检查初始化完成后的配置和日志。

主要回退原因包括 `disabled`、`single_question`、`verified_boundary_required`、`physical_block_sizes_unknown`、
`verified_boundary_misaligned`、`data_parallel_unsupported_or_unknown`、`sparse_cache_retention_unsupported`、
`no_common_prefix`、`below_verified_boundary` 和 `below_min_shared_tokens`。其他能力限制也使用独立原因字符串。

## 4. 可观测性

启动日志记录能力检查结果、可见的实际物理块大小、hash 匹配单位和留存设置。
每个父请求输出一行 `SystemOne shared_prefix {JSON}`，不记录 state、token 内容或 salt。

| 字段 | 用途与边界 |
|---|---|
| `parent_request_id`、`request_id` | 原客户端 ID 与加入 UUID 的父组 ID；内部 producer 追加 `-prefill`，子题追加序号 |
| `reason`、`status` | 回退/规划原因；`completed`、`failed` 或 `cancelled` |
| `common_tokens`、`prefill_tokens`、`unshared_common_tokens` | LCP、C、残余公共尾部；单题/能力回退时 LCP 未测，0 表示未测而非实际为空 |
| `prompt_tokens` | 各子题完整输入长度，按请求顺序 |
| `planning_ms`、`prefill_ms`、`branch_ms` | endpoint 阶段墙钟；包含对应线程池或引擎等待，不能当纯设备时间 |
| `prefill_cached_tokens`、`branch_cached_tokens` | 引擎最终 `num_cached_tokens`；没有字段或值时保留 `null` |
| `reuse` | 未尝试、已知不足、指标不可用或 `reported_sufficient` |

`reported_sufficient` 仅表示每个子题报告的命中数至少 C，不证明 GDN 数值正确，也不覆盖后续抢占重算。
失败请求的 `reuse` 可能保留 `not_attempted`，应优先看 `status`、阶段耗时和已收到的命中数组。
日志覆盖到子题清理完成；完整 HTTP 耗时仍以客户端测量为准，包括答案构造、序列化和返回。

## 5. 本地验证

使用仓库 `.venv/Scripts/python.exe`，Python 3.12；测试依赖包括 pytest 8.3.5、Ruff 0.14.0、Pydantic 2.14.1、FastAPI 0.143.1、TypeSafe SDK 0.7.2。
服务测试注入框架依赖，执行真实 `serving.py` 与真实规划/协议源码，使用可控假引擎；SDK 测试使用真实客户端和 MockTransport。

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  --confcutdir=tests/ut/entrypoints `
  --basetemp=.pytest_cache/shared-prefix-final `
  tests/ut/entrypoints/test_systemone_config.py `
  tests/ut/entrypoints/test_systemone_shared_prefix.py `
  tests/ut/entrypoints/test_plumb_protocol.py `
  tests/ut/entrypoints/test_plumb_serving.py `
  tests/ut/entrypoints/test_systemone_shared_prefix_benchmark.py
```

最终定向测试 **161 passed，无跳过**；格式检查结果见第 7 节。
覆盖空 LCP、相同 prompt、一个 prompt 是另一个前缀、边界两侧、阈值、能力未知/关闭、两种读出模式、无 salt、usage 和候选/答案顺序。
故障测试覆盖 producer 无完成输出、完成后抛异常、错误/abort、预填中取消、分支中取消与单分支失败，以及返回前完成兄弟清理。
新增注册窗口假引擎故意模拟“原生 close 后仍已注册”，验证 public abort 兜底；重复客户端 ID 的父组取消互相隔离。
一快一慢 abort 测试确认父任务等待全部清理，事件循环没有未消费异常；abort 失败仍保留原始错误或取消。

普通 pytest 入口在 `tests/ut/conftest.py` 导入 `vllm_ascend` 时因缺少 vLLM 失败；本次 CPU 测试通过 `--confcutdir` 隔离该硬件公共 fixture。
这些检查不等于完整仓库测试或 NPU 测试。

## 6. 服务器验证与执行命令

### 6.1 固定部署并验证边界

沿用 [Qwen 启动说明](../usage/systemone_token_readout.md)，记录实际包版本、Git SHA、导入路径、checkpoint/tokenizer revision、TP/DP 和完整参数。
源码参考的 vLLM tag 为 `v0.27.1`，本地 tag 对应 `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`；本地 `refs/vllm` HEAD 是另一个版本，核对时使用 `git show v0.27.1:路径`，不可混用。

在服务器同一个 Python 环境刷新安装并完整重启：

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -c 'import importlib.metadata as m, vllm, vllm_ascend; print(m.version("vllm"), m.version("vllm-ascend")); print(vllm.__file__, vllm_ascend.__file__)'
```

先使用共享关闭的配置记录实际物理块、混合缓存组、调度边界及原路径基线。
在独占测试实例用候选边界做 producer→多子题探针，确认所有缓存组完成、各子题命中、GDN 复制和数值一致性；探针通过后才能把该间隔称为 verified。
P1 没有自动完成这个硬件探针。首次配置检查只能排除已知不支持项，不能代替探针。

再合并上面的 `shared_prefix` JSON 至现有 `--additional-config`，保留 TP=4、DP=1、eager、同步、APC/chunked/align。
先逐项比较 `max-num-seqs=20/32` 和 `max-num-batched-tokens=2048/4096/8192/16384`；每个组合重新记录容量和边界，不能仅因额度够大就宣称 20 题能一轮执行。

### 6.2 配对 HTTP 测量

准备真实的 20-question JSON 文件，事先固定具有代表性的单题 ID。脚本会覆盖输入的 salt，以新 salt 隔离每次逻辑冷请求；APC 始终开启。

```bash
python benchmarks/scripts/benchmark_systemone_shared_prefix.py \
  --input /path/to/decision_20_questions.json \
  --single-question representative_question_id \
  --url http://127.0.0.1:8010/v1/systemone \
  --repeats 30 --seed 42 --timeout 300 --include-warm \
  --output /path/to/results/shared_prefix_enabled.json
```

脚本随机交错每对 1/20 题次序，计时包含内部 producer 和完整 HTTP body 返回。
冷组每个题数使用独立新 salt，热重复只复用对应冷请求的 salt。
报告保留每次响应、request ID、耗时、配对比、失败及超标重复号，给出成功样本 P50/P95/P99；HTTP 失败、缺答案和超时仍计入未通过分母。
每组都满足原比值且无失败时退出码为 0，否则为 1。

同一部署关闭 `shared_prefix.enabled` 后另跑一次并保存独立报告；比较两版绝对 T1/T20，同时保留逐题独立基线。
这份脚本只测固定输入在当前部署的 HTTP 比值，不自动测候选数值误差、实际 KV 重用、所有单题基线、混跑吞吐或故障资源释放。
小样本 P99 不代表可靠尾延迟估计，也不能用汇总百分位替代逐组目标。

### 6.3 必须追加的远程验收

| 验证项 | 通过依据 | 当前状态 |
|---|---|---|
| 缓存复用 | 冷 producer 后每题 `cached_tokens>=C`；scheduler/runner 实际新增 token 与预期一致，无额外重算 | `pending_remote` |
| 数值正确性 | 与每题独立完整 prompt 比较候选分数、概率和答案；容差来自该 dtype 基线波动，近似平局单列 | `pending_remote` |
| 混合状态隔离 | 重排、重复题和独立重跑结果一致；GDN 卷积/递推状态各分支私有可写 | `pending_remote` |
| 性能目标 | 16K/32K 及短 state、128/256/512 后缀，逐对 `T20/T1<=1.5`，报告绝对耗时和全部失败 | `pending_remote` |
| 故障资源 | producer/分支取消、单分支失败、OOM/淘汰后无引用泄漏、遗漏请求或跨 rank 死锁 | `pending_remote` |
| 压力与混跑 | 多父请求、Generation 混跑时记录排队、抢占/重算、缓存峰值和吞吐 | `pending_remote` |

无抢占冷组的新增输入 token 理论对照是 `C + sum(len(R_i)-C)`。
必须按调度步累计实际 token，不能只用 logical usage 或首次 cached tokens 推断全部设备工作量。
TP ranks 执行相同逻辑 tokens，不能把各 rank token 数相加当业务计算量。

## 7. 验证记录

| 检查 | 实际结果 |
|---|---|
| 五个定向测试文件 | `161 passed in 2.23s`，包括真实 TypeSafe SDK 解码，无跳过 |
| 8 个新增/修改 Python 文件的 Ruff check | 通过 |
| 相同文件 Ruff format --check | 通过 |
| 本次文件 codespell、typos、Markdownlint | 通过 |
| benchmark `--help` | 通过，CLI 可加载 |
| `git diff --check` | 通过 |
| 普通 pytest 入口 | 公共 conftest 导入失败：`ModuleNotFoundError: No module named 'vllm'` |
| `bash format.sh ci` 全库检查 | 已执行，未全部通过；详情如下 |
| NPU、实际 APC/GDN、完整端到端性能 | `pending_remote` |

全库格式检查的 Ruff、format、Markdownlint 等通过，但历史 `docs/report/1009` 的嵌入内容/提交号触发 codespell、typos，`build_demo.py` 的既有 `import re` 触发 forbidden-imports。
Windows 上 actionlint 被应用程序控制策略阻止（WinError 4551），部分 shell hook 找不到 `/bin/bash`，且 shellcheck 未安装。
未修改上述历史报告或扩大到工具链修复；本次全部文件另做定向检查。
这份记录不代表完整 CI 或 NPU 验收通过。
