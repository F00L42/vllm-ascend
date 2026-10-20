# SystemOne 下一轮实验：原生 APC、首题预热与完整 HTTP 延迟

日期：2026-10-20。实现基线：本分支上 `4d7786dbb` 之后的本次提交。

本轮先回答：**冷 HTTP 到达后，哪些题重复计算了长前缀，哪些题命中了同轮刚产生的缓存？**
随后在同一输入和启动配置下比较三种提交策略。代码和 CPU 回归已完成；本轮 NPU 数值、缓存和性能结果为 `pending_remote`。

沿用现有服务器、checkpoint、tokenizer、四卡 TP、8181 端口和输入文件，不安装新环境，不执行前置环境调查。
已经确认的实际 Attention/Mamba 块为 **1536**；本轮继续使用 **2048** 的 `max_num_batched_tokens`，其余启动参数沿用现有命令。
1536 来自 [Discussion #3 的服务器日志](https://github.com/F00L42/vllm-ascend/discussions/3)，它通过对齐检查并不替代这轮数值验收。

## 1. 这次改了什么

| 修改 | 用途 |
|---|---|
| 使用 `vllm.logger.logger` | 正常输出 INFO；不再依赖没有 handler 的 `vllm_ascend` logger |
| `systemone_config.diagnostics=true` | 返回未舍入概率、逐题 prompt 长度、缓存命中、盐摘要、请求 ID 和提交/完成时间 |
| `plan_request_ms / lcp_ms / executor_queue_ms` | 分开完整 prompt 规划、共享前缀规划和线程池排队；尚未细拆模板与分词 |
| 可选 `SystemOneTraceScheduler` | 在计数推进之前记录每步 `q_i`、`c_i`、首次 APC 命中、等待状态、抢占和总 token 预算 |
| `shared_prefix.strategy=first_question` | 先完成原输入中的首题，保存答案，然后提交其余题；没有额外 producer 或 LCP 扫描 |
| 启动、采集、分析脚本 | 自动归档版本和参数，验 salt、比较概率，并生成调度时间线 JSON 与 Markdown 表格 |

三种策略均使用原始完整 prompt、相同候选顺序和温度，APC 始终开启。默认仍是 `shared_prefix.enabled=false`。
`first_question` 保留现有 DP=1、同步、eager 和混合缓存能力检查；`min_shared_tokens` 只约束独立 producer。
首题失败或取消时不提交其余题；其余分支失败时取消并等待清理。这里只改变编排，不保证缓存驻留或加速。

## 2. 更新代码，然后沿用原启动命令

在原有 vLLM-Ascend checkout 和已激活的 Python 环境中执行：

```bash
git pull --ff-only origin codex/systemone-token-readout
python -m pip install --no-deps --no-build-isolation -e .
```

停止原服务，然后启动诊断版。**把原启动命令前面加上以下前缀，`--` 后接原来的完整 `vllm serve ...` 命令。**
实际模型目录、设备环境变量、served model name 和其余参数全部沿用；无需重新查找或提交环境信息。
使用与 `vllm` 相同虚拟环境中的 `python`，这样自动归档的包源码与实际进程一致。

```bash
python benchmarks/scripts/serve_systemone_apc.py \
  --strategy direct --trace-scheduler \
  --output-dir benchmarks/results/systemone_apc/direct-diagnostic -- \
  vllm serve
```

上面最后一行的 `vllm serve` 后面接**你现有命令的模型目录及全部参数**；这里不填猜测的 checkpoint 路径。
包装脚本会合并已有 `--additional-config`，只设置本轮 strategy、诊断和 1536 边界；诊断阶段额外选择观测 scheduler。
它不主动改变 APC、预算、并发上限、图模式、dtype 或设备环境。现有命令应仍为 APC 开启、同步 eager、DP=1。

服务输出同时显示在终端并保存为 `direct-diagnostic/server.log`；`manifest.json` 自动保存原命令、最终命令、两个包的 Git SHA、tracked diff、status 和源码位置。
每次使用新输出目录，避免覆盖已有启动证据。若 vLLM 来自没有 Git 元数据的 wheel，SHA 会显式为 null，后面的版本一致性检查不会把它当已确认。
status 中未跟踪的源码仅列名，未包含其文件内容；本轮提交代码后正常更新启动不需要依赖未跟踪补丁。

## 3. 先做 direct 小样本机制实验

服务就绪后，在另一个同环境终端直接执行：

```bash
bash benchmarks/scripts/run_systemone_apc_round.sh diagnostic direct
```

脚本直接读取现有输入：

```text
/opt/p00906181/260929_plumb/benchmark_inputs/decision_20_questions_16k.json
/opt/p00906181/260929_plumb/benchmark_inputs/decision_20_questions_32k.json
```

每种长度按顺序执行：

1. **salt 验证，3 组**：只取 `department`，发送新 A → 同一个 A → 新 B；读取真实 `cached_tokens`，要求 **0 → 正数 → 0**。缺字段、salt 摘要不一致或概率异常都失败。
2. **原生 APC 桥接，3 组**：先 3 题，再 20 题；HTTP 并发依次为 1、3。三题固定为 `department / refund / urgency`。每轮 cold 使用新 salt，同轮所有 HTTP 共享这个 salt；warm 再使用同一个 salt。
3. **数值检查，1 组**：每题独立冷执行两次记录基线波动；随后完整批量、逆序、重复一道题，按 question ID 比较全部未舍入候选概率和返回的 choice/score/noul。
4. **调度关联**：自动生成 `16k-trace.md/json` 和 `32k-trace.md/json`；请求缺 trace、日志损坏、chunk 缺失、命中字段未知或缺 scheduler runtime 均报告失败。发生抢占时保存轨迹并要求分析重算，不能静默算作完整验收。

`atol=1e-5、rtol=1e-4` 是本轮在测量前固定的候选阈值，逐项使用 `abs(test-ref) <= atol + rtol*abs(ref)`；不是已被证明适合此 BF16 模型的标准。
独立重复本身超阈值时停止后续结论，先检查误差和近似平局；需要更改阈值时，明示新的验收条件并将所有策略重新跑在新目录，脚本不会按观测误差自行放宽。
返回值发生离散变化会单独报告，即使最大概率误差较小也不会悄悄忽略。

三题使用的是当前文件的相同 state 与原题定义。如果它们与 [Discussion #2](https://github.com/F00L42/vllm-ascend/discussions/2) 当时输入不同，只能称“当前三题桥接”，不能凭 16K/32K 标签声称精确复现历史输入。
每份结果自动记录原输入文件 SHA256 和每题实际 prompt token 长度，避免把长度标签当实际 token 数。

## 4. 用表格判断原生 APC 的复用过程

打开 `benchmarks/results/systemone_apc/direct-diagnostic/16k-trace.md`。按同一父请求查看：

| 首次命中与调度顺序 | 可以支持的解释 |
|---|---|
| Q0 首步命中 0；Q1/Q2 在较晚步骤开始且命中长前缀 | 后调度的题复用了已经产生的前缀缓存 |
| 三题首次命中均为 0，前缀计算区间重叠 | 多题在可用前缀缓存产生前进入了计算 |
| 首次命中相同，但完整 HTTP 延迟不同 | 继续看规划、排队、调度步数和分支时间，不能仅由命中数推断设备耗时 |

`timeline` 包含 `step_id / computed_before / num_scheduled_tokens / num_cached_tokens / status / queue / waiting_reason` 和原日志行号。
**`computed_before=c_i` 包含自身已经完成的 chunks，不能累计成 APC 命中。** `num_cached_tokens` 是首次 prefill 的缓存统计，之后可能为空；它不是每次恢复时的新命中计数。
`scheduled_new_tokens=sum(q_i)` 是调度新增工作量，含重算；只有完整成功请求和完整轨迹才用于比较。它不包含 worker padding，不是设备执行时间或 FLOPs。
等待原因不能从公开队列状态唯一确定时记录 `unknown`；预算耗尽、序列额度占满另列事实，避免伪造因果。

endpoint 提交/完成时间分别包住 native generator 和清理；不等于 scheduler admission/finish。
公开子请求 ID 与引擎内部 ID 按 vLLM 0.27.1 的“相同 ID 或追加 8 位 UUID”规则精确关联，避免 Q0 错配 Q01。
该子类在 `_update_after_schedule()` 推进计数前读取，随后调用原实现；未复制或替换调度算法。

## 5. 再比较 producer 和 first_question

完成 direct 后停止该服务。后续不必再次输入启动参数，直接复用首次自动保存的原命令。
在服务终端依次使用以下命令；每次等该策略采集完成并停止服务，再启动下一个：

```bash
python benchmarks/scripts/serve_systemone_apc.py \
  --strategy producer --trace-scheduler \
  --from-manifest benchmarks/results/systemone_apc/direct-diagnostic/manifest.json \
  --output-dir benchmarks/results/systemone_apc/producer-diagnostic
```

对应采集终端：

```bash
bash benchmarks/scripts/run_systemone_apc_round.sh diagnostic producer
```

然后服务终端：

```bash
python benchmarks/scripts/serve_systemone_apc.py \
  --strategy first_question --trace-scheduler \
  --from-manifest benchmarks/results/systemone_apc/direct-diagnostic/manifest.json \
  --output-dir benchmarks/results/systemone_apc/first_question-diagnostic
```

对应采集终端：

```bash
bash benchmarks/scripts/run_systemone_apc_round.sh diagnostic first_question
```

脚本要求响应的实际策略与预期匹配；配置回退不能混进“producer/first_question 已执行”的结果。
单题仍走 direct，这是设计行为。`first_question` 的 `common_tokens=0` 表示未扫描 LCP；其 `first_question_ms` 与剩余分支 `branch_ms` 分开。

三种策略完成后，自动比较版本、运行配置、prompt 长度及数值：

```bash
(
set -e
for length in 16k 32k; do
  python benchmarks/scripts/analyze_systemone_apc.py --compare \
    --results benchmarks/results/systemone_apc/direct-diagnostic/$length-correctness.json \
              benchmarks/results/systemone_apc/producer-diagnostic/$length-correctness.json \
              benchmarks/results/systemone_apc/first_question-diagnostic/$length-correctness.json \
    --manifests benchmarks/results/systemone_apc/direct-diagnostic/manifest.json \
                benchmarks/results/systemone_apc/producer-diagnostic/manifest.json \
                benchmarks/results/systemone_apc/first_question-diagnostic/manifest.json \
    --output benchmarks/results/systemone_apc/$length-strategy-correctness.json
done
)
```

这轮只比较三种编排。先不增加 state-only producer、不拼接分别编码的 state token，也不同时调 token 预算与序列上限。

## 6. 机制与数值通过后，关闭逐步日志跑 30 组

逐步日志会产生额外 CPU/I/O 开销，因此只用第 3～5 节解释机制，不用那里的耗时宣称正式收益。
依次重启三种策略，**省略 `--trace-scheduler`**；endpoint 诊断保持相同，概率仍逐次验证。以 direct 为例：

```bash
python benchmarks/scripts/serve_systemone_apc.py \
  --strategy direct \
  --from-manifest benchmarks/results/systemone_apc/direct-diagnostic/manifest.json \
  --output-dir benchmarks/results/systemone_apc/direct-performance
```

对应采集终端：

```bash
bash benchmarks/scripts/run_systemone_apc_round.sh performance direct
```

然后将上面两条命令中的 `direct` 分别换为 `producer`、`first_question`（`--from-manifest` 始终保留首次 direct 文件），依次执行。
保持相同服务器负载。需要验证时段影响时，用新输出根目录按相反策略顺序重跑；不要把三种串行时段的均值差直接当随机交错因果估计。

性能阶段先自动 warm up 并验 salt，再采集 30 组随机交错的 1/20 题完整 HTTP 冷热配对。
冷单题和冷20题分别使用新 salt；warm 紧随对应 cold，沿用同一 salt。独立 producer 与首题屏障均包含在 HTTP 计时内。
冷多题内部后续题可以命中本轮缓存，不要求所有分支命中均为0。

每份 `*-performance.json` 分开保存：

- `data_passed`：请求完整、策略正确、缓存与数值校验通过。
- `target_passed`：逐对 `T20/T1 <= 1.5`。
- `summary`：冷/热分别的完整 HTTP P50/P95/P99、逐对比值、失败次数和失败 repeat。
- 原始 `pairs`：所有样本，包括失败响应和完整概率；另有逐条 flush 的 `.requests.jsonl`。

Python 脚本退出码 **0=全通过、1=数据有效但性能目标未达、2=数据或正确性失败**。
批量 shell 在退出码1时继续另一种长度并保留结果；退出码2时停止。shell执行结束本身不等于性能通过，以 JSON 的两个字段为准。
正式比较三策略的绝对 `many` HTTP 延迟及尾延迟，同时看目标比值；不能只改善分母或只报告成功子集。
调度新增工作量来自对应小样本 trace，正式无 trace 阶段不伪造每步工作量。

三策略正式采集完成后执行以下比较。它再次核对正式运行的版本、参数、输入、逐题长度和概率，输出绝对 HTTP 分位数及相对 direct 的变化；`target_passed=false` 会保留为未达标，不混同数据无效：

```bash
(
set -e
for length in 16k 32k; do
  python benchmarks/scripts/analyze_systemone_apc.py --compare \
    --results benchmarks/results/systemone_apc/direct-performance/$length-performance.json \
              benchmarks/results/systemone_apc/producer-performance/$length-performance.json \
              benchmarks/results/systemone_apc/first_question-performance/$length-performance.json \
    --manifests benchmarks/results/systemone_apc/direct-performance/manifest.json \
                benchmarks/results/systemone_apc/producer-performance/manifest.json \
                benchmarks/results/systemone_apc/first_question-performance/manifest.json \
    --output benchmarks/results/systemone_apc/$length-strategy-performance.json
done
)
```

## 7. 本地验证与证据边界

本地 275 项相关 CPU 测试通过：服务/协议/采集与启动 229 项、调度观察 19 项、证据分析 27 项。
覆盖服务控制流、原始参数/输出保持、取消与异常清理、logger、诊断 schema、盐和概率门槛、调度字段及轨迹缺失检查；包含三策略 × 单/多题的实际服务响应与采集脚本合同测试。
CPU stub 和模拟 HTTP 不证明 NPU 状态数值或性能。实际收益、GDN 状态复用及 30 组性能均由上述服务器脚本验收，当前为 `pending_remote`。

仓库全量 `bash format.sh ci` 已尝试；现有报告文件的拼写/禁用 import 检查，以及 Windows 缺少 `/bin/bash`、shellcheck 的 hook 尚有失败。
本次变更独立执行 Ruff、Markdown 和对应回归检查；全量检查不能标成通过。无需为本轮实验修改旧报告、重新配置服务器或重做环境调查。

相关实现：[服务](../../vllm_ascend/entrypoints/systemone/serving.py)、[诊断 scheduler](../../vllm_ascend/core/systemone_trace_scheduler.py)、[采集](../../benchmarks/scripts/benchmark_systemone_apc.py)、[启动](../../benchmarks/scripts/serve_systemone_apc.py)、[分析](../../benchmarks/scripts/analyze_systemone_apc.py)。
方法来源：[讨论中的实验结论](https://github.com/F00L42/vllm-ascend/discussions/3)、[vLLM 0.27.1 scheduler](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py)、[请求 ID 分配](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/engine/input_processor.py)。
