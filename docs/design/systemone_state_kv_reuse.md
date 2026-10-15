# SystemOne state KV 充分复用：边界、观测与后续实现

日期：2026-10-15。关联：[原设计](systemone_shared_prefix_latency.md)、[阶段一执行文档](systemone_shared_prefix_phase1_execution.md)。

本文基于当前 P1 编排代码、本地 Ascend 源码及官方 `v0.27.1` / `v0.27.1rc1` 资料。当前未实现 P2 细粒度复用或 P3 引擎分组管理；本机未安装 vLLM，未执行 NPU 推理。所有实际混合缓存命中、数值一致性、容量和时延结论均为 `pending_remote`。

## 1. 当前能保证什么

P1 建立 producer 完成屏障：按原完整 prompt token IDs 求 LCP，选择已声明验证的 C，消费内部 `generate(max_tokens=1)` 到正常完成并关闭生成器，再提交各题原始 prompt。单题、配置关闭或能力不足时回到原路径。预填失败不扇出，分支失败或父请求取消时取消并等待其他分支关闭。

这些是编排语义，不是缓存驻留保证。producer 完成后，请求引用可以释放；其他流量仍可能淘汰缓存。endpoint 不持有设备页引用，不设置 `num_computed_tokens`，也不证明每个 worker 都保存了完整混合状态。

当前 `additional_config` 中的 `systemone_config.shared_prefix` 默认关闭；显式开启还需 `verified_boundary_tokens`。该值是运维对当前部署边界能力的声明，不是代码自动完成的设备验证。`min_shared_tokens` 是可选阈值；它不证明采用共享路径一定更快。配置入口为 [config.py 的 `load_readout_config()`](../../vllm_ascend/entrypoints/systemone/config.py)。

当前条件为 APC、chunked prefill、Mamba `align`、DP=1、无外部/混合 DP 路由、无 CP、无 KV transfer、eager、同步调度，以及默认稠密留存。能力检查通过只表示可见配置满足限制。

代码定位：[shared_prefix.py](../../vllm_ascend/entrypoints/systemone/shared_prefix.py) 的 `inspect_shared_prefix_capability()` 检查能力，`plan_shared_prefix()` 规划 C；[serving.py](../../vllm_ascend/entrypoints/systemone/serving.py) 的 `SystemOneService.__init__()` 读取留存设置，`_execute()` 建立屏障，`_generate()` 保留 salt/priority/trace，`_prefill()`、`_consume_question()`、`_execute_questions()` 执行及清理。

当前 `_request_stream()` 在异常及取消时关闭原生生成器并显式调用公共 `engine.abort()`，覆盖请求注册后、collector 返回前的取消窗口。父组内部 ID 带独立 UUID，避免相同客户端 request header 互相 abort；`_execute_questions()` 用 `as_completed()` 等待分支，再由单一 owner 取消并等待清理，避免二次 cancel 打断慢 abort。这些本地控制流仍需实际 vLLM/worker 验证注册竞态及资源回收。

## 2. 三种边界必须分别确认

| 边界 | 含义 | 不能由什么推断 |
|---|---|---|
| 物理块 B | 一页容纳的 token 数或混合状态布局单位 | CLI `block-size` 原值 |
| hash 单位 h | 最细 prefix-cache 匹配位置 | 该位置已有 GDN 快照 |
| 有效状态终点 C | Attention KV、卷积状态和递推状态在同一位置完整可用 | hash 已登记或 scheduler 已分配页 |

本地 [patch_mamba_config.py](../../vllm_ascend/patch/platform/patch_mamba_config.py) 的 `verify_and_update_config()` 根据 SSM 页大小调整真实 Attention `block_size`，并设置 `mamba_block_size`。官方部署说明也提示混合缓存有效块可能达到 2048 tokens。[U1]

当前 P1 要求已声明边界同时为可见的最终 Attention/Mamba 物理块大小的整数倍。部署时还必须核对 worker 初始化后的实际 `kv_cache_groups`、scheduler/hash 单位，不能只看 API 进程日志。若配置尚未解析或与 worker 不一致，先解决能力记录，再启用共享。

`prefix_match_unit` 仅控制匹配；官方说明明确它不控制状态保存频率。[U2] 当前不直接用 h=32/128 选择 C。保守边界为 G 时：

\[
C=G\left\lfloor\frac{\min(L,\min_i(|R_i|-1))}{G}\right\rfloor
\]

这里 L 是完整 prompt IDs 的 LCP 长度。保留消费者至少一个新 token，是因为 vLLM 查找最多复用 `prompt_length-1`。[U3] 不追加任意 token，不分别编码 state 和问题后拼接，不改变模型输入来凑边界。

## 3. 复用对象是完整混合状态

Full Attention 公共完整页可以引用共享；分支写公共尾页前要有私有页或可靠 CoW。GDN 的卷积状态和递推状态需要每个活动分支独立可写，不能让 20 题更新同一个 state slot。

官方混合协调器返回各缓存组一致认可的终点；Attention 命中更长、GDN 缺失时，统一命中仍会缩短。细粒度混合命中还有 `align` 和无上下文并行的限制。[U4]

本地 [model_runner_v1.py](../../vllm_ascend/worker/model_runner_v1.py) 的 `NPUModelRunner._update_states()` 调父类；官方父类会处理 `kv_cache_block_copies`，其 helper 对 backing storage 按页复制。[U5] 因而不能因 Ascend 文件没有显式 CoW 分支就认定它不存在；继承路径在 NPU、实际页布局和取消时序上的正确性仍需验证。

本地 `NPUModelRunner.execute_model()` 准备 Mamba copy，在 forward 前执行复制或等待逐层复制结束。[patch_mamba_utils.py](../../vllm_ascend/patch/worker/patch_mamba_utils.py) 的 `preprocess_mamba()` 推导源/目标状态列并准备卷积和递推复制；[gdn.py](../../vllm_ascend/ops/gdn.py) 的 `_forward_core()` 读取初始递推状态并写回最终状态。

默认稠密留存由上游中央 `vllm.envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL=None` 表示。0 也属于自定义稀疏策略。P1 从中央 envs 读取，兼容检查 cache config 字段，并对所有非 None 值回退。无需新增 Ascend 环境变量。稀疏模式后续必须显式证明 C 被保留；当前 producer 在 C 结束，其 `prompt_length-1` 回放边界可能落在更早的块。[U6]

## 4. 三层观测才能证明少算了 state

| 层次 | 当前/后续数据 | 解释边界 |
|---|---|---|
| endpoint | 已记录 L、C、残余公共尾部、规划/预填/分支耗时、每题 cached tokens、回退原因 | `reported_sufficient` 仅表示引擎报告值达到 C |
| scheduler | 后续按 request ID 逐轮记录 scheduled tokens、computed 起点、抢占/恢复、DP rank | 重建实际 token 执行量，不能只看首次命中 |
| worker/device | 后续记录 forward 有效行数、各缓存组页/状态索引、复制次数/字节、占用与淘汰 | 验证完整状态、隔离、复制顺序及设备工作 |

当前日志字段位于 `SystemOneService._execute()`；每题命中在 `_consume_question()` 从最终 `RequestOutput.num_cached_tokens` 取得，缺字段保留 None。`reuse=unavailable` 不能解释为没有命中或已经命中。该字段可从上游输出通路追溯，但记录初始命中不能覆盖后续抢占重算。[U7]

冷组无额外重算时，理论新增 token 工作量为：

\[
W=C+\sum_i(|R_i|-C)
\]

以 scheduler 实际累加值对照 W，解释对齐、抢占或重复预填造成的差额。`usage.input_tokens` 始终是完整 prompt 的逻辑总和；TP 各 rank 的相同逻辑 token 也不能相加当业务 token 数。最终设备 profiler 还需计入 padding、复制、Attention、GDN、head 和通信；W 不是 E2E 时延公式。

复用终点增加 Δ，N 题冷组理论少算 `(N-1)Δ` 个新增 token。20 题多复用 128 tokens 对应 2432 tokens；是否值得增加 CoW 开销，必须由绝对 E2E 实测决定。

## 5. 保住跨 HTTP 与组内复用

**稳定身份。** 当前沿用请求 `cache_salt`，producer 与子题完全一致。生产环境每个 HTTP 随机生成新 salt 会阻断跨 HTTP 热复用；salt 应沿用既有租户/缓存隔离语义。测试才为每个冷样本使用新 salt。实际键还受 token 前缀、LoRA 等身份影响。[U8]

**DP 亲和。** P1 限定 DP=1。扩展 DP 前，需要用安装版本真实支持的路由接口将 producer 与全部分支绑定同一引擎；外部负载均衡的偶然命中不算保证。记录实际 rank，再通过故意打散路由的对照确认原因。当前没有实现父组路由 API。

**引用与准入。** 当压力下命中下降时，再增加引擎父组：在 producer 释放前取得公共页及完整 GDN 快照引用；复制源保持到设备复制完成，分支退出后可靠释放。只持有 Python request ID 或设置 TTL 不能防淘汰。组容量需要公共 KV、私有尾页、活动分支 GDN、后缀 KV、复制临时区和激活峰值。按实际可用页准入，必要时有限分批或拒绝，不能靠复制 20 份完整前缀保正确。

**细粒度。** 若稳定命中后残余公共尾部仍占主要时延，再依次验证 h=128、32。必须同时覆盖 producer 尾部注册、统一混合命中、Attention CoW、GDN 初始状态、源/目标留存及取消。官方存在部分块路径不等于当前部署验证通过。[U2][U4][U5]

**同 state single-flight。** 若多个父请求同一 state 同时冷到达，可研究按实际缓存身份和 token 前缀合并 producer。该能力尚未实现。共享 producer 必须独立于任一调用者：一个 waiter 取消只移除自身，全部 waiter 退出后才允许取消 producer；生产失败传播给所有等待者，失败项不能永久占键。就绪信号与设备资源引用应由同一生命周期管理，不能让 future 已完成但缓存已淘汰。

## 6. 后续优先级由退化证据决定

| 顺序/触发条件 | 实施动作 | 退出条件 |
|---|---|---|
| 先完成 P0/P1 远程证据 | 冻结版本/输入，核对最终块和缓存组，采集三层轨迹 | 独占 DP=1 下每题命中 C，新增工作量可解释，数值一致 |
| 压力下命中下降 | 公共快照 pin、父组准入和有限分批 | 指定负载下无意外前缀重算，取消后资源归还 |
| 准备扩展 DP | 父组亲和路由 | producer 与所有分支实际 rank 一致 |
| 自定义稀疏留存 | 显式保留 C 和所需引用 | 稀疏模式仍命中 C，源生命周期正确 |
| 残余公共尾部成本明显 | P2 细粒度 CoW + GDN 全链路 | 正确性和容量通过，绝对 E2E 改善 |
| 多父请求重复冷预填明显 | single-flight 与 waiter 所有权 | 只预填一次，各种取消组合无悬挂/误 abort |
| 已充分复用仍超时 | 批量预算、序列额度、复制/head/CPU 优化 | profiler 对应占比下降，E2E 和容量同时通过 |

DP=1 独占环境没有淘汰问题时，不必先实现整套 P3。P1/P2 在目标范围通过即可交付；公平调度、图和异步路径分别验证。

## 7. 可执行的远程验收

1. 保存 server/worker 包版本、Git SHA、源码导入路径、checkpoint/tokenizer/dtype、完整启动参数与实际缓存组。开启 P1 前，先验证声明的每个 G 边界可保存完整混合状态。
2. 用 producer 长度 G、2G，再覆盖 G 前后和块内终点；原路径与共享路径分别执行相同完整 prompts，保留候选分数/概率/答案、问题顺序和全部 trace。
3. 每组逻辑冷缓存使用新 salt，组内共享 salt；APC 始终开启。记录 producer 在计时范围内，所有分支 cached tokens ≥ C；否则判复用不足，禁止盲目重预填掩盖问题。
4. 对 scheduler 逐轮求和，核对 W；输出每题缺口、抢占和重算。用 worker 页表、状态索引与 copy trace 证明公共 Attention 页共享、GDN 写状态隔离及复制先于读取。
5. 覆盖 16K/32K 和短 state，后缀 128/256/512，问题数 1/3/10/20；加入重排、同题重复、不同候选数、空 LCP、一个 prompt 是另一个前缀和长度上限。
6. 故障覆盖预填取消、分支取消、单题失败、内存不足、淘汰压力、复制中取消。专测注册期取消：在原生请求已登记、collector 尚未返回时取消父请求，确认公共 abort 找到相应 producer/分支，清理未被第二次 cancel 打断；并发发送相同客户端 request header，确认互不误 abort。检查任务/缓存引用回到基线，无答案污染或跨 rank 卡死。这些场景均为 `pending_remote`；通过后再加多父请求与 Generation 混跑。
7. 固定代表单题，随机交错 1/20 题，保存完整 HTTP E2E。逐对检验 `T20/T1 <= 1.5`，报告绝对时延、分布、超标和失败。CPU 测试或单个 HTTP 200 均不能替代这些验收。

当前 HTTP 配对采集脚本可运行：

```bash
python benchmarks/scripts/benchmark_systemone_shared_prefix.py \
  --url http://127.0.0.1:8010/v1/systemone \
  --input request_20.json --single-question representative_question_id \
  --repeats 20 --include-warm --output artifacts/shared_prefix_http.json
```

脚本只采集 HTTP 配对及冷热记录，不自动证明数值等价、设备计算量或混合快照正确。分别在原配置与 P1 配置采集，并关联 endpoint、scheduler、worker 证据。[脚本](../../benchmarks/scripts/benchmark_systemone_shared_prefix.py)。

## 8. 固定官方来源

以下是 2026-10-15 核对的固定版本引用；部署仍需记录实际安装 commit。

| 标记 | 来源与定位 |
|---|---|
| U1 | [Ascend v0.27.1rc1 Qwen3.5/3.6 部署](https://docs.vllm.ai/projects/ascend/en/v0.27.1rc1/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html)：有效大块、批量额度和 HBM 取舍 |
| U2 | [v0.27.1 CacheConfig，53–61、133–139 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/config/cache.py#L53-L61)：匹配单位与 align 语义 |
| U3 | [KVCacheManager，209–241 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_manager.py#L209-L241)：消费者查找上限 |
| U4 | [HybridKVCacheCoordinator，543–559、639–759 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_coordinator.py#L543-L559)：细粒度条件与一致终点 |
| U5 | [GPU runner，1130–1141 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/gpu_model_runner.py#L1130-L1141)、[copy helper，483–516 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/utils.py#L483-L516)：CoW 执行路径 |
| U6 | [中央 envs，1088–1097 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/envs.py#L1088-L1097)、[Coordinator，112–118 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_coordinator.py#L112-L118)、[Mamba 留存及部分尾部，1260–1310、1561–1625 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/single_type_kv_cache_manager.py#L1260-L1310)：稠密、稀疏与有效保存位置 |
| U7 | [RequestOutput，129–140 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/outputs.py#L129-L140)、[输出通路，594–602 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/engine/output_processor.py#L594-L602)、[首次预填统计，782–788 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py#L782-L788) |
| U8 | [hash 身份，508–566 行](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_utils.py#L508-L566)：salt、token 前缀及其他键 |
