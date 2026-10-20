# SystemOne：迁移原生答案 token 概率读出

同一套代码可分别启动 Plumb-4B 和增训后的 Qwen3.6-27B，每个 vLLM 实例加载一个 checkpoint。
两种模式共用 `ascend_systemone` entrypoint、`/v1/systemone` 和 TypeSafe SDK 响应格式。
这里迁移的是读出方法：使用所加载模型自己的 tokenizer、chat template 和 LM head，读取候选选项的概率。
不额外训练，不对新模型的决策能力或校准效果作假设。

| 模式 | 适用模型 | 读出温度 |
|---|---|---|
| `plumb`（默认） | 原始 Plumb-4B | 必须读取 checkpoint 的 `jevk5_config.json`，不接受覆盖 |
| `answer_tokens` | 结构与官方 Qwen3.6-27B 一致的增训 checkpoint | 默认 `1.0`，可显式设置有限正数 |

Plumb 的安装、启动和请求说明见 [Plumb SystemOne](plumb_systemone.md)。
以下命令用于增训后的 Qwen3.6-27B，沿用 vLLM-Ascend `v0.27.1rc1` 和 vLLM `v0.27.1` 基线。

## 安装和启动 Qwen3.6-27B

在已经安装配套 CANN、torch-npu、vLLM 和 vLLM-Ascend 的 A2 服务器中：

```bash
cd /path/to/vllm-ascend
git fetch origin codex/systemone-token-readout
git switch codex/systemone-token-readout
git pull --ff-only
python -m pip install --no-deps --no-build-isolation -e .

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling,ascend_systemone
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

vllm serve /path/to/qwen3.6-27b-checkpoint \
  --served-model-name qwen3.6-27b-systemone \
  --runner generate --dtype bfloat16 \
  --enforce-eager --no-async-scheduling \
  --enable-prefix-caching --enable-chunked-prefill --mamba-cache-mode align \
  --max-model-len 65537 --max-num-batched-tokens 2048 --max-num-seqs 8 \
  --tensor-parallel-size 4 --max-logprobs 16 --logprobs-mode raw_logprobs \
  --additional-config '{"systemone_config":{"mode":"answer_tokens"}}' \
  --host 0.0.0.0 --port 8010
```

把 `/path/to/qwen3.6-27b-checkpoint` 换成实际增训 checkpoint，保留其 tokenizer 和 chat template。
模型权重由原生 vLLM 模型实现加载，包括 checkpoint 自己的独立 LM head；不使用 Plumb 的 tied embeddings 或校准文件。
首次部署要刷新 editable 安装元数据，修改代码后完整重启 API server 和 workers。
`VLLM_PLUGINS` 必须包含 `ascend_systemone`，否则 `/v1/systemone` 不会挂载。

这是四卡 TP=4 的初始配置。`65537` 为完整输入预留 65536 tokens，再留一个引擎内部输出 token。
输入包含模板、state、instructions 和 options，超长直接报错，不截断。
`max-num-seqs=8` 是并行执行的问题序列上限，超出后进入引擎队列；不代表已验证八条 64K 请求能同时驻留显存。
本地检查没有验证 NPU 执行、显存容量或吞吐，服务器运行仍为 `pending_remote`。

如需显式设置温度，把配置替换为：

```bash
--additional-config '{"systemone_config":{"mode":"answer_tokens","temperature":1.0}}'
```

temperature 只控制候选分布的归一化，默认 `1.0` 不代表完成过校准。
配置在服务启动时固定；请求不覆盖温度。若已有其他 `--additional-config` 设置，将 `systemone_config` 合并进同一个 JSON 对象。

## 请求和响应

请求可省略 `model`，此时使用该实例的第一个 served model name。
也可显式传入 `qwen3.6-27b-systemone`；名称必须匹配启动参数，不能通过请求切换 checkpoint。

```bash
curl http://127.0.0.1:8010/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "退款需要购买凭证，且购买时间不超过30天。客户12天前购买，但没有凭证。",
    "questions": {
      "refund": {
        "type": "noul",
        "instructions": "按这条规则，是否允许退款？"
      },
      "team": {
        "type": "choice",
        "instructions": "这个请求应该交给哪个团队？",
        "criteria": {"billing": "付款和退款", "shipping": "物流和配送"}
      },
      "urgency": {
        "type": "score",
        "instructions": "这个请求有多紧急？",
        "criteria": ["低", "中", "高"]
      }
    }
  }'
```

`noul` 返回 true 的概率；`choice` 返回选项分布和最大概率对应的选项；`score` 返回等级分布及从 0 开始的期望分数。
`choice` 和 `score` 支持 2–16 个选项，`score.legend` 保留等级键与描述，满足现有 TypeSafe SDK 的解析要求。
每道 question 并发提交到原生引擎，即使乱序完成，响应仍按请求中的问题顺序聚合。

响应继续使用 `model`、`answers`、`usage` 和 `latency_ms`。
`usage.input_tokens` 是每道问题完整 prompt 的 token 数之和，缓存命中不改变这个逻辑计数。
`usage.output_tokens=0` 是决策 API 的业务计数；引擎内部执行一次首步采样，丢弃生成 token，只取候选分数。

## 读出流程和 prefix caching

沿用 JevK5 的提示内容：固定 system 指令，user JSON 依次包含 evidence、criterion 和 options。
用当前 checkpoint 的 chat template 渲染，并设置 `enable_thinking=false`，直接在答案起始位置读取 A–P 的候选分数。
服务启动时检查这些字母在当前 tokenizer 下均为不同的单 token；不兼容时启动报错。

每道问题执行原生 `generate(max_tokens=1)`，通过 `logprob_token_ids` 精确取得候选的原始 logprob。
随后计算 `softmax(candidate_logprobs / temperature)`；全词表 log-softmax 的公共常数在候选归一化时抵消。
内部采样使用的 `temperature=0` 不改变这里读取的原始分数。
缺少候选分数、子请求失败或结果不完整时返回错误，不构造均匀分布或部分答案。

相同 state 的问题可共享模板和证据前缀，复用原生 prefix caching、GDN state、缓存淘汰和请求取消。
命中依赖完整 token 前缀及可选的 `cache_salt`；第一次同时到达的冷请求不保证只计算一次公共前缀。
服务不增加 state 到 DP rank 的固定映射，DP 副本的缓存由各自引擎维护。

阶段一新增可选的共享前缀编排：在 DP=1 和已验证的混合状态边界下，先完成一次公共前缀预填，再提交所有原始完整 prompt。
通过 `systemone_config.shared_prefix` 配置，默认关闭；能力不足时记录原因并沿用原路径。
配置、验证命令和当前 `pending_remote` 项见 [阶段一执行记录](../design/systemone_shared_prefix_phase1_execution.md)。
该编排建立提交顺序；缓存留存、真实命中和时延仍须实测。

下一轮原生 APC 观察、salt 隔离及三种策略的可执行脚本见
[APC 实验方案](../design/systemone_apc_next_experiment.md)。
其中 `shared_prefix.strategy="first_question"` 在启用并通过能力检查时先完成首题，再执行其余题；默认策略仍为 `producer`。
`systemone_config.diagnostics=true` 显式增加响应 `diagnostics` 字段，提供完整候选概率、运行配置、逐题缓存和时序；默认关闭。

当前范围为 V1 runner、BF16、原始 logprobs，关闭量化、speculative decoding 和 `enable_reduce_sample`。
启用 prefix caching 时要求 chunked prefill 和 `mamba-cache-mode=align`。
无需新增 entrypoint、独立读出进程或自定义 LM head。

## 分别运行两个模型

同一个 checkout 可运行任一种模式。停止当前实例后，可在同一组设备上启动另一模型。
若需要两种模型同时在线，启动两个独立实例并分配不同端口和设备。
例如八卡服务器上，Qwen3.6-27B 使用 `0,1,2,3` 和端口 `8010`；Plumb 使用设备 `4` 和端口 `8009`，按其文档启动。
客户端根据模型选择对应服务地址，两边都使用 `/v1/systemone`；一次请求由一个实例的一个 checkpoint 处理。

服务器问题可提交到 [个人 fork 的 Issues](https://github.com/F00L42/vllm-ascend/issues)，附提交号、启动命令、日志和可复现请求。
当前目标只要求返回有限、归一化且 SDK 可解析的候选概率，不额外建立决策质量或校准评测流程。

参考：[官方 Qwen3.6-27B 配置](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json)、[官方 chat template](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/chat_template.jinja)、[基线 Ascend 部署说明](https://docs.vllm.ai/projects/ascend/en/v0.27.1rc1/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html)。
