# Plumb-4B：Ascend 原生 SystemOne 服务

本分支在 vLLM-Ascend `v0.27.1rc1` 上新增 `/v1/systemone` endpoint，配套 vLLM `v0.27.1`。
直接加载原始 Plumb BF16 权重，不需要 Kev 导出器、pointer head 或单独的 JevK5 推理进程。
每道 question 是一个原生生成请求，复用 vLLM 的并发调度、chunked prefill、prefix caching、取消和 DP 路由。

## 安装和启动

在已经装好 vLLM `0.27.1`、vLLM-Ascend `0.27.1rc1` 及其配套 CANN/torch-npu 的 A2 环境中：

```bash
cd /path/to/vllm-ascend
git fetch origin codex/plumb-4b-a2
git switch codex/plumb-4b-a2
git pull --ff-only
python -m pip install --no-deps --no-build-isolation -e .

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling,ascend_systemone
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve crh225/plumb-4b \
  --revision 1c5f4483addb049476ac33107796d217a1dc089d \
  --served-model-name plumb-4b \
  --runner generate --dtype bfloat16 \
  --enforce-eager --no-async-scheduling \
  --enable-prefix-caching --enable-chunked-prefill --mamba-cache-mode align \
  --max-model-len 65537 --max-num-batched-tokens 2048 --max-num-seqs 8 \
  --tensor-parallel-size 1 --max-logprobs 16 --logprobs-mode raw_logprobs \
  --host 0.0.0.0 --port 8009
```

也可把模型参数换成本地下载目录；目录中保留 `jevk5_config.json`、tokenizer 和 chat template。
temperature 从该文件读取，当前固定模型为 `2.07`；缺失时启动报错，不静默退回 `1.0`。
修改源码后完整重启 API server 和 workers。首次安装必须刷新 editable 元数据，否则新增 entrypoint 不会出现。
遗漏 `VLLM_PLUGINS` 中的 `ascend_systemone` 时，普通 vLLM 服务可能正常启动，但 `/v1/systemone` 返回 404。

该版本的兼容补丁补齐文本 M-RoPE，以及 `model.language_model.*` 到原生 backbone 的权重映射。
保留 tied embeddings 和 FP32 GDN recurrent state，沿用 vLLM 的 Qwen3.5 文本 RoPE 处理，不修改模型 JSON 或重导出权重。
上游文本生成配置会移除多模态 RoPE 的分段字段；文本三轴位置相同，此处兼容接口不改变原生配置处理。
若安装的是带开发后缀的 vLLM 源码包，先确认它确实对应 `v0.27.1`，再按仓库惯例设置 `VLLM_VERSION=0.27.1`；不要用该变量伪装其他版本。

`65537` 为完整输入预留 65536 tokens，再留一个引擎内部输出 token。
输入计数包含模板、state、instructions 和 options；超长直接报错，不截断。
这只是服务长度设置，不代表已经验证 Plumb 在 64K 上的决策质量。
`max-num-seqs` 是问题序列的执行上限，超出的请求进入原生等待队列；具体显存容量由引擎和输入长度决定。

## 请求

```bash
curl http://127.0.0.1:8009/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "plumb-4b",
    "state": "Refunds need a receipt and a purchase within 30 days. The customer bought 12 days ago and has no receipt.",
    "questions": {
      "refund": {
        "type": "noul",
        "instructions": "Is a refund permitted under the policy?"
      },
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {"billing": "Payments and refunds", "shipping": "Delivery"}
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["Low", "Medium", "High"]
      }
    }
  }'
```

多个客户端可以同时请求；单个请求的多道 question 也会并发提交，结果按输入顺序聚合。
不使用原始 `jevk5-serve` 的进程内串行锁，也不另加 HTTP 并发计数、默认 deadline 或 state 到 DP rank 的固定映射。

| 字段/类型 | 含义 |
|---|---|
| `state` | 公共证据，保留 JSON 原值，放在每道问题 prompt 的共同前缀 |
| `noul` | true/false，返回 true 的概率 |
| `choice` | 2–16 个不同选项，支持名称列表或名称到描述的对象 |
| `score` | 2–16 个有序等级，返回以 0 为起点的期望分数 |
| `confidence` | 最大选项概率，沿用 JevK5 定义，不四舍五入 |
| `priority` | 透传原生请求优先级，是否启用优先调度由 vLLM 配置决定 |
| `cache_salt` | 可选缓存隔离值；只有相同 salt 的请求可共享相应前缀 |

响应保留 `model`、`answers`、`usage` 和 `latency_ms`。
`score` 答案包含 TypeSafe SDK 必填的 `legend`，例如 `{"0": "Low", "1": "Medium", "2": "High"}`，与 `probabilities` 使用相同的等级键。
字符串、对象和数组 criteria 原样保留；直接 HTTP 请求中的数字、布尔和 null 等级描述在 legend 中转换成字符串。
已通过 `typesafe-sdk==0.7.2` 的客户端响应解析；SDK 将等级键转换成整数，可用 `result.scores["urgency"].legend[2]` 读取描述。
`usage.input_tokens` 为每道 question 完整 prompt 的 token 数之和，缓存命中不改变此逻辑计数。
`usage.output_tokens=0` 沿用开源决策 API 的业务含义；引擎内部实际执行一次首步采样，丢弃该 token，只读取候选原始 logprob。

## Prefix caching 和多卡

编码与 JevK5 相同：固定 system 指令，user JSON 中先 evidence，再 criterion 和 options，使用模型自身模板并关闭 thinking。
相同 state 的问题自然共享前缀；tokenizer、JSON 键序、内容或 cache salt 变化都会影响命中。
缓存、GDN state 和 eviction 都由原生引擎管理，没有 Kev 的 state-only lookup patch。
首次同时到达的冷请求不保证只计算一次公共前缀；缓存完成后才有可复用内容。

要使用单机四卡原生 DP，可把可见设备改为 `0,1,2,3`，并在启动命令中增加：

```bash
--data-parallel-size 4 --data-parallel-size-local 4
```

八卡时相应改为八个可见设备和两个 `8`。保持 TP=1 时，每个 DP 副本拥有完整模型和独立缓存。
服务不覆盖 vLLM 的负载路由，因此不保证相同 state 总被分配到同一个副本。
也可以按卡启动独立端口供外部负载均衡使用。

## 实现入口和当前边界

| 文件 | 职责 |
|---|---|
| `setup.py` | 注册 `vllm.endpoint_plugins` 下的 `ascend_systemone` |
| `vllm_ascend/entrypoints/systemone/protocol.py` | TypeSafe schema、官方 prompt、候选映射及概率/答案 |
| `vllm_ascend/entrypoints/systemone/serving.py` | 原生 `generate()`、候选 logprobs、异步聚合与 HTTP 生命周期 |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | `0.27.1` 文本 M-RoPE 和权重命名兼容 |

候选分数通过 `logprob_token_ids` 精确读取，而非全词表 top-k；计算 `softmax(candidate_logprobs / temperature)`。
全词表 log-softmax 的公共常数在候选内归一化时消掉，保留官方读出语义。
`temperature=0` 仅用于内部采样；评分使用采样处理前的原始 logprobs，再应用 checkpoint 的校准温度。
缺候选、子请求失败或不完整结果会报错，不返回伪造的均匀概率或部分答案。

第一版要求 V1、原始 logprobs，关闭 speculative decoding、量化和 `enable_reduce_sample`。
推荐先用上面的 eager 命令运行；本地检查不代表 NPU 精度、64K 容量或吞吐已通过。
NPU 联调仍为 `pending_remote`。无需准备额外评测流程，可先发送自己的并发请求与开源结果比较。
问题请提交到 [个人 fork 的 Issues](https://github.com/F00L42/vllm-ascend/issues)，附分支提交号、完整启动命令、报错日志及可复现请求。

参考：[固定模型](https://huggingface.co/crh225/plumb-4b/tree/1c5f4483addb049476ac33107796d217a1dc089d)、[JevK5 runtime](https://github.com/allebee/jevk5/blob/85238d7be5527370c43206fe54cd752eb3134c1b/jevk5/runtime.py)。
