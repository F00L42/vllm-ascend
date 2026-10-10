# Jev-like 工程能力进展报告

2026 年 10 月 9 日初稿 · 2026 年 10 月 10 日补充性能规格

本阶段完成了两条 Jev-like 概率读出路线在 vLLM-Ascend 中的原生适配：Kev 使用专用 Pointer Head，Plumb 使用模型原生 LM head 的答案 token 分数。两条路线都接入 `/v1/systemone`，围绕 TypeSafe SDK 的请求与响应契约提供结构化决策服务；其中 Plumb 路线进一步抽象为可用于兼容 Qwen checkpoint 的通用读出能力。

根据服务器验证反馈，当前 Qwen-27B 已能够在同一个 vLLM 实例、同一份模型权重上提供 Decision 与 Generation。下一阶段的重点是恢复并保护 Generation 的高性能执行路径，同时控制 Decision 输入计算、候选读出和混合调度带来的开销。

当前服务器反馈的 Decision 单次推理用时：Qwen3.5-4B 主干的 Plumb-4B / Kev-4B 约 **300 ms**，Qwen3.6-27B 主干模型约 **1,000 ms**，与相应模型的 vLLM Prefill 耗时相当。在 **4 × 910B4、上下文配置 65,537 token、关闭图编译与投机** 的部署条件下，两档模型分别约可提供 **64 并发**和 **10 并发**服务。详细规格及指标口径见 2.5 节。

## 1. 背景：两种概率读出路线

Jev-like 服务将证据、问题和候选项转成一个候选分布，再返回判断、选择或评分。它适用于规则判断、任务路由和等级评估等场景。本文中的 Decision 指完成输入计算后直接读出候选概率的请求；Generation 指继续自回归生成文本的请求。

Kev 与 Plumb 都经过“候选 logits → 归一化 → 概率”这一步。关键差异在于候选 logits 如何产生。

![Kev 与 Plumb 的概率读出路线](C:/Users/F-O-O/Projects/vllm-ascend/docs/report/1009/ref/kev_plumb_routes.png)

*图 1：Kev 与 Plumb / JevK5 双流程对比，来自已下载的网页版讨论示意图。图中 Pointer Head 点积为简写，实际实现还包含投影偏置和缩放。*

### 1.1 Kev：用专用 Pointer Head 给候选项打分

Kev 将每个候选项编码到输入中，提取各个选项结束位置的 hidden state，以及决策位置 `<decide>` 的 hidden state。新增的读出头分别得到决策向量和候选向量，计算两者的匹配分数与未校准分布：

```math
q=W_qh_{\mathrm{decide}}+b_q,\qquad
k_i=W_kh_{\mathrm{option},i}+b_k
```

```math
z_i=\frac{q^\top k_i}{\sqrt{d_p}},\qquad
r_i=\operatorname{softmax}(z)_i
```

最终概率还要应用匹配 checkpoint 的校准温度，训练阶段使用 `T=1`。原始 Kev 在 eval 时缩放 logits；本项目服务层沿用适配时的后处理逻辑，对初始分布施加 `1e-9` 下限后再按温度归一化。[本项目温度后处理](https://github.com/singzhou/vllm-ascend/blob/80d40e7030edf948eddc41db0e98f0ab1f5fb77a/vllm_ascend/entrypoints/systemone/serving.py#L28-L43)。

有多少个候选表示，就计算多少个候选分数。当前 Kev 协议及本项目读出规划将选项数限制为 **1–255**；这是当前实现的约束，与 hidden state 维度、Pointer Head 投影维度分别是不同的量。[读出规划校验](https://github.com/singzhou/vllm-ascend/blob/80d40e7030edf948eddc41db0e98f0ab1f5fb77a/vllm_ascend/decision/planner.py#L14-L42)。

Pointer Head 是与主干表示配套训练的参数。已有匹配 checkpoint 时可以直接加载；换用一个未经适配的新主干时，不能默认复用旧头。Kev 的常规训练路线包含主干 LoRA 与新增 Pointer Head 的联合训练。[Kev 原始读出实现](https://github.com/jaredpalmer/kev/blob/fdfdfd2c7a98225715a9cc8f4d9e1ef29ebbe79c/kev/model.py#L182-L200)，[本项目 Pointer Pooler](https://github.com/singzhou/vllm-ascend/blob/80d40e7030edf948eddc41db0e98f0ab1f5fb77a/vllm_ascend/ops/kev_pointer.py#L24-L89)。

### 1.2 Plumb：复用原生 LM head 的答案 token 分数

Plumb / JevK5 将业务选项映射到 A、B、C 等答案字母。输入经过主干后，利用答案起始位置的表示和原生 LM head，取得这些字母的 logits，再在当前候选集合内归一化：

```math
z_i=W_{\mathrm{LM}}[t_i]h_{\mathrm{last}},\qquad
p_i=\frac{\exp(z_i/T)}{\sum_{j=1}^{K}\exp(z_j/T)}
```

其中，`t_i` 是当前 tokenizer 对应的答案 token，`T` 是读出温度。Plumb 固定发布 checkpoint 的温度为 **2.07**。该发布版单轮读出准备 A–P 共 16 个槽位，`choice` 支持 **2–16** 个选项；当前原生适配保留这一范围。16 是该版本的表示与实现边界，不能解释成模型训练得到的绝对容量上限。[Plumb 读出源码](https://github.com/crh225/plumb/blob/1dff3abf0c0300f7a622e9fea88427e53dec97bc/jevk5/runtime.py#L92-L104)，[发布系统卡](https://github.com/crh225/plumb/blob/1dff3abf0c0300f7a622e9fea88427e53dec97bc/bench/out/v5/SUBMISSION.md)。

| 比较维度 | Kev | Plumb / 原生答案 token 路线 |
|---|---|---|
| 分数来源 | 决策表示与候选表示的指针匹配 | 原生 LM head 的候选答案 token logits |
| 所需输出参数 | 额外训练好的 Pointer Head | 当前模型自带的 LM head |
| 读出位置 | 决策位置及各选项结束位置 | 答案起始位置对应的最后输入表示 |
| 候选扩展方式 | 增加输入中的候选及其表示 | 增加答案槽位，或另行设计多轮读出 |
| 切换主干 | 需要匹配的读出头与 checkpoint | 可复用方法，重新使用新模型的 tokenizer、模板和 LM head |
| 本项目原生入口 | `encode()` / pooling | `generate(max_tokens=1)` / 候选原始分数 |

**读出方法可复用，与决策能力可迁移是两件事。** Plumb 也经过针对决策任务的后训练；把相同读出方法接到普通 27B 模型上，并不会自动继承 Plumb-4B 的后训练收益或温度校准结果。不同读出路线的质量，需要在相同任务和评测口径下比较。[训练与读出讨论](https://chatgpt.com/c/6abb2a21-1954-83eb-b8eb-b440aa120a7b)。

## 2. 工程进展：从模型适配到可复用服务能力

工程实现统一了外部协议，并为不同读出方式保留相应的原生引擎路径。Kev 部署使用 pooling；Plumb 与通用 Qwen 部署使用 generation runner。模型执行、调度和缓存由 vLLM-Ascend 承担，SystemOne 层负责输入规划、候选映射和答案转换。

![SystemOne 原生服务实现](C:/Users/F-O-O/Projects/vllm-ascend/docs/report/1009/assets/native_serving.png)

*图 2：依据最终提交绘制的原生服务实现图。两条分支表示按部署模型选择的适配路线；每个实例加载自己的 checkpoint。*

### 2.1 Kev：完成 Pointer Head 与原生 pooling 生命周期接入

Kev 适配的核心工作是将专用读出头放到原生 Qwen 主干执行链路中，并让每道问题成为 vLLM 的独立 pooling 请求。最终方案通过 `engine.encode()` 提交，通过 `merge_async_iterators()` 聚合，复用原生排队、取消、路由、追踪和共享执行器。

| 解决的问题 | 最终实现及价值 |
|---|---|
| 专用读出头如何接入 NPU 主干 | 组合原生 Qwen backbone，在 worker 侧使用 FP32 Pointer Pooler；主干权重交由原生 loader 加载 |
| 长输入分块后如何保留选项表示 | 每个 chunk 收集其覆盖的 option-end 表示，保存低维候选投影；最后一个 chunk 计算决策向量并完成读出 |
| 前缀缓存可能跳过所需选项表示 | Kev 缓存查询范围限制在公共 `state`，保证问题及选项后缀仍经过必要计算；缓存 block 与 hybrid state 生命周期继续由原生引擎管理 |
| Qwen3.5 文本位置接口不匹配 | wrapper 补齐文本 M-RoPE 接口，处理纯文本三轴同位置的情况 |
| HTTP 请求数与模型执行容量混淆 | 删除自定义父请求并发计数与 429，超出执行容量的问题进入原生等待队列；`max_num_seqs` 限制实际执行的序列数 |
| 服务层重复维护请求生命周期 | 删除自定义 deadline、fan-out 限制和 DP affinity，回归原生 encode、聚合与取消机制 |

这些改动使 Kev 能以原生服务结构运行，专用部分集中在输入规划、读出头及必要的模型/缓存适配。公共 `state` 有机会复用前缀缓存，但命中仍取决于真实 token 前缀、缓存是否已建立及副本分配。[Kev 最终服务链路](https://github.com/singzhou/vllm-ascend/blob/80d40e7030edf948eddc41db0e98f0ab1f5fb77a/vllm_ascend/entrypoints/systemone/serving.py#L90-L150)，[缓存适配](https://github.com/singzhou/vllm-ascend/blob/80d40e7030edf948eddc41db0e98f0ab1f5fb77a/vllm_ascend/patch/platform/patch_kev.py#L19-L78)。

### 2.2 Plumb：将答案 token 读出接入原生生成引擎

Plumb 适配从 vLLM-Ascend `v0.27.1rc1`、vLLM `v0.27.1` 基线建立。服务直接加载原始模型权重、tokenizer、chat template 与校准配置。每道 question 独立渲染输入、关闭 thinking，并提交一个原生请求：

```python
SamplingParams(
    temperature=0,
    max_tokens=1,
    detokenize=False,
    logprobs=len(candidate_token_ids),
    logprob_token_ids=candidate_token_ids,
)
```

服务通过 `logprob_token_ids` **指定要读取的候选**，避免全词表 top-k 遗漏低分候选。引擎内部完成首步采样，服务丢弃该 token，只使用候选的原始 logprob；模型前向在完成输入计算后即可得到首步分数，无需将输出 token 再喂回主干进行后续 decode。

引擎返回的原始 logprob，是从每个 logit 减去相同的全词表归一化常数。因此，在候选集合 `C` 内有：

```math
\operatorname{softmax}_{i\in C}(\ell_i/T)
=\operatorname{softmax}_{i\in C}(z_i/T)
```

公共归一化项在候选内再次归一化时抵消，因此这一实现保留了 Plumb 的候选概率读出语义。内部贪心采样的 `temperature=0` 与候选读出的校准温度是两个不同参数。

适配同时补齐 Qwen3.5 文本 M-RoPE 与 checkpoint 权重命名兼容，复用原生 chunked prefill、prefix caching 和 GDN 状态管理。缺少候选分数、子请求失败或结果不完整时明确返回错误。多道问题可以并发执行，结果按输入顺序聚合。[Plumb 原生适配提交](https://github.com/F00L42/vllm-ascend/commit/9f28e2c1f)，[当前候选提取与原生请求实现](https://github.com/F00L42/vllm-ascend/blob/ecd7d1263f6f2ca33d8a92abdecf2c76e9b12006/vllm_ascend/entrypoints/systemone/serving.py#L64-L165)。

### 2.3 TypeSafe SDK：补齐可解析的请求与响应契约

两条路线对外采用 SystemOne 结构：输入为 `state` 和 `questions`，问题支持 `noul`、`choice`、`score`，响应包含 `model`、`answers`、`usage` 和 `latency_ms`。

| 类型 | 对外结果 |
|---|---|
| `noul` | true 的概率 |
| `choice` | 候选分布及最大概率对应的业务选项 |
| `score` | 有序等级分布、从 0 开始的期望分数，以及等级描述 `legend` |

兼容过程中发现过一个实际响应契约问题：接口返回 HTTP 200，但 `score` 缺少 SDK 必填的 `legend`，客户端仍然解析失败。修复后，`legend` 与 `probabilities` 使用一致的等级键，并保留等级描述。Plumb 路线增加了 **`typesafe-sdk==0.7.2` 的真实客户端解码回归**，通过 MockTransport 验证响应可解析，覆盖三种答案类型及评分等级映射。[SDK 契约修复提交](https://github.com/F00L42/vllm-ascend/commit/12d3cc4f1)，[Plumb 接口说明](https://github.com/F00L42/vllm-ascend/blob/ecd7d1263f6f2ca33d8a92abdecf2c76e9b12006/docs/usage/plumb_systemone.md)。

这里的兼容结论按 SDK 请求/响应契约理解。Kev 已实现对应 schema，但提交证据中未发现同范围的真实 SDK 解码回归；两条路线的 `confidence` 计算定义也有差异，因此不能把协议兼容扩展成全部统计语义完全相同。

### 2.4 通用化：把 Plumb 读出方式复用于兼容 Qwen 模型

后续提交将固定 Plumb 的读出逻辑拆成模型相关配置和共用服务流程。`ascend_systemone` entrypoint、API、输入规划、候选提取、并发聚合和 SDK 答案转换都沿用原实现，新增两种启动模式：

| 模式 | 模型与权重 | 读出温度 |
|---|---|---|
| `plumb` | 原始 Plumb checkpoint，读取其 tokenizer、模板与 LM head | 从同一 checkpoint 的 `jevk5_config.json` 读取，固定发布版为 2.07 |
| `answer_tokens` | 兼容结构的 Qwen checkpoint，读取该模型自己的 tokenizer、模板与 LM head | 默认 1.0，可在启动时配置有限正数 |

例如，Qwen3.6-27B 通过下列配置使用通用读出：

```json
{"systemone_config": {"mode": "answer_tokens", "temperature": 1.0}}
```

换模型时，使用新模型自己的输出头，无需新增一个专用分类头或训练 Pointer Head。服务启动时检查 A–P 是否为互不相同的单 token，输入通过模型自身模板渲染；超长输入明确报错。

**当前已实现的范围是结构兼容的 Qwen3.5 / Qwen3.6 文本模型，并非所有 Qwen 架构都已适配。** 通用模式不依赖 Plumb 的 tied embeddings 或校准文件；默认温度 1.0 表示未额外缩放，不能据此称为已校准概率。[通用读出提交](https://github.com/F00L42/vllm-ascend/commit/ecd7d1263f6f2ca33d8a92abdecf2c76e9b12006)，[通用模式使用说明](https://github.com/F00L42/vllm-ascend/blob/ecd7d1263f6f2ca33d8a92abdecf2c76e9b12006/docs/usage/systemone_token_readout.md)。

### 2.5 性能进展：单次时延与部署并发规格

以下数据来自 **2026 年 10 月 10 日补充的用户服务器运行反馈**，反映当前 Jev-like Decision 服务的用时与并发能力。

| 模型 / 服务路线 | 主干模型 | 单次 Decision 推理用时 | 当前部署可提供的并发服务量 |
|---|---|---|---|
| Plumb-4B、Kev-4B 等 4B 模型 | Qwen3.5-4B | 约 300 ms | 约 64 并发 |
| Qwen3.6-27B 通用答案 token 读出 | Qwen3.6-27B | 约 1,000 ms（1 秒） | 约 10 并发 |

两档模型的单次 Decision 用时，与 vLLM 提供相应主干模型的 **Prefill 用时相当**。这是处理输入后读出候选分数的推理用时量级；Decision 无需继续生成长文本。该观察说明当前决策服务的耗时与主干输入计算相对应，读出额外开销的大小仍需分项 profiling。

上述并发服务量对应以下共同部署规格：

| 规格项 | 当前条件 |
|---|---|
| 推理框架 | 本项目原生 vLLM-Ascend 服务 |
| 设备 | 4 张 Ascend 910B4 |
| 上下文配置 | 65,537 token |
| 图编译 / 执行图优化 | 未开启 |
| 投机解码 | 未开启 |
| 并发服务量 | 4B 约 64；27B 约 10，按所部署模型列示 |

**指标口径：**65,537 是上下文配置规格，反馈未给出时延测量时各请求的实际输入长度；并发数是当前部署可提供的服务量，不是请求/秒，也不表示全部并发请求同时维持上述单次时延。单次推理用时尚未附带排队、缓存冷热、question 数、重复次数及统计分位数，不能直接当作完整 HTTP 端到端 P95/P99。这组规格用于说明当前 Decision 能力，D/G 持续混合负载的延迟与图优化收益另行验收。

### 2.6 当前证据与验证状态

| 项目 | 已有证据 | 结论范围 |
|---|---|---|
| Kev 原生接入 | 已提交的模型 wrapper、Pointer Pooler、缓存适配与原生 encode 链路；本轮补充 Kev-4B 服务器运行反馈 | 原生工程适配完成；提交历史未包含 NPU 验收，后续运行用时与并发规格见 2.5 节 |
| Plumb 与通用 Qwen 读出 | 原生 generate 链路、协议/配置测试与 SDK 0.7.2 解码回归；本轮补充设备运行反馈 | 读出及接口契约已实现；已有两档模型的时延与并发观察，离线契约测试和服务器反馈分列 |
| Decision 性能与部署容量 | 本轮用户服务器反馈：4B 约 300 ms / 64 并发，27B 约 1,000 ms / 10 并发 | 4 × 910B4、上下文配置 65,537 token、关闭图编译与投机；指标口径见 2.5 节 |
| Qwen-27B 同实例 D/G | 网页讨论中记录的服务器验证反馈 | 工程与功能共存已验证；Decision 单次约 1 秒，当前 eager 配置下 Generation 速度低于可用水平 |
| 混合服务性能 | 已识别配置、图分派、采样和资源竞争问题 | 图、异步、投机及持续混合负载的性能验收仍需推进 |

服务器反馈包含此前同实例 D/G 功能验证及本轮补充的 Decision 性能规格，尚未附带原始压测日志和完整负载分布。本文据此更新功能与性能进展；固定负载下的延迟分位数和持续混合服务性能按 3.4 节继续验收。[此前服务器验证与后续分析讨论](https://chatgpt.com/c/6ac854e4-b440-83eb-bf0c-fb7bb33c741c)。

## 3. 同一模型提供 Decision 与 Generation：可行性与性能路径

本节以当前 Qwen-27B 的原生答案 token 路线为对象。两类请求都进入同一个 generation engine，共享模型权重、LM head、设备与调度器：

- **Decision**：处理证据与候选输入，在 prefill 末位置取得候选分数，返回结构化结果。长冷输入可能经历多轮 chunked prefill。
- **Generation**：处理对话输入后继续 decode，逐步输出文本；后续可使用适配完成且实测有收益的投机路径。

Kev 当前部署使用专用 pooling wrapper；本节的同模型 D/G 共存结论对应 Plumb / 通用 Qwen 的 generation runner。

### 3.1 当前瓶颈：保守配置影响整个引擎

当前使用说明推荐 `--enforce-eager --no-async-scheduling`，用于降低首版联调复杂度。它们是**整个实例的配置**，所以 Generation 同样失去相应的图执行和调度重叠收益。

Decision 只读取首步分数，并不意味着计算原理要求 eager。通用读出的 `validate_runtime()` 未强制 eager 或同步调度，但明确拒绝投机、量化和 `enable_reduce_sample`，并要求 V1 runner 与 `raw_logprobs`。因此，需要分别解决文档保守配置和插件显式兼容限制。[当前运行时校验](https://github.com/F00L42/vllm-ascend/blob/ecd7d1263f6f2ca33d8a92abdecf2c76e9b12006/vllm_ascend/entrypoints/systemone/serving.py#L40-L61)。

首项性能验收应当是：**启用 SystemOne 插件、没有 Decision 流量时，Generation 能否保持独立优化部署的执行路径和性能。** 这项对照先隔离全局配置造成的损失，再评估实际 Decision 流量的代价。

### 3.2 DG 同 batch：prefill 改变图执行条件

恢复 `FULL_DECODE_ONLY` 后，适合的 uniform decode 批次可以回放完整执行图。Decision 的多 token prefill 与 Generation decode 进入同一次 forward 时，批次形态会变化，该步可能走非图路径，影响同批 Generation。

![DG 同 batch 对 FULL_DECODE_ONLY 的影响](C:/Users/F-O-O/Projects/vllm-ascend/docs/report/1009/ref/GD共存分析.png)

*图 3：已下载的 DG 共存分析示意图。图中 CUDA Graph 表示执行图概念，在本项目 Ascend 后端对应 ACL Graph。图示描述启用图模式后的分派机制；当前全局 eager 启动配置尚未获得其中的图收益。*

图示可以按三个时段理解：

1. **纯 Generation decode**：满足图分派与已捕获形状等条件时，使用 decode 图。
2. **Generation decode + Decision prefill**：新增 prefill 改变批次形态，本轮可能回退，影响该次 forward 中的生成请求。
3. **Decision 完成、再次形成适合的 decode 批次**：后续可以恢复已有图的回放。

这里影响的是**执行时的图命中与回放**。若 Decision 持续到达，或长输入连续占据多个 prefill chunk，Generation 可能频繁遭遇非图步骤；具体影响持续多久，需要记录实际图分派及输出间隔。图中的纯 decode 场景是简化表达，真实分派还受 query length、batch size 和已捕获形状约束。[本分支 runner 图分派](https://github.com/F00L42/vllm-ascend/blob/ecd7d1263f6f2ca33d8a92abdecf2c76e9b12006/vllm_ascend/worker/model_runner_v1.py#L3138-L3178)，[vLLM 0.27.1 图模式定义](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/config/compilation.py#L55-L58)。

**同实例服务两类请求，可以根据执行形态选择混批或交替执行批次。** 如果同批会持续破坏 Generation 的关键快路径，可让 Decision prefill 按受控预算插入，尽量保留纯 decode / verify 批次。拆批会增加调度和权重访问开销，收益也要通过对照实测。

### 3.3 还需要处理的三类开销

| 层次 | 具体问题 | 后续处理方向 |
|---|---|---|
| 候选读出与采样 | 当前仍通过完整 LM head 和原生 sampler 取得候选；有人要求 raw logprobs 时，相关归一化可作用于整批 logits。当前同时设置 `logprobs=K` 与指定 token，还可能产生冗余 top-k 工作 | 先验证只请求指定候选的参数组合，再考虑按请求行分流读出；更进一步研究仅计算候选 LM head 行 |
| 投机兼容 | 当前插件拒绝投机。混合投机批次涉及目标模型分数、bonus token、候选输出和请求行的映射，Decision 还应及时完成并退出后续草稿组织 | 明确普通候选读出与拒绝采样的张量/结果契约，保留 Generation 的投机路径，并排除 Decision 的冗余草稿计算 |
| 输入与资源竞争 | 长冷 Decision 仍需大量 prefill；多 question 放大序列数量。两类请求共享计算、缓存和 CPU 请求处理能力 | 按未缓存输入量、question 数和批次执行时间管理预算，控制突发、排队与缓存驻留 |

指定返回 16 个候选，只约束读取结果的范围；当前实现尚未做到仅计算 16 行 LM head，也没有完成读出成本的逐请求隔离。[vLLM 0.27.1 sampler](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L79-L126)。

缓存可以减少相同 token 前缀的重复计算，但聊天与 Decision 的模板可能不同。同一份文档内容，并不自动形成可共享的 token 前缀。图缓冲区、KV/GDN state 和投机草稿状态也需要独立预算，共享权重不会消除这些开销。

### 3.4 目标方案与推进顺序

目标是在同一份模型权重上提供两种请求策略：Generation 保留优化后的 decode / verify 路径，Decision 在正确位置读出必要候选；调度器根据批次形态与资源预算组织执行。在约定输入长度、并发和到达率下，双方都满足业务延迟目标。

| 阶段 | 工作 | 验收重点 |
|---|---|---|
| 1. 恢复图执行 | 去掉全局 eager，先验证 `FULL_DECODE_ONLY`；第一轮保持无投机，隔离变量 | Generation-only 的实际图回放与输出速度；插入 Decision 后的回退次数和持续时间 |
| 2. 降低读出额外工作 | 验证去掉冗余 top-k；按 profiling 决定是否按行分流、优化候选投影 | 指定候选完整返回；Generation 普通输出及 logprobs 契约保持正确；sampler 耗时下降 |
| 3. 恢复异步与适配投机 | 分别验证结果复制、结束/slot 回收，以及拒绝采样和草稿输入映射 | 混合请求结果无串扰；Generation 有实际收益；Decision 不承担无效草稿计算 |
| 4. 管理持续混合负载 | 受控 prefill 预算、question 扇出与准入，比较混批和交替批次 | Decision 完整响应 P95/P99、Generation 首字延迟与持续输出间隔同时满足目标 |

压测应固定 checkpoint、TP、输入输出长度、并发和热身条件，分别覆盖短 Decision、长冷 Decision、暖缓存、多 question，以及持续 Generation 中的单次和连续插入。除平均 token/s 外，记录排队时间、图命中/回退、prefill 与 sampler 耗时、缓存命中和抢占重算。服务器已有的功能共存结果可作为起点，性能阶段用这些对照定位损失来源。

## 4. 提交与资料索引

### 4.1 关键提交

Kev 路线以 `codex/kev-native-engine` 的最终提交 `80d40e703` 为准；Plumb / 通用 Qwen 路线以 `codex/systemone-token-readout` 的 `ecd7d1263` 为准。它们是两条不同的提交链。

| 提交 | 工程进展 |
|---|---|
| [8a9a1caf6](https://github.com/singzhou/vllm-ascend/commit/8a9a1caf6a4e4ff6bf72f66f1f550053b40265ed) | Kev 原生 SystemOne、Qwen wrapper、Pointer Pooler、缓存适配与导出器 |
| [a08dbd921](https://github.com/singzhou/vllm-ascend/commit/a08dbd9211395d67fe450c98bb70a943c09ba7e4) | 明确上下文配置来源与原生引擎审计边界 |
| [aa739d878](https://github.com/singzhou/vllm-ascend/commit/aa739d8789312ac39fb21a67102ee50b7277011b) | 补齐 Qwen3.5 wrapper 文本 M-RoPE |
| [28a933f5c](https://github.com/singzhou/vllm-ascend/commit/28a933f5cd1bdd8c6562710ff7a2415ac1d3f1e1) → [c746535b2](https://github.com/singzhou/vllm-ascend/commit/c746535b2106b1dcca90284e03e759d16ee3c25c) | 从部署容量修复演进为删除自定义 HTTP 并发限制、由原生调度器排队 |
| [80d40e703](https://github.com/singzhou/vllm-ascend/commit/80d40e7030edf948eddc41db0e98f0ab1f5fb77a) | Kev 最终原生 pooling 生命周期与路由 |
| [9f28e2c1f](https://github.com/F00L42/vllm-ascend/commit/9f28e2c1f) | 在 release 基线上新增 Plumb 原生候选概率服务 |
| [12d3cc4f1](https://github.com/F00L42/vllm-ascend/commit/12d3cc4f1) | 修复 TypeSafe SDK `score.legend` 契约并加入解码回归 |
| [ecd7d1263](https://github.com/F00L42/vllm-ascend/commit/ecd7d1263f6f2ca33d8a92abdecf2c76e9b12006) | 将 Plumb 专用逻辑扩展为兼容 Qwen checkpoint 的 `answer_tokens` 模式 |

### 4.2 讨论与图源

- [解读适配开发提交](https://chatgpt.com/c/6ac854e4-b440-83eb-bf0c-fb7bb33c741c)：同实例 D/G 功能验证反馈、性能分析与图 3 来源。
- [Kev 实现路线对比 Plumb-4B](https://chatgpt.com/c/6abb2a21-1954-83eb-b8eb-b440aa120a7b)：读出路线、训练与跨模型复用边界，以及图 1 来源。
- 图 1、图 3 使用 `ref` 中的原始 PNG；图 2 的可编辑源为 [native_serving.svg](C:/Users/F-O-O/Projects/vllm-ascend/docs/report/1009/assets/native_serving.svg)。
- 2026 年 10 月 10 日本轮用户服务器反馈：补充 4B / 27B 的单次推理用时，以及四卡 910B4、65,537 token 上下文配置下关闭图编译和投机时的并发服务量；详见 2.5 节。

本稿作为 HTML 演示的事实源和资料索引。代码中既有的 `pending_remote` 记录，与后来提供的服务器功能及性能反馈分别列示；恢复图、异步与投机的方案均属于下一阶段工作。
