# 参考：服务器验证、Telemetry 与本地诊断闭环

日期：2026-09-18。适用条件：本地 Codex 不访问服务器，服务器执行者拉取个人分支、编译安装后运行测试并回传资料。

## 1. 一轮实验只解决一个明确的不确定性

标准流程：

1. 本地冻结当轮源码组合，写明假设、代码变更、预期输出及失败时要捕获的证据。
2. 本地执行实际能运行的检查；缺 torch/NPU 时明确 skipped，不能 mock 一个设备测试后报告 NPU PASS。
3. 随分支提供服务器命令/测试程序及回传目录规范，不能只说“服务器跑一下”。
4. 服务器先核对 SHA/依赖/加载产物，再构建、smoke、目标实验。
5. 本地收到结果，先核对 provenance 和命中情况，再定位最早失败层。
6. 用一次受控修改回答失败假设；更新阶段状态及下一轮最小实验。

每轮目录建议为 `runs/<round_id>/`。下面文件名为拟议交付规范，需 Codex 在实现中生成，不表示本包已提供脚本。

| 文件 | 内容 | 何时必需 |
| --- | --- | --- |
| `manifest.json` | 四仓/嵌套依赖 SHA、dirty 状态、环境、模型、运行参数 | 每轮 |
| `build.log` | 完整编译/安装日志、命令退出码 | 改原生代码或依赖 |
| `load.json` | Python 包路径、扩展/OPP 实际位置、产物 hash、设备信息 | 每轮 |
| `results.json` + `junit.xml` | 用例总数、pass/fail/skip、失败参数、退出码、误差 | 测试轮 |
| `route.jsonl` | 资格/路由、fallback 原因、实际命中证据 | 模型接入轮 |
| `events.jsonl` | 选定请求/layer/rank 的状态阶段事件 | 状态诊断轮 |
| `case_<id>/` | 可回放的输入、pre/post state、metadata、reference | 精度/状态失败 |
| `perf_samples.csv` + trace | 原始时延、工作负载、稳定期 trace | 性能轮 |
| `summary.md` | 本轮结论、异常/退出阶段、限制、下一步 | 每轮 |

服务器即使失败也应尽量保留 manifest、日志和已完成结果。NPU 异步错误后未必能安全 dump，所以关键输入和初始状态应在受控复现的 launch 前保存；不能承诺 crash 后再补抓所有数据。

## 2. 最小 manifest

至少包含以下字段；值由实际运行采集，不能填示例当事实。

```text
{
  "schema_version": 1,
  "round_id": "P2-r01",
  "hypothesis": "noncontiguous checkpoint IDs preserve vLLM state semantics",
  "source": {
    "vllm_ascend_sha": "FULL_SHA",
    "vllm_sha": "FULL_SHA",
    "xllm_sha": "FULL_SHA",
    "xllm_ops_sha": "FULL_SHA",
    "pto_sha": "FULL_SHA_IF_USED",
    "dirty": false
  },
  "environment": {
    "soc": "ACTUAL_SOC",
    "cann": "ACTUAL_VERSION",
    "driver": "ACTUAL_VERSION",
    "torch": "ACTUAL_VERSION",
    "torch_npu": "ACTUAL_VERSION",
    "python": "ACTUAL_VERSION"
  },
  "workload": {
    "model_id_and_revision": "ACTUAL_MODEL",
    "dtype": "ACTUAL_DTYPE",
    "tp": 1,
    "speculative_tokens": 3,
    "batch_mode": "pure_spec_uniform",
    "execution_mode": "eager",
    "async_scheduling": false,
    "prefix_cache_mode": "ACTUAL_SETTING"
  },
  "telemetry": {"mode": "metadata", "performance_result_valid": false}
}
```

另外记录：安装命令、编译选项、实际 model config 的 heads/D/Conv/eps、runner 版本、seed、请求集 hash、输入/输出长度、硬件卡数、并行配置及相关环境变量白名单。不要 dump 整个进程环境。

上述 tp/K 是格式示例，不是用户最终配置。trace/dump 与 manifest 必须具有同一 round/run 标识。

## 3. 本地与服务器测试职责

| 检查 | 本地无 NPU | 服务器 |
| --- | --- | --- |
| 源码语法、lint、diff、注册/构建入口审查 | 可按现有依赖执行 | 构建时再验证实际产物 |
| 索引、游标、生命周期 CPU reference | 可运行，推荐纯 Python/可用 CPU 框架 | 可复用 |
| dtype/shape 资格、路由逻辑 | 可测试；mock 仅证明控制逻辑 | 用真实 metadata 补齐 |
| PTO/CANN 编译、OPP 符号、真实 stream | 无环境即不能证明 | 必须 |
| BF16/FP32 精度、state alias、910B 并发 | CPU 仅参考 | 必须 |
| 图捕获/replay、异步请求、E2E/性能 | 不能用静态检查替代 | 必须 |

本地测试报告区分 executed/pass、failed、skipped\_env、pending\_remote。不得把“服务器尚未运行”归为 PASS。

## 4. 测试矩阵：按阶段增加，失败即收敛

### A. 构建和算子加载

- 精确 910B SoC 编译；父仓 editable 安装；新进程真实调用算子。
- 记录实际 `_C_ascend` 与自定义 OPP 来源；符号存在但 kernel 缺失的情况必须能识别。
- 对比开关关闭的原生最小运行；缺少候选包时的资格/回退行为。
- P1 使用原版合法 contiguous 状态池，不接生产模型状态。

### B. 数学和形状

首轮小集：B=1/2，K=1/3，实际模型 TP 后 NK/NV，D=128、Conv4。然后覆盖 B=32、K=16 及至少一个动态模板 K（如 K=6），并验证原版 non-FLA 所走 tiling key。无需一开始做全部笛卡尔积。

- 正常随机输入、全零、非零初始状态、小值/大 gate 值与 softplus 阈值附近。
- 对比 convOut、g/beta、每 token H、pre-norm/readout、最终输出。中间值必要时由诊断 kernel 路径提供，不改生产默认输出。
- 非法 B/S/head ratio、dtype/stride/bias/eps 等必须在写状态前拒绝或回退；不要故意把越界索引下发给原版不校验值域的 kernel。
- Dk=Dv 相等会掩盖转置 shape 问题；使用非对称初始矩阵和结构化向量验证 `[V,K]`，不能只用零矩阵。

### C. 物理状态与跨轮事务

| 用例 | 故障目标 | 必须比较 |
| --- | --- | --- |
| checkpoint ID 随机排列、不连续 | 错用 slot×S | 每 token H 所在物理块 |
| block\_stride 大于逻辑大小、非零 storage\_offset | 漏掉页 padding/view 偏移 | 有效数据及页 padding sentinel |
| Conv 与 SSM 物理布局不同 | 共用错误 state\_id | Conv 窗口与 SSM 对应前缀 |
| m=1、中间值、m=S | accepted 偏一 | 正确初始 H/Conv 与输出 |
| 两轮以上：拒绝首候选/部分接纳/全接纳 | 只验证单轮最终状态 | 下一轮输入、选中状态与后续输出 |
| postprocess 复制并重置游标 | 重复应用旧 accepted | 复制后物理状态与 m=1 语义 |
| batch 重排、收缩/扩张、请求完成后 slot 重用 | 用 batch 下标当请求身份 | 请求 ID/epoch 与物理映射 |
| 两请求共享读、独立写 | COW/共享快照污染 | 共享源不变、两个结果独立 |
| write 唯一但覆盖另一请求 read | 忽略读写交叉冲突 | 调用前拦截或明确隔离 |
| 同请求原位与独立输出池 | 覆盖未读历史/丢未写区 | pre/post 全写集合与 sentinel |
| abort/抢占/恢复/缓存复用 | 过早释放或状态失配 | 状态生存期与请求下一次执行 |

CPU reference 只需建少量标记块，就能证明地址/游标规则；NPU 测试使用独立 clone 的状态池验证真实写入。验证 offset >2³¹ 时，CPU 地址算术测试和硬件条件允许的实测分开，避免为一个算术边界强制申请巨大显存。

### D. 模型集成

- 首轮 eager、纯 spec、固定 S；TP=1 能装下的代表模型先测，实际目标模型/TP 随后验证。
- 加 trace/route 证明 Mega 被调用，不能接受“模型能生成但一直 fallback”的假通过。
- 固定 token 轨迹/强制已知接纳决策验证状态机；另做真实 sampler 的 E2E，避免随机采样掩盖/放大第一处分歧。
- 对同一 pre-state 和输入运行 baseline/candidate：**分别 clone，两者不能串行修改同一个 cache**。
- 比较 token/logits、state、接受长度分布和错误率；有差异时先回到单层重放。
- 未启用 Mega、资格不满足、混合 prefill/decode、无投机、无实际 draft 的批次都验证原生分支。

### E. Graph / async / prefix

- 图捕获后连续 replay，改变 indices/accepted/活跃请求但保持合法 shape；检查读取的 metadata 是新值。
- padding 行、零活跃行，以及 `spec → no-spec → spec`，确认旧请求状态不被继续更新。
- sink 独占与容量；或 device mask 不解引用哨兵且不破坏 barrier。
- prefix miss/hit、多个请求共享同一 prefix、跨块对齐、只发布已确认状态。
- async 开/关分别验证，并检查 metadata 更新、复制、compute、postprocess 的 stream/event 顺序。
- 仅对项目准备支持的模式跑组合测试；尚未支持的配置应明确回退，不为了扩大表格提前引入实现复杂度。

## 5. Telemetry 设计

以下开关为 **拟新增接口**，本地 Codex 应按本仓配置规范实现并登记，不可直接当现有环境变量使用：

| 逻辑模式 | 输出 | 约束 |
| --- | --- | --- |
| off（默认） | 无逐层事件/张量 dump | 快路径只保留低成本模式判定 |
| metadata | shape/dtype/stride、路由与静态配置 | 不通过读设备张量产生隐式同步 |
| state\_summary | 选定状态的有限统计量/检查结果 | 允许额外诊断成本，结果不用于正式性能 |
| replay\_dump | 指定 step/layer/rank 的最小重放包 | 明确上限、采样条件与保存阶段 |

推荐配置字段：`mode`、`run_id`、`rank_filter`、`layer_filter`、`request_filter`、`step_range`、`max_events`、`max_dump_bytes`、`output_dir`、`dump_on_first_mismatch`。生产路径不默认打印完整 tensor、prompt 或生成文本；诊断尽量使用合成输入/局部激活。

### 5.1 插桩位置

| 位置 | 捕获内容 | 原因 |
| --- | --- | --- |
| metadata 生成后 | 请求到 batch/group 映射、query 长度、cursor 版本、资格判断 | 判断上游是否已经错误 |
| adapter 前、state 尚未修改 | shape/stride、输入和选定 pre-state、读写集合 | 最小重放起点 |
| kernel 完成后 | output、Conv、逐 token H、未触及 sentinel | 判断计算/写入错误 |
| 接纳与 postprocess 前后 | 采样接受数、选中 checkpoint、复制目标、cursor 更新 | 判断跨轮语义错误 |
| 下一轮 forward 前 | 实际读到的 pre-state 与 token 边界 | 证明状态真正被正确消费 |

事件字段建议：`schema_version/run_id/round_id/step/rank/layer/request_key/request_epoch/batch_row/group_id/phase/mode/B/S/T/NK/NV/dtype/shape/stride/storage_offset/route/reason`，并给相关 dump 文件名。

`request_epoch` 或等价 generation 标识用于区分物理 slot 回收后不同请求；仅记 slot ID 无法判断是否串状态。accepted 字段分开命名：`sampled_draft_accept_count`、`effective_state_cursor`、`cursor_stage`；没有采到的数据写 null，不猜测。

### 5.2 同步和 graph 限制

- 正常路径禁止因日志新增 `.item()`、`.cpu()`、全局 synchronize 或张量 repr。
- diagnostic 的 device summary / dump 会改变执行时序，明确标记；不要据此判断 hostbound 性能。
- 确需 dump 时，在 capture 外安排拷贝，用正确 stream/event 顺序保证 pre-state 尚未被覆盖、post-state 已写完。
- 可先保存有限设备快照，再在安全阶段转 CPU；不能仅保留一个指向仍在被修改的 Tensor 引用。
- graph 内的 Python 日志通常只在 capture 时执行，不能拿它统计 replay 命中。用图调度记录、trace kernel 证据或专用诊断计数交叉确认。
- 设备日志/printf 仅限极小重现；默认在框架边界采集，避免把内核时序扰动引入主线。

## 6. 可重放 dump 规格

目录建议：

```text
case_0001/
  meta.json
  inputs.npz
  pre_state.npz
  baseline_post.npz
  candidate_post.npz
  compare.json
  replay_instructions.md
```

上述树是文件列表约定，不是程序输出示例。具体序列化格式可按本仓测试基础确定；保证本地 CPU 可读 metadata/比较结果。

必须包含：

1. 算子输入 qkv/z/a/b、Conv/norm/A\_log/dt\_bias 等权重与配置，或能稳定定位且服务器可访问的权重引用/hash。
2. 所有被读取的 pre-state 和所有可能写入的区域；足够恢复共享读、写 alias、sentinel 的映射。
3. checkpoint 表、Conv ID/offset、accepted/cursor、有效长度、padding mask、tensor shape/dtype/stride/storage\_offset。
4. baseline/candidate 的输出和实际写入状态；如缺任一侧需注明。
5. 数值比较配置、seed、原始坐标、第一处分歧位置、源码与环境关联。

不必 dump 整个模型/状态池。用稀疏块快照及显式 ID 映射重建小池，但要保留共享关系、真实 stride、storage\_offset 和 alias；不能独立保存成 contiguous 张量后丢失原位错误。BF16 若用 NPZ，可按原始 uint16 位模式保存并在 meta 写明 dtype/endianness；不能悄悄转 FP16。格式及恢复程序须互测。

建议复现入口支持两种模式：CPU 检查索引/状态边界；服务器 NPU 重放原生和候选。CPU 数值仅为参考，不能宣称等价于 NPU 指令舍入。

## 7. 结果怎么判定

| 结果 | 判定 | 下一步 |
| --- | --- | --- |
| 编译/加载失败 | P1 未通过 | 依据最早 error 缩到构建层 |
| 生成正常，但 Mega 命中为 0 | 无候选验证证据 | 查资格与包加载，测试强制命中模式应报错 |
| 输出近似但 checkpoint/未触及块错误 | 状态失败 | 禁止进入性能阶段 |
| 单轮正确，第二轮失败 | 游标/重定位/Conv 历史/所有权优先 | 用两轮 dump 比较消费边界 |
| eager 正确，graph 失败 | 图/metadata 更新失败 | 查稳定地址、失活行、mutation 和 replay 依赖 |
| 差异从 g 开始 | 数值合同差异 | 核对 FP32/BF16 舍入与 decay，先修首个差异 |
| kernel 更快但服务不变 | 未证明端到端收益 | 看适配成本、命中比例、关键路径、图模式和其他层 |

测试 summary 同时列出 exact checks 与 float checks。浮点值至少报 max\_abs、RMSE、带下界的 max\_rel、NaN/Inf 数量和失败坐标；未触及区域要求位级不变。不要用较大的全局容差掩盖索引错误。

候选 launch 后抛错可能意味着状态已被部分写入，也可能异步污染后续工作。常规处理是让该轮失败并重建独立状态/重启测试进程；不捕获异常后对同一 cache 再跑原链。具有完整 pre-state 的专用测试可以恢复重放，但不能自动推广到服务现场。

## 8. 性能实验

正确性门槛通过后，独立启动 telemetry off 的测量轮。baseline/candidate 使用同一源码组合、模型、请求集、cache 冷热条件与调度参数；先预热，分别报告冷启动和稳态。

| 层次 | 计时/指标 | 必须包含 |
| --- | --- | --- |
| Device | 设备事件或 profiler 的实际 kernel 完成时间 | tiling key、B/S/heads、非 FLA、warmup、重复样本 |
| 调用/层 | 包含必要 metadata、布局处理、分配、state 整理的完整边界 | 新增 gather/scatter、权重转换是否在热路径、launch 次数 |
| 服务 | TTFT、TPOT/ITL、吞吐、P50/P95、接受长度、错误率 | 并发、输入/输出长度、命中率、fallback 分布、TP、graph/async |

投机一次迭代可能返回多个 token，迭代延迟不等于 TPOT；定义统计分母并固定输出 token 计数口径。内核总时长之和也不等于服务墙钟，多 stream/rank 有重叠。

对 stateful 微基准，baseline/candidate 每次从同一受控状态开始或使用相同合法演化轨迹；不要重复覆盖后随意使用错误 cursor。用于恢复初始状态的额外测试拷贝置于 kernel-only 计时外；业务必需的拷贝仍计入完整调用成本。

推荐交错 A/B 多次重复并保留原始样本，而不是只报最低时延。profiler 采集轮用于解释，独立无 profiler 轮用于性能数值。若采集用当前 vLLM `--profiler-config` 或服务端点，必须按当轮源码生成准确命令，不沿用旧参数。

收益归因回答四问：少了哪些 launch？CPU 下发/设备空隙减少多少？新增适配/状态搬运花多少？服务瓶颈是否真的落在这段？没有 trace 证据时只能写“融合预期减少下发”，不能写“已消除 hostbound”。

## 9. 推荐首轮服务器请求

P0 先请求一次原生路径小样本：环境和加载 manifest；单个 GDN 层、固定 K/B 的 shape/dtype/stride；上一轮 accepted 到下一轮 cursor 的两轮记录；相关物理 checkpoint/Conv offset；原生层的短 trace。P1 再验证最小 Mega 编译/运行。

若已有足够同版本证据可复用，不重复收集。若缺模型或内存不足，先进行合成算子/状态测试并报告限制；不能把合成测试当目标模型 E2E。
