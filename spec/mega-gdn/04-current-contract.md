# 当前源码契约与 P1 接口

核对日期：2026-09-20。精确依赖见 [source-manifest.json](source-manifest.json)。
基线为 upstream `8bd6fe0bc519e2beb9930687ac3e89f2245aa70d`；vLLM 使用本仓 CI 验证提交，参考引擎使用本轮 fetch 的 main。

## 调用与状态链

| 环节 | 当前源码 | 核对结果 |
| --- | --- | --- |
| 投影、输出边界 | `vllm_ascend/ops/gdn.py::forward` | core 写入后调用 `self.norm(core,z)` 一次，再 out_proj；Mega 已含 norm，P3 必须同时替换 core+norm |
| 原生 core | 同文件 `_forward_core` | 等待 connector → Conv → 拆 Q/K/V → gating → Q/K L2 → recurrent → 保存钩子 |
| spec metadata | `vllm_ascend/ops/gdn_attn_builder.py` | 从组对应 block table 取前 S 列，按 spec 请求选择；Conv cache_indices 保留二维表 |
| cache mode | `refs/vllm/vllm/v1/attention/backends/utils.py::mamba_get_block_table_tensor` | none/all 使用传入表；align 按 seq_len/block_size 选窗口，不能绕过此选择 |
| Conv 物理块 | `csrc/moe/causal_conv1d/op_kernel/causal_conv1d.h` | 按 cacheIndicesStride 取每行首列，游标为 accepted−1 |
| SSM 物理块 | `csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h` | 读表中第 accepted−1 列；第 t 步写表中第 t 列，矩阵方向 `[NV,V,K]` |
| 接纳与跨轮 | `vllm_ascend/worker/model_runner_v1.py`、`ops/triton/mamba/postprocess.py` | accepted 来源是采样输出非 −1 数量；按 cache group 搬状态，整理后游标可能重置；不能把旧 accepted 直接沿用 |
| v2/异步 | `worker/v2/model_states/mamba_hybrid.py`、`ops/triton/v2/mamba/precopy.py` | 有 request-state-slot/batch 重排与 precopy；本轮未验证、不放行 |

CPU 地址模型在 [contract.py](../../tools/mega_gdn/contract.py)，仅解释当前固定 S 的访问集合。
它不是 runtime adapter，也不证明 prefix/async 生命周期正确。真实 stride、模型配置和连续两轮状态需要 P0 dump 复核。

例如 S=3、物理表 `[7,2,11]`、accepted=2：Conv 从块 7 的行 1 开始读三行；SSM 读块 2，依次写 7、2、11。
原版 Mega 对 read_slot=7 会计算 `7*3+1=22`，显然不是块 2。non-FLA 只改变矩阵方向，不改变这个地址差异。

## 本轮独立算子

```python
conv_out, norm_out = torch.ops._C_ascend.npu_mega_gdn_mtp_decode(
    qkv, z, b, a, conv_weight, conv_state, a_log, dt_bias,
    ssm_state, read_state_indices, write_state_indices,
    num_accepted_tokens, norm_weight, fla_ssm_state_layout=False,
)
```

`conv_state` 和 `ssm_state` 原位修改；两个返回值独立分配。schema 用不同 alias 集合显式标记 mutation，Meta 返回符号形状。

| 输入 | 形状 | dtype |
| --- | --- | --- |
| qkv | `[B,S,(2*NK+NV)*128]` | BF16 |
| z | `[B,S,NV,128]` | BF16 |
| a、b | `[B,S,NV]` | BF16 |
| conv_weight | `[4,C]` | BF16 |
| conv_state | `[N,S+2,C]` | BF16 |
| a_log、dt_bias | `[NV]` | FP32 |
| ssm_state | `[N*S,NV,128,128]` | FP32 |
| read/write indices、accepted | `[B]` | INT32 |
| norm_weight | `[128]` | BF16 |

所有输入在同一 NPU、连续 ND；B=1..32、S=2..17、N=1..1024，NK 为不超过 16 的 2 次幂，NV/NK 为 1..4 的整数。
仅 910B 构建注册。bias、其他维度、FP16、strided state、变长行、padding、图模式均不属于本轮接口。

每行读取 `read_slot*S+accepted−1`，写 `write_slot*S+t`。Conv 初始窗口为 `[accepted−1:accepted+2]`；
写目标槽前两行来自读槽 `[accepted:accepted+2]`，后 S 行来自 qkv。未写槽逐位保持。

host 只检查张量元数据，不读取设备索引数值。低层调用者必须保证：索引在 `[0,N)`、accepted 在 `[1,S]`、
目标槽唯一，任何请求的写集合不覆盖其他活跃请求的读/写集合。共享只读源和单请求原位更新允许；负 PAD 不可 clamp。
验证脚本在数据仍位于 CPU 时调用契约检查。此接口不能直接接收未经验证的模型 metadata。

## 数值边界

| 阶段 | 原版 Mega | 当前原生链 / 待解决项 |
| --- | --- | --- |
| Conv | 固定宽度 4、SiLU、无 bias、BF16 输出 | 核对实际层 bias、activation、权重布局 |
| Q/K L2 | FP32 内部归一化，eps=1e−6 | 原生 `l2norm_fwd` 会物化激活 dtype，舍入点不同 |
| gate g | 先转 BF16，再转 FP32 求 exp | 原生 gating 输出 FP32；这是明确的数值差异 |
| beta | BF16 sigmoid | 继续以当前 kernel 对齐 |
| SSM | FP32，FLA 或 non-FLA | vLLM `[NV,V,K]`，须核对真实 stride 与块 padding |
| norm/z | core 先 BF16、eps=1e−6、norm 后 SiLU(z) | 核对实际 eps、group_size、norm_before_gate；不得重复 norm |

P1 验证沿用来源测试容差：Conv rtol=8e−3/atol=1e−6，SSM 5e−3/2.5e−5，out 5e−3/2e−2；Conv state 和未触及块严格相等。
与来源测试一致，SSM/output reference 使用独立原生 Conv 的输出，以隔离 Conv 舍入误差。这里的容差不是 vLLM 模型等价标准。
发现首个分歧时保留 dump，禁止通过扩大容差宣称接入成功。

## 阶段状态

- I1–I3：个人分支、源码 pins 和本地 CPU 检查已落地。
- P0：静态调用链、CPU 地址/跨轮测试、显式原生采集脚本已落地；真实模型基线 `pending_remote`。
- P1：Host/Device、PTO、C++/Meta、独立验证入口已落地；910B 编译/加载/数值 `pending_remote`。
- P2–P6：尚未实施。先回传本轮构建和原生状态证据，再改物理索引/stride ABI、模型路由、调度与性能。
