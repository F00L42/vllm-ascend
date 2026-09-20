# Round 模板

状态：draft。复制到 `rounds/<round_id>/report.md` 后填写；此文件不代表已有运行结果。

## 目标与版本

- round_id / 计划阶段：
- 本轮唯一假设：
- 候选 tag / 完整父仓 SHA：
- upstream 基点 / vLLM、xLLM、xLLM-ops、PTO、CATLASS SHA：
- 各仓 dirty 状态及补丁来源：
- 与上一轮的代码/依赖语义变化：

## 修改和验收范围

- 文件与目的：
- ABI、数值、状态生命周期变化：
- 本轮声明支持 / 必须 fallback 的范围：
- 精确比较条件 / 浮点容差及依据：

## 本地检查

| 命令 | 结果 | 证据 |
| --- | --- | --- |
| 待填写真实命令 | pending | 待填写 |

注明 failed / skipped_env，不能把尚未运行写成 pass。包括适用测试、`bash format.sh ci` 结果和修复后的 diff 检查。

## 服务器执行单

- 镜像 digest、SoC、工具链和环境准备步骤：
- fetch/tag/SHA 核验及本轮独立目录：
- 依赖准备、构建安装、实际加载核验命令：
- 测试命令、seed、参数、预期结果：
- 路由强制命中或实际 kernel 调用的证据：
- 默认关闭的诊断配置及捕获范围：
- 失败后停止位置、必须保留的日志和 launch 前快照：
- 回传目录、文件列表和校验和：

发布候选前替换全部占位项，确认测试入口真实存在。构建失败也保留环境清单、完整 build.log 和退出码。

## 结果和下一步

- 执行来源是否与候选一致：
- passed / failed / skipped / 未执行用例数：
- 首个分歧、fallback、污染区域或异常：
- 已验证配置及证据 run_id：
- 当前状态：local_ready / pending_remote / failed / validated_scope。
- 不能由本轮证明的内容：
- 下一轮最小假设：
