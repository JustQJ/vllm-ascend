# MLA 相关模型兼容性

本文记录 `mla-quant` 分支基于 `release/v0.23.0rc1` 的代码级兼容性结论。除非另外记录真实 Ascend 运行日志，否则“支持”表示仓库已有代码、支持矩阵或 nightly 配置依据，不代表本地已经完成真权重运行验收。

## DeepSeek V3.1

状态：支持。

依据：

- 仓库支持矩阵明确列出 DeepSeek V3/3.1。
- 存在 `DeepSeek-V3.1-BF16` 多节点 nightly 配置。
- 当前 MLA 实现覆盖 DeepSeek V3.1 使用的 Q/KV LoRA、decoupled RoPE 和 latent KV cache 路径。

相关文件：

- `docs/source/user_guide/support_matrix/supported_models.md`
- `tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-V3.1-BF16.yaml`

## LongCat-Flash

状态：实验性支持原始 `LongcatFlashForCausalLM` 架构。

对应的官方 checkpoint 系列包括：

- `meituan-longcat/LongCat-Flash-Chat`
- `meituan-longcat/LongCat-Flash-Thinking`

两者的核心配置为 28 层、hidden size 6144、64 个 attention heads、512 个 routed experts、`kv_lora_rank=512`、`q_lora_rank=1536` 和 131072 最大上下文。

LongCat-Flash 支持在 vLLM Ascend v0.13.0rc2 引入，并在 v0.13.0 release note 中标记为实验性支持；当前分支仍保留 `longcat_flash` 的模型 runner、speculative config、量化及 KV connector 适配。

## LongCat-Flash-Lite

状态：当前不能确认开箱即用，预计需要独立适配。

原因：

- LongCat-Flash-Lite 声明的架构是 `LongcatFlashNgramForCausalLM`。
- 当前 vLLM v0.23.0 原生模型注册表注册的是 `LongcatFlashForCausalLM`。
- Lite 不能仅凭名称视为原始 LongCat-Flash 的小参数版本，它引入了不同的 Ngram architecture 入口。

在为 Lite 做真实启动验证之前，需要至少检查：

1. vLLM architecture registry。
2. Lite 的 remote modeling code 与当前 Transformers 版本兼容性。
3. 权重 key、Q/KV projection 和 Ngram 模块映射。
4. Ascend MLA backend 是否进入预期路径。
5. 真权重启动、首个请求、ACLGraph/EP 等功能验证。

## 当前运行验证边界

当前本地环境没有 Ascend NPU 和模型权重，因此尚未完成以下验证：

- DeepSeek V3.1 真权重服务启动及请求。
- LongCat-Flash Chat/Thinking 真权重服务启动及请求。
- LongCat-Flash-Lite architecture adaptation。
- TP all-gather 在真实 HCCL 环境下的运行验证。
