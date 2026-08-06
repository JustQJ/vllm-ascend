# MLA Q 与 Latent-KV 数据采集

当前分支提供一个默认关闭的调试采集器，用于保存 MLA 的 Q 和 latent-KV 原始张量样本，以支持离线数据分布分析与 KV cache INT2 量化校准。

启用采集会引入 NPU 同步、TP all-gather、NPU 到 CPU 的数据传输和文件 I/O，不应在生产服务中长期启用。

## 采集内容

每个文件对应一个 MLA 层和一个执行阶段，阶段分为 `prefill` 和 `decode`。

| 字段 | 含义 | 典型形状 |
| --- | --- | --- |
| `kv_latent_pre_norm` | KV RMSNorm 前的压缩 latent | `[tokens, 1, kv_lora_rank]` |
| `kv_latent` | KV RMSNorm 后、写入 cache 前的 latent | `[tokens, 1, kv_lora_rank]` |
| `k_rope_pre` | decoupled RoPE K 在 RoPE 前的值 | `[tokens, 1, rope_dim]` |
| `k_rope_post` | decoupled RoPE K 在 RoPE 后的值 | `[tokens, 1, rope_dim]` |
| `q_latent` | `q_nope` 吸收 `W_UK` 后的 latent Q | `[tokens, global_q_heads, kv_lora_rank]` |
| `q_rope_pre` | Q 的 RoPE 分量在 RoPE 前的值 | `[tokens, global_q_heads, rope_dim]` |
| `q_rope_post` | Q 的 RoPE 分量在 RoPE 后的值 | `[tokens, global_q_heads, rope_dim]` |
| `positions` | 与采集 Token 对齐的绝对位置 | `[tokens]` |

保存的 Q 张量已经在 head 维完成 TP all-gather，因此 rank 0 文件里是完整的全局 Q heads。MLA latent KV、RoPE K 和 positions 本来就在 TP rank 之间复制，不会再次拼接。

## 环境变量

```bash
export VLLM_ASCEND_MLA_DATA_DUMP_DIR=/workspace/mla-data
export VLLM_ASCEND_MLA_DATA_DUMP_LAYERS=0,30,60
export VLLM_ASCEND_MLA_DATA_DUMP_MAX_TOKENS=128
export VLLM_ASCEND_MLA_DATA_DUMP_MAX_STEPS=1
```

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VLLM_ASCEND_MLA_DATA_DUMP_DIR` | 未设置 | 输出目录；未设置时完全关闭采集 |
| `VLLM_ASCEND_MLA_DATA_DUMP_LAYERS` | 空 | 层索引或完整层名，逗号分隔；空表示全部层 |
| `VLLM_ASCEND_MLA_DATA_DUMP_MAX_TOKENS` | `128` | 每个阶段、每一步最多保存的当前 Token 数 |
| `VLLM_ASCEND_MLA_DATA_DUMP_MAX_STEPS` | `1` | 每层、每阶段最多保存的文件数 |

## 启动示例

```bash
vllm serve /models/DeepSeek-V3.1 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enforce-eager
```

采集器跳过 ACLGraph capture。为了确保 Python 采集路径实际执行，校准运行应使用 `--enforce-eager`。

## 保存策略

实现参考 OSCAR 的 TP 采集模式：

1. 所有 TP rank 对 Q 系列张量调用 TP group 的 `all_gather(dim=1)`。
2. 只有 TP rank 0 执行 `.cpu()` 和 `torch.save()`。
3. 已复制的 latent KV、RoPE K 和 positions 只保存 rank 0 本地完整副本。
4. 单个 TP group、单层、单阶段、单步只产生一个文件。

如果启用了 DP 或 PP，每个 TP group 会有一个 writer，以免丢弃不同请求或不同 pipeline stage 的数据。

## 文件格式

输出为 `torch.save` 的 format version 2 字典：

```text
format_version
layer_name
phase
rank
tp_rank
tp_size
step
metadata
tensor_metadata
tensors
```

`tensor_metadata` 包含原始形状、保存形状、dtype 和 TP gather 维度；`tensors` 包含实际 CPU Tensor。

## 当前限制

- 主要面向 BF16/FP16 MLA 基线分布采集。
- A5 FA 动态量化路径中的 Q 在量化前采集。
- `kv_latent` 使用与正式 cache 写入路径相同的 RMSNorm weight 和 epsilon 重新计算，不是从 paged cache 读回。
- fused MLAPO 或量化 decode preprocess 可能绕过 Python 采集点，校准时应关闭 MLAPO 并使用未量化 checkpoint。
- Context Parallel 使用独立 MLA 实现，当前采集器尚未覆盖。
- 当前保存的是当前步骤样本，不是完整历史 paged KV cache。
- 激活数据可能包含由用户输入派生的敏感信息，输出目录必须妥善保护并在分析完成后清理。
