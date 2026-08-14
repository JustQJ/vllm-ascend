# MLA Q 与 Latent-KV 数据采集

当前分支提供一个默认关闭的调试采集器，用于保存 MLA 的 Q 和 latent-KV 原始张量样本，以支持离线数据分布分析与 KV cache INT2 量化校准。

启用采集会引入 NPU 同步、TP all-gather、NPU 到 CPU 的数据传输和文件 I/O，不应在生产服务中长期启用。

## 采集内容

每个文件对应一个 MLA 层和一个执行阶段，阶段分为 `prefill` 和 `decode`。

输出根目录下按完整层名创建子目录，层名中的不安全字符会替换为下划线。例如：

```text
/workspace/mla-data/
├── model.layers.0.self_attn.attn/
│   ├── prefill.rank0.pid1234.step0000.pt
│   ├── decode.rank0.pid1234.step0000.pt
│   └── decode.rank0.pid1234.step0001.pt
├── model.layers.13.self_attn.attn/
│   └── ...
└── model.layers.26.self_attn.attn/
    └── ...
```

`prefill` 和 `decode` 的 step 在每个层内独立计数。TP 启动时所有 rank 参与 Q 的
all-gather，但每个 TP group 仍然只有 rank 0 在相应层目录写一份完整数据。

| 字段 | 含义 | 典型形状 |
| --- | --- | --- |
| `kv_latent_pre_norm` | KV RMSNorm 前的压缩 latent | `[tokens, 1, kv_lora_rank]` |
| `kv_latent` | KV RMSNorm 后、写入 cache 前的 latent | `[tokens, 1, kv_lora_rank]` |
| `k_rope_pre` | decoupled RoPE K 在 RoPE 前的值 | `[tokens, 1, rope_dim]` |
| `k_rope_post` | decoupled RoPE K 在 RoPE 后的值 | `[tokens, 1, rope_dim]` |
| `q_nope_prefill` | prefill FIA 实际使用的非 RoPE Query；只存在于 prefill 文件 | `[tokens, global_q_heads, qk_nope_head_dim]` |
| `q_latent` | `q_nope` 吸收 `W_UK` 后的 latent Q | `[tokens, global_q_heads, kv_lora_rank]` |
| `q_rope_pre` | Q 的 RoPE 分量在 RoPE 前的值 | `[tokens, global_q_heads, rope_dim]` |
| `q_rope_post` | Q 的 RoPE 分量在 RoPE 后的值 | `[tokens, global_q_heads, rope_dim]` |
| `positions` | 与采集 Token 对齐的绝对位置 | `[tokens]` |

保存的 Q 张量已经在 head 维完成 TP all-gather，因此 rank 0 文件里是完整的全局 Q heads。MLA latent KV、RoPE K 和 positions 本来就在 TP rank 之间复制，不会再次拼接。

### 保存张量的计算角色

这里需要区分两类 NPU 算子：

1. **KV 预处理与 cache 写入算子**：`npu_kv_rmsnorm_rope_cache`，负责对 latent KV
   执行 RMSNorm、对 RoPE K 执行旋转，并写入 paged KV cache；
2. **MLA attention 计算算子**：prefill 使用
   `npu_fused_infer_attention_score`，decode 使用
   `npu_fused_infer_attention_score_v2`。

| 保存字段 | 写入 paged KV cache | KV 预处理算子 | decode MLA attention | prefill MLA attention |
| --- | --- | --- | --- | --- |
| `kv_latent_pre_norm` | 否 | 是，属于 `kv_no_split` 的 latent 输入部分 | 否 | 否 |
| `kv_latent` | **是，写入 `kv_cache[0]`** | 是，RMSNorm 后的逻辑输出 | **是，同时作为 latent K 和 latent V** | 间接使用：经 `kv_b_proj` 上投影成完整 `k_nope` 和 `value` |
| `k_rope_pre` | 否 | 是，属于 `kv_no_split` 的 RoPE K 输入部分 | 否 | 否 |
| `k_rope_post` | **是，写入 `kv_cache[1]`** | 是，RoPE 后的逻辑输出 | **是，作为 `key_rope`** | **是，作为 `key_rope`** |
| `q_nope_prefill` | 否 | 否 | 不存在于 decode 文件 | **是，作为 `query`** |
| `q_latent` | 否 | 否 | **是，作为吸收 `W_UK` 后的 Query** | **否，仅为量化分析额外计算** |
| `q_rope_pre` | 否 | 否；它是 Q RoPE 算子的输入 | 否 | 否 |
| `q_rope_post` | 否 | 否；它是 Q RoPE 算子的输出 | **是，作为 `query_rope`** | **是，作为 `query_rope`** |
| `positions` | 否 | 不作为张量输入；用于取得 RoPE `cos/sin` 和保持 token 对齐 | 否 | 否 |

因此，真正写入 paged KV cache 的只有：

```text
kv_cache[0] <- kv_latent       # RMSNorm 后的 512 维 latent KV
kv_cache[1] <- k_rope_post     # RoPE 后的 64 维 decoupled K
```

`kv_latent_pre_norm` 和 `k_rope_pre` 是 cache 写入算子处理前的输入样本，不会原样
保存在 cache 中。`q_nope_prefill`、`q_latent`、`q_rope_pre`、`q_rope_post` 和
`positions` 也不会写入 KV cache。

### Decode 算子的对应关系

经典 MLA decode 的逻辑调用关系为：

```text
npu_fused_infer_attention_score_v2(
    query      = q_latent,
    key        = paged_cache.kv_latent,
    value      = paged_cache.kv_latent,
    query_rope = q_rope_post,
    key_rope   = paged_cache.k_rope_post,
)
```

也就是说，decode 时 `kv_latent` 同时承担 latent K 和 latent V 两个角色。保存文件中的
Q 已经 all-gather 成全局 heads；真正执行算子时，每个 TP rank 使用自己的本地 Q heads，
并可能在进入算子前进行 reshape、head padding 或动态量化。因此保存值和算子逻辑输入
一致，但保存 shape 不一定等于单个 rank 的物理输入 shape。

### Prefill 算子的对应关系

Prefill 不直接使用吸收后的 `q_latent` 做 attention，而是采用完整 Q/K/V 路径：

```text
kv_latent --kv_b_proj--> k_nope, value

npu_fused_infer_attention_score(
    query      = q_nope_prefill,
    key        = k_nope,
    value      = value,
    query_rope = q_rope_post,
    key_rope   = k_rope_post,
)
```

所以 `q_nope_prefill` 是该次 prefill NPU attention 算子的实际非 RoPE Query；
`q_latent` 则是 collector 为后续 C8A2、INT2 KV 等实验额外计算的对照数据，**不是**
prefill 算子的直接输入。`kv_latent` 先经过 `kv_b_proj` 上投影，间接生成算子使用的
完整 K/V。

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
4. 单个 TP group、单层、单阶段、单步只在该层子目录产生一个文件。

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
- `kv_latent` 和 `k_rope_post` 使用与正式 cache 写入路径相同的 RMSNorm/RoPE 参数
  重新计算，不是从 paged cache 读回；若启用量化 cache，保存的是量化前的逻辑参考值，
  不代表 cache 中的物理 dtype、布局和 scale 元数据。
- fused MLAPO 或量化 decode preprocess 可能绕过 Python 采集点，校准时应关闭 MLAPO 并使用未量化 checkpoint。
- Context Parallel 使用独立 MLA 实现，当前采集器尚未覆盖。
- 当前保存的是当前步骤样本，不是完整历史 paged KV cache。
- 激活数据可能包含由用户输入派生的敏感信息，输出目录必须妥善保护并在分析完成后清理。
