# MLA 计算流、采集点与 TP 聚合

## 计算流

黄色节点表示当前已经保存的数据。

```mermaid
flowchart TD
    H["hidden_states<br/>[T, hidden_size]"]

    H --> A["fused_qkv_a_proj"]
    A --> QC["q_c<br/>压缩 Q"]
    A --> KVNS["kv_no_split"]

    QC --> QNORM["Q RMSNorm"]
    QNORM --> QB["q_b_proj<br/>Column Parallel"]
    QB --> QNOPE["q_nope<br/>[T, Hlocal, P]"]
    QB --> QRPRE["采集 q_rope_pre<br/>[T, Hlocal, R]"]

    QNOPE --> ABSORB["乘 W_UK 转置<br/>吸收到 KV latent 空间"]
    ABSORB --> QLAT["采集 q_latent<br/>[T, Hlocal, Lkv]"]

    QRPRE --> QROPE["RoPE"]
    QROPE --> QRPOST["采集 q_rope_post<br/>[T, Hlocal, R]"]

    KVNS --> KVPRE["采集 kv_latent_pre_norm<br/>[T, 1, Lkv]"]
    KVNS --> KRPRE["采集 k_rope_pre<br/>[T, 1, R]"]

    KVPRE --> KVNORM["KV RMSNorm"]
    KVNORM --> KVLAT["采集 kv_latent<br/>[T, 1, Lkv]"]

    KRPRE --> KROPE["RoPE"]
    KROPE --> KRPOST["采集 k_rope_post<br/>[T, 1, R]"]

    POS["采集 positions<br/>[T]"] --> QROPE
    POS --> KROPE

    KVLAT --> CACHE1["Paged KV Cache<br/>latent KV"]
    KRPOST --> CACHE2["Paged KV Cache<br/>RoPE K"]

    QLAT --> DECODE["Decode MQA"]
    QRPOST --> DECODE
    CACHE1 --> DECODE
    CACHE2 --> DECODE

    KVLAT --> KVB["kv_b_proj<br/>展开 K-nope 与 V"]
    KVB --> PREFILL["Prefill MHA"]
    QNOPE --> PREFILL
    QRPOST --> PREFILL
    KRPOST --> PREFILL

    classDef collected fill:#ffe8a3,stroke:#cc7a00,stroke-width:2px,color:#111;
    class QLAT,QRPRE,QRPOST,KVPRE,KVLAT,KRPRE,KRPOST,POS collected;
```

## TP 中哪些数据会切分

MLA 的 Q projection 是 Column Parallel，Q heads 按 TP rank 切分：

```text
Hlocal = Htotal / TP
```

因此 `q_latent`、`q_rope_pre` 和 `q_rope_post` 在单卡上都只包含本地 Q head 分片，需要沿 head 维 `dim=1` 做 all-gather。

MLA 使用单个压缩 KV head，`W_DKV`/fused A projection 不进行 TP 切分。因此以下数据在纯 TP 模式下每张卡都有完整副本：

- `kv_latent_pre_norm`
- `kv_latent`
- `k_rope_pre`
- `k_rope_post`
- `positions`

它们不能沿 head 维拼接，否则会错误地得到 TP 份重复 KV。

## rank-0 保存流程

```mermaid
flowchart LR
    Q0["TP rank 0<br/>Q head shard 0"] --> AG["TP all-gather<br/>dim=1"]
    Q1["TP rank 1<br/>Q head shard 1"] --> AG
    QN["TP rank N<br/>Q head shard N"] --> AG

    AG --> FULLQ["完整 Q heads"]
    FULLQ --> SAVE["仅 TP rank 0<br/>CPU copy + torch.save"]

    KV0["TP rank 0<br/>完整 latent KV"] --> SAVE
    KV1["其他 TP ranks<br/>重复 latent KV"] --> DROP["不保存"]
```

`all_gather` 的结果会短暂存在于所有 TP rank 的设备内存中，这是 all-gather collective 的语义；但只有 TP rank 0 会将结果复制到 CPU 并落盘。

## 典型模型示例

| 模型 | 全局 Q heads | TP | 每卡 Q heads | 每卡 latent KV heads |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek V3.1 | 128 | 8 | 16 | 1 个完整副本 |
| LongCat-Flash | 64 | 8 | 8 | 1 个完整副本 |

因此离线分析时不需要再读取和拼接其他 TP rank 的文件；rank 0 文件已经包含完整 Q 和一份不重复的 latent KV。
