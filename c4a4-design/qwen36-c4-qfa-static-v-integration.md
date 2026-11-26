---
title: Qwen3.6 静态 V 的 C4 QFA 接入方案
type: tech-note
project: vllm-ascend
status: implemented
tags: [kv-cache, kv-cache-quant, mxfp4, attention-backend, block-manager, quantization]
created: 2026-11-12
updated: 2026-11-13
related: []
summary: BF16 权重、静态 V、动态 Q/K 和正式 mode5 QFA 的 TP2 eager 接入
---

## 实现范围

已在 200 服务器的 `tp_a4c4_gqa_vllm` 容器完成 BF16 校准、C8A8 基线和 C4A4 整网接入。实际结果、版本、运行入口与限制见 [验证记录](qwen36-c4-qfa-static-v-validation.md)。本说明已按用户要求从共享知识库移到本项目 `c4a4-design`，后续详细记录保存在本项目。

源码基准：本项目分支 `gqa-c4a4`，HEAD `3288b04c4`；ops-transformer 分支 `quant-flash-attn-c4a4-handwrite-softmaxfp32`，HEAD `f9067e085`，生产实现 `f42ff75fe`。参考服务器 `tmp2/vllm-ascend` 的 `qwen3.6-c4a4@c6d7e7d58` 校准流程，没有覆盖其源码，也没有沿用旧 C7 的 `/7` 量化配方。

实际模型 `/mnt/share/weight/Qwen3.6-35B-A3B` 是 BF16 checkpoint，结构为 10 个 full-attention GQA 层和 30 个 GDN 层。全局 Q16/KV2/D256，TP2 后每 rank Q8/KV1/D256。仅兼容的 decoder GQA 层安装 C4 后端；GDN 保持原有 dtype 和状态管理。

首版已经实现：eager、TP2、PP1、文本生成、chunked prefill、decode、混合长度并发、静态 V scale、真实 packed KV cache、正式 QFA mode5。2026-11-12 首轮范围不包括图模式、MTP、PD/PCP/DCP、跨 TP 复用 sidecar，prefix caching 在首轮模型验证中关闭。2026-11-13 后续已接通原生 MTP3 和 ACLGraph；修复共享池静态 Vscale 所有权与末尾 scale pair 后，已完成三精度各三轮 GPQA。prefix 冷/热串行一致，反序并发严格 token 一致性仍未通过；最终结论见 [后续验证](qwen36-prefix-mtp3-graph-gpqa.md)。

## 数据流和静态 V

模型原有 Q/K norm、RoPE 等变换完成后：

1. Q/K 使用 `npu_dynamic_mx_quant`，按 D32 动态量化成 E2M1 和 E8M0。
2. V 使用每层、每个本地 KV head、每个通道固定的 E8M0 scale。
3. Triton writer 将 packed K、动态 Kscale 和量化后的 packed V 写到指定物理页/slot，并为本层实际写入的 64-token 单元填充完整 Vscale pair。
4. 正式 QFA 使用 `quant_mode=5`、`PA_BBND`、`quant_schema=2`，输出 BF16。

V 标定保存原始逐通道 `amax`，分别生成 C8/C4 scale：

```text
C8: scale = 2^ceil(log2(amax / 448))
C4: scale = 2^ceil(log2(amax / 6))
zero channel: scale = 1
E8M0 stored byte = exponent + 127
```

C8 的编码 scale 不直接复用于 C4。缺文件、模型路径/TP/layer/rank/shape 不匹配或 maxima 非有限/负数均报错。sidecar 是本次 TP 的本地 head 数据，需要更换 TP 时重新校准；当前只校验模型路径，不提供 checkpoint 内容哈希校验。

`mxfp_static.mode=calibrate` 仍执行普通 BF16 attention。服务就绪后创建 `scale_dir/calibration.enabled` 才开始观测，只统计 `num_actual_tokens`，排除 profiling 与 padding；逐层原子保存 maxima、模型路径、TP、rank 和观测 token 数。校准结束移除标记。加载 hook 在权重加载完成后、cache spec 和显存 profiling 之前运行，适用于没有 ModelSlim recipe 的 BF16 checkpoint。

目标 torch_npu 的 `npu_quantize` 静态 FP4 路径返回 ACL161002，动态 Q/K 路径可以工作。因此 V 采用 Triton 编码：FP32 乘倒数 scale，E2M1 最近偶数舍入，饱和到幅值 6，偶数逻辑维度放低半字节，奇数维度放高半字节，保留负零。没有完整 BF16 历史缓存或 PyTorch 数学回退路径。

## Packed 缓存布局和预算

P 为 kernel page 数，S 为每页 token 数，H 为本 rank KV heads，D 为逻辑 head dim：

```text
K,V       [P,S,H,D/2]       packed E2M1，Python storage 为 uint8
Kscale    [P,S,H,D/64,2]    E8M0
Vscale    [P,S/64,H,D,2]    E8M0
Q         [T,Hq,D/2]       packed E2M1
QscaleDN  [T,Hq,D/64,2]
QscaleND  [H,T,Hq/H,D/64,2]
```

Vscale 沿页、64-token 组以及两条 32-token lane 广播同一通道常量。每页底层是 `[K | Kscale | V | Vscale | padding]` 字节包，四个 view 共用页 pitch，保留 storage offset；算子入口才 bitcast 到正式 FP4/E8M0 dtype。C++ 桥接负责将 FP4 字节 stride/offset 转为逻辑半字节单位。

每 token、每 KV head 的 GQA 未 padding 载荷为：

| D256 | BF16 | C8 | C4 |
|---|---:|---:|---:|
| bytes | 1024 | 528 | 272 |

`FullAttentionSpec.state_content_bytes = (D+Dv)/2 + (D+Dv)/32`，逻辑 D 不变。模型加载前的 hybrid page alignment 也通过 C4 `customize_spec` 使用真实打包大小，包括合成的一 token spec；否则 uint8 的 element_size 会按未打包宽度预算，使 GDN padding 掩盖容量收益。

kernel block 固定 512；scheduler block 由混合管理器选择，为 kernel block 的整数倍。按 r=scheduler/512 展开页表，kernel page pitch 为 scheduler page bytes/r。view 构造检查页容量和可整除性，支持非零 storage offset 和页 padding。

2026-11-13 已改为按当前层实际拥有的 slot 写入静态 Vscale。MTP 混合规划可让多个 cache group 共用同一物理页池，初始化或回收时对每层全池广播会使各层读到最后一层的 scale，因此已移除全池填充及 `C8KVBlockZeroer`。C8/C4 在写入 KV 后调用 `scatter_v_scales`，仅更新当前 slot 所属的 64-token 单元，两条 32-token lane 均写入本层通道常量；必须补齐最终半组，否则 QFA 对未初始化 scale 的计算可产生 NaN。页回收仍使用原生 zeroer，新 owner 在实际写入时恢复自己的 scale，随后才发送 KV 写入完成通知。

## 代码入口

| 文件 | 作用 |
|---|---|
| `vllm_ascend/attention/mxfp_static.py` | 显式启用校准、sidecar 校验/加载、scale 编码、选择匹配 QFA 包 |
| `vllm_ascend/attention/attention_c4_mxfp.py` | BBND spec/view、C4 backend、动态 Q/K、正式 mode5 路由 |
| `vllm_ascend/ops/triton/mxfp4_cache.py` | 静态 V E2M1 编码及 K/V/Kscale 定址写入 |
| `vllm_ascend/ops/triton/mxfp_v_scale.py` | C8/C4 共用的按 slot 写入 Vscale pair，支持共享页池、页 padding 和图回放 |
| `vllm_ascend/quantization/methods/kv_cache/mxfp_c4.py` | 复用 C8 参数处理并安装 C4 后端 |
| `vllm_ascend/attention/attention_c8_mxfp.py` | 共用 QFA 执行框架、qscale 布局选择和 Vscale 定址写入；按 mode/layout 隔离 metadata |
| `vllm_ascend/platform.py` | 加载模型前的 C4 hybrid 页预算 |
| `vllm_ascend/patch/{platform,worker}/patch_kv_cache_dtype.py` | `mxfp4` 配置和 worker dtype 往返 |
| `vllm_ascend/worker/v2/attn_utils.py` | packed spec、raw allocation 和四个 cache views |
| `vllm_ascend/worker/v2/{model_runner,utils}.py` | 加载 hook、原生页清零；不再对共享池广播静态 scale |

现有 BF16 checkpoint 使用独立 additional_config 入口，不修改 ModelSlim recipe。配置字段集中于 `mxfp_static`，支持 `mode`、`scale_dir`、`qfa_package`、`v_scale_divisor`、`quant_schema`。

## 正式 QFA 调用

metadata 和主算子都使用 mode5/PA_BBND，Q/output 为 TND；`cu_seqlens_q` 是 int32[B+1]，`seqused_kv` 是 int32[B]。首版始终使用 causal mask_mode3。长 prefill Qscale 用 TND/DN，decode/短 Q 用 N2TGD/ND，复用 C8 阈值并已验证这两种布局；未宣称该阈值性能最优。

主算子使用 layer.scale，`p_scale/sinks/v_tail=None`。C4 当前不支持 V tail/sinks。metadata cache key 包含 mode、layout、头数、维度、mask 和序列信息，C8/C4 不共用 plan。正式 OPP 和 Python/C++ 扩展必须来自匹配的新增 quantSchema ABI，本次以隔离 vendor 安装并真实调用验证。

## 后续验收

首轮 eager 结果见验证记录；MTP3、真实 target/draft ACLGraph、三精度三轮 GPQA 和 prefix cold/warm 的最新结果见后续验证报告。继续扩大代表性校准语料、进行 logits 比较及验证 PD 时需单独验收。三个 P quant_schema 的质量对比和 BF16/C8/C4 同口径吞吐、延迟、峰值显存测试也仍需单独执行。

### 静态 FP4 原生接口限制的完整证据（2026-11-13 补核）

重新读取首次探针的 CANN 完整 plog：`torch_npu.npu_quantize` 已将 FP4 类型传到 `aclnnAscendQuantV3`，其 `CheckDtypeValid` 明确报 `Tensor y not implemented for DT_FLOAT4_E2M1`，支持输出列表为 INT8、INT32、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8。这个结论限定于本次 CANN9.2 / torch_npu2.10.0.post7 组合的静态量化路径；不是参数 shape 推断，也不是全部原生 FP4 算子都不可用。INT4 与所需 FP4 E2M1 编码不同，不能直接替换；`npu_dynamic_mx_quant` 的动态 FP4 已实测可用。此次只读历史日志，没有新增设备测试或修改生产代码。原始完整错误已归档为 `artifacts/c4-qfa-integration-20261112/runtime/logs/fp4-static-dtype-error.log`。
