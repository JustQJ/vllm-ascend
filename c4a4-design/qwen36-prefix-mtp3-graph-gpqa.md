---
title: Qwen3.6 prefix、MTP3、图模式与 GPQA 验证
type: tech-note
project: vllm-ascend
status: active
tags: [kv-cache, kv-cache-quant, mxfp4, attention-backend, verification]
created: 2026-11-13
updated: 2026-11-13
summary: 九轮GPQA完成，prefix并发严格一致性未通过
---

## 最终结论

已将 200 的容器打包到共享目录，通过用户指定的 `/mnt/share/t00970481/docker_run.sh` 在 **141.61.133.123** 启动 `tp_a4c4_gqa_vllm`。所有模型测试使用启动前核实为健康空闲的 **0、2 卡，TP=2**，共享模型、代码和静态 V scale 路径保持。

BF16/C8A8/C4A4 各三轮完整 GPQA Diamond 已完成。精度测试统一 **MTP=3、prefix caching 关闭、真实 target/draft ACLGraph 开启**，共 1782 个请求全部成功。C4 三轮平均 81.9865%，比 BF16 低 2.0202 pp，比 C8 低 0.6734 pp。该结果是固定校准与原采样配置下的三轮观测，不作为统计显著性结论。

C4 prefix 功能执行成功、真实命中，MTP0/3 的冷/热串行 token 一致；**反序并发的严格 token 一致性未通过**，不能宣称 prefix 全面验收通过。所有本任务服务、worker 和调度进程均已退出，0、2 卡已释放；其他任务未被停止。

## GPQA 结果

每轮 198 题，括号为正确数。三个模式使用同一 BF16 checkpoint，改变的是 target GQA KV/attention 量化路径；GDN 保持原生实现，native MTP draft 均使用 BF16 cache。

| 模式 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 平均 | 样本标准差 |
|---|---:|---:|---:|---:|---:|
| BF16 | 83.3333%（165） | 83.3333%（165） | 85.3535%（169） | **84.0067%** | 1.1664 pp |
| C8A8 | 81.8182%（162） | 84.3434%（167） | 81.8182%（162） | **82.6599%** | 1.4580 pp |
| C4A4 | 83.3333%（165） | 82.8283%（164） | 79.7980%（158） | **81.9865%** | 1.9121 pp |

所有轮次均为 198 成功 / 0 失败、0 prefix query/hit、0 preemption；MTP draft 和 accepted 计数均大于零。每轮逐题 prompt、选项、参考答案及顺序完全一致。各模式三轮两两比较，198 条完整响应中均为 0 条完全相同；独立 work-dir 和实际请求记录证明未复用旧响应。

| 模式 | 达到 32768 输出上限 / 无效答案提取（第 1/2/3 轮） | MTP token 接受率（第 1/2/3 轮） |
|---|---|---|
| BF16 | 3 / 2 / 1 | 73.20% / 73.29% / 73.11% |
| C8A8 | 3 / 1 / 3 | 73.45% / 72.70% / 73.37% |
| C4A4 | 6 / 4 / 5 | 73.49% / 73.11% / 73.15% |

这些无效提取答案全部按原规则计入分母；未删题、延长输出或为提高分数重跑。审计同时核对结果 SHA256、逐题正确数、AIS Bench 的 Post/Received/Failed 和服务 metrics。

## 固定评测配置

- 模型：`/mnt/share/weight/Qwen3.6-35B-A3B`，实际架构 `Qwen3_5MoeForConditionalGeneration`，10 GQA + 30 GDN 层。TP2 后每 rank Q8/KV1/D256。
- 输入/输出配置完整取自 `/mnt/share/t00970481/benchmark_cp/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`。保留 max_out_len=32768、batch_size=32、stream=false、temperature=0.6、top_k=20、top_p=0.95、原后处理；源配置未设置 seed，正式运行也不新增 seed。仅替换模型路径、服务名/abbr 和端口。
- 数据：`gpqa_gen_0_shot_cot_chat_prompt.py`，GPQA Diamond 全部 198 题，原 CoT chat prompt、选项轮转和答案提取规则。源模型配置、数据集配置及 CSV 内容快照和 hash 均保留。
- 服务：TP2，devices=0,2，gpu-memory-utilization=0.75，启动前要求 idle、health=OK 且额外预留 8192 MiB；max-model-len=65536，max-num-seqs=32，max-num-batched-tokens=2048。
- 图模式：MRV2 原生 ACLGraph `FULL_DECODE_ONLY`，capture sizes 4/8/16/32/64/96/128，compilation mode=0，enable_npugraph_ex=false、enable_static_kernel=false。mode=0 关闭 Torch 编译变换，decode 仍真实 capture/replay；target 与 draft 的实际回放日志均已核验。
- Native MTP：num_speculative_tokens=3、draft enforce_eager=false、draft kv_cache_dtype=auto/BF16。正式精度测评不启用 prefix caching。
- C4：Q/K 动态原生 MXFP4；V 使用每层本地 KV head、每通道静态 E8M0 scale；packed PA_BBND，正式 QFA mode5、quant_schema=2。使用首轮 BF16 校准的 20 个 sidecar（10 层 × 2 ranks），未使用 GPQA 答案调整 scale。

## 修复后 prefix 与 MTP 复测

两种 MTP 设置均开启图模式和 prefix；使用 20037/20035-token 两个共享长前缀，temperature=0、seed=0、每次最多生成 128 tokens。每组执行 cold、warm、branch，以及两条反序并发请求。

| 设置 | 成功请求 | prefix 命中 tokens | 冷/热串行 | 反序并发严格比较 | 图回放 |
|---|---:|---:|---|---|---|
| C4 + MTP0 | 5/5 | 65536 | token IDs 相同 | 失败：water/solar 首差索引 4/31 | target 已确认 |
| C4 + MTP3 | 5/5 | 65536 | token IDs 相同 | 失败：water/solar 首差索引 4/18 | target/draft 已确认 |

首差索引从 0 开始。MTP3 实际 drafted=627、accepted=434。MTP0 与 MTP3 的 cold、warm、branch 三条串行输出逐 token 相同；两条并发输出不同。未放宽严格检查断言，测试退出码 1 和完整响应都保留。

这说明真实 prefix 复用和 MTP 提议/接受可执行，但当前实现不满足本组并发严格一致性要求。此前正式 QFA batch 探针观察到分段随 batch 改变、最大约 1.53e-5 的数值差异，CPU kernel reference 同样复现；尚未证明该现象是整网差异的全部原因，不能把现存差异直接判定为缓存损坏或已完全解释。GPQA 明确关闭 prefix，因此其九轮结果与此限制分开报告。

MTP3 首次复测启动在 HCCL profile 阶段失败：设备侧 `libccl_kernel.so` 的 `RunAicpuCommInit` 符号不存在，errcode=11003。底层日志已保留。失败进程退出、设备重新确认空闲/健康后，使用相同代码和配置重启成功并完成上述测试；未修改 QFA、回退 eager 或重跑已完成 GPQA。缺符号错误的环境根因仍未进一步确认。

## 本轮修复与回归

1. 接入原生 `mtp` 标准名，校验 draft HF 架构和 BF16 cache；按实际 cache spec dtype 区分 target packed cache 与合并后丢失 is_draft 标记的 BF16 draft。
2. 修正混合页预算：`state_content_bytes` 显式定义的 C4/C8 packet 保持逻辑 block，统一物理页 stride padding；避免当成普通 K/V 再对齐。
3. 修正静态 V scale 所有权。MTP 混合规划下 41 个 cache group 可共用物理页池；旧全池初始化/回收填充让每层读到末层 scale。C8/C4 现在在 KV 写入后，仅向当前层 slot 所属的 64-token 单元写入本层两份 32-token 通道 scale，随后发送 KV 写入完成通知。移除 V2 全池广播及 `C8KVBlockZeroer`。
4. 补齐尾部 scale pair。只写有效半组时，毒化未使用 scale 可令正式 QFA 输出 NaN；必须写完整 64-token 单元的两份静态 scale，不能依赖 mask 排除无效反量化。

根因证据包括两个 rank、10 层共 80 份实际 QFA 输入/输出快照：旧 Vscale 全部等于末层 256/256 通道，而该样例 Q4 与逐 Q1 算子结果逐位相同。修复没有更改 QFA 数学实现或精度门限。

最终验证：112 项 CPU 回归（109 attention + 3 runner）；12 组真实 NPU 整块 allocation 字节回归（C4/C8、D128/256、H1/2、padding、非零 offset、负 slot、页回收、多 owner、图回放）；26 组正式 QFA poisoned-tail eager/graph 边界测试（L1/17/33/65/513，Q1/4/11，N2TGD/TND）全部通过。36 份 Python 源码通过 Ruff lint/format，并在本地与 123 逐文件 SHA256 核验一致。

## qscale 布局

按 `G × max_query_len <= 80` 选择 **N2TGD**，否则 **TND**，不按 batch 总 token 数选择。本模型 TP2 的 G=8，因此 Q<=10 用 N2TGD：普通 decode Q=1 和 MTP3 verify Q=4 均为 N2TGD；长 prefill 用 TND。metadata 与主算子保持一致。

## 环境、版本与证据

123 为 Ascend950DT_9574 / DAV3510；镜像 `c4a4-qfa-runtime:20261113`，ID `6ce8a39d1cef8e445c404d5127c899dc05f1f7ff7aa84415c258fe45be545238`。容器归档放在 `/mnt/share/t00970481/A4C4-Quant/container-migration-20261113`，200/123 双端 SHA256 和加载 image ID 一致。用户的 docker_run.sh 已备份并只增加镜像/容器名参数覆盖，默认值保留。

环境为 CANN9.2、Torch2.10.0+cpu、torch_npu2.10.0.post7、Triton3.6 dev20261013220622；上游 vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`，插件 `gqa-c4a4@3288b04c4` 加本轮未提交改动；ops-transformer `f9067e085`，生产提交 `f42ff75fe`，正式隔离 vendor `c4vllm_transformer`。没有创建提交、推送或 PR。

正式运行目录为 `/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/gpqa-mtp3-graph-123-20261113-r6`。BF16 三轮沿用已审计的 r3；其路径未使用本轮修改的量化 scale writer。r5 C8 结果因尾部 scale pair 缺陷全部作废，不进入最终统计。修复后 prefix 分别来自 `prefix-after-scale-pair-fix-123-20261113-r2` 的 MTP0 与 `prefix-after-scale-pair-fix-123-20261113-mtp3-retry` 的 MTP3。

- [最终 GPQA 摘要](../artifacts/gpqa-mtp3-graph-123-20261113-r6/gpqa-r6/gpqa-summary.json)、[CSV](../artifacts/gpqa-mtp3-graph-123-20261113-r6/gpqa-r6/gpqa-summary.csv)、[九轮逐项审计](../artifacts/gpqa-mtp3-graph-123-20261113-r6/gpqa-audit.json)。
- [修复后的 prefix/MTP 报告](../artifacts/gpqa-mtp3-graph-123-20261113-r6/gpqa-r6/prefix-validation-report.json)、[图回放/独立响应/源码核查](../artifacts/gpqa-mtp3-graph-123-20261113-r6/gpqa-final-evidence-check.json)。
- [完整证据包](../artifacts/gpqa-mtp3-graph-123-20261113-r6/qwen36-gpqa-prefix-final-123-20261113.tar.gz)：244 份证据及 SHA256 索引，含九轮逐题结果、原配置、metrics、日志、静态 scale、36 份冻结源码和 prefix 启动异常底层日志。归档 SHA256：`751d475f64af37007fa25b3b9d50014ce546b3dea8784f43179619f0235d86a2`，远端逐文件复核及本地下载 hash 均通过。
- [最终 scale 修复算子证据](../artifacts/gpqa-mtp3-graph-123-20261113-r6/c8-mtp-diagnostic-scale-pair-final-20261113.tar.gz)，SHA256：`cbc9deda1b1c15c89945263ad95f969ec73168a95b5d18f7f736c9e8c10493a1`。
- [最终设备快照](../artifacts/gpqa-mtp3-graph-123-20261113-r6/gpqa-r6/final-npu-smi.txt)和[进程清理核验](../artifacts/gpqa-mtp3-graph-123-20261113-r6/gpqa-r6/final-process-check.json)。

迁移及排障的原始过程另存 [历史记录](qwen36-prefix-mtp3-graph-gpqa-history.md)，其中等待、旧故障和未完成状态不代表最终状态。设计见 [接入方案](qwen36-c4-qfa-static-v-integration.md)，11-12 eager 首轮记录见 [接入验证](qwen36-c4-qfa-static-v-validation.md)。
