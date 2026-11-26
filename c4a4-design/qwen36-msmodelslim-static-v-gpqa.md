# Qwen3.6-35B-A3B：msmodelslim 静态 V scale 与 GPQA

日期：2026-11-13。状态：校准、六轮 GPQA、最终审计与本任务进程清理全部完成。

新 C4 均值为 **82.8283%**，比旧 C4 高 **0.8418 个百分点**；新 C8 为 **80.9764%**，比旧 C8 低 **1.6835 个百分点**。这里的 C4 是用户确认的 ModelSlim MXFP4 参数规则静态 V 适配，C8 有仓库原生静态 V 配方。两者均保持 BF16 模型权重。

| 方案 | 三轮正确题数（每轮 198 题） | 均值 | 样本标准差（百分点） | 相比对应旧 scale |
| --- | --- | ---: | ---: | ---: |
| 新 C8 | 160 / 160 / 161 | 80.9764% | 0.2916 | −1.6835 pp |
| 旧 C8（历史） | 162 / 167 / 162 | 82.6599% | 1.4580 | — |
| 新 C4 适配 | 166 / 162 / 164 | 82.8283% | 1.0101 | +0.8418 pp |
| 旧 C4（历史） | 165 / 164 / 158 | 81.9865% | 1.9121 | — |
| BF16（历史参照） | 165 / 165 / 169 | 84.0067% | 1.1664 | — |

本次新增 C8/C4 各 3 轮，共 **1188/1188 成功请求、0 失败**。BF16 沿用上一轮已审计的 3 次结果，没有在本次重跑。所有新测试均 TP2、MTP3、prefix OFF、FULL_DECODE_ONLY，target/draft 的真实 graph replay 均已核验。新 C4 比本次 C8 高 1.8519 pp，比历史 BF16 低 1.1785 pp；这些是本轮观测结果，未作显著性推断。

## 支持范围与本次使用方式

核查本地 msmodelslim 提交 `cb83735f94604d308fa67784b1c7d5b8981addf0`，工作树干净。

| 项目 | 源码结论 | 本次处理 |
| --- | --- | --- |
| Qwen3.6-35B-A3B | 有 `lab_practice/qwen3_5_moe/qwen3_6_35b_a3b_w8a8c8_mxfp8.yaml`，标记 vLLM-Ascend + Ascend_950 | 使用同一模型与该配方指定的校准数据 |
| 静态 C8 V | `fa_v: dtype=mxfp8, scope=per_channel, symmetric=true, method=minmax` | 原生参数计算函数，导出每个 KV head/channel 的 E8M0 |
| 静态 C4 V | 无完整 per-channel C4 IR/配方注册；MXFP4 的 FA3 注册为 per-block 动态 | 使用同一通道统计，调用其 MXFP4 参数函数；明确标注为静态 V 适配 |
| 线性层量化 | 原配方还包含 W8A8 等线性层处理 | 本实验权重、线性计算保持 BF16，只比较注意力 C8/C4 的静态 V scale |

这不是 msmodelslim 全量 CLI W8A8C8 导出实验。采用其校准数据、模型适配器的分词模板、FA3 per-channel 全局 min/max 统计语义和实际 `calculate_qparam` 函数，使用现有 vLLM BF16 服务执行前向收集。用独立双 batch、三 head、不同 channel 的张量核对 `_FA3PerChannelObserver` 与现有 absmax 收集结果完全一致。

## 计算方式

先按 `(head, dim)` 聚合所有有效输入 token：`amax = max(abs(V))`。TP2 的各 rank 分别持有一个 KV head，不能跨 rank 合并通道。

- 旧算法：`e = ceil(log2(amax / F))`，C8 的 `F=448`，C4 的 `F=6`。
- 新 C8：实际 `ir/api/impl/mx_quantization.py::calculate_mx_qparam`，`e = floor(log2(amax)) - 8`，并采用源码的零值与指数范围处理。
- 新 C4 适配：实际 `calculate_mxfp4_qparam`，`e = floor(log2(amax * 8/7 + 9.6e-7)) - 2`。该函数原注册 scope 是 per-block；本次将每通道已归约的 amax 输入该函数，并保留其指数规则。并非另一套 `core/dtype_ops/mxfp4_ops.py` 的 OCP floor 公式。
- 本次参数函数输入为 FP32 amax（原 V 的 BF16 数值可无损表示）。CPU 复核同一批 amax 以 BF16/FP32 计算：C8 的 5120 个指数全部一致；C4 适配有 4 个通道不同。正式实验固定使用最初导出的 FP32 计算结果，未根据分数重新选 scale。
- 落盘 `v_scale_e8m0 = e + 127`，另存 `quant_dtype`、算法名、源码提交和校准集 SHA。运行时直接加载编码结果，避免重新走旧 `ceil` 公式。
- Q/K 动态量化、V 的原生 FP8 / Triton FP4 量化及正式 QFA 计算保持原实现。本实验不是 msmodelslim 伪量化前向与 QFA 全算子逐 bit 对齐测试。

新增加载检查拒绝 dtype/shape 不匹配、NaN 编码及 FP32 无法表示的 scale 倒数；旧 sidecar 没有编码字段时保持原计算方式。31 项相关单元测试通过。真实 NPU 检查 `npu_quantize` 对超出 ±448 的 FP8 输入做有限饱和裁剪。

## 校准数据与隔离

- 数据：`lab_calib/qwen3_cot_vlm_text/index.jsonl`，58 条纯文本 CoT，全文输入，不截断。
- 原始数据 SHA256：`371d68d9220223dace2034982e756e1af0fedcb6bf75dbbaa3c6e373cc366432`。
- 使用 Qwen3_5ModelAdapter 相同的 AutoProcessor `apply_chat_template`，user 的 list-of-parts 文本，`add_generation_prompt=True`。
- token IDs 固定后通过 completions API 发送；合计 127259 tokens，最长 2958 tokens。输入归档 SHA256：`1641cd199abbb2f23f20c62973bbb46bf059e6fdc0cbb8f8199269f43fb30829`。
- 校准 TP2、eager、MTP0、prefix OFF、temperature=0、max_tokens=1；统计真实 prefill token，不将 startup/warmup 张量纳入。
- 新目录：`/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/msmodelslim-vscale-123-20261113`。
- `calibration-raw` 保存新 amax，`scales/c8` 与 `scales/c4` 保存独立编码 sidecar。旧 runtime/calibration 原样保留，并在前后核对全部文件 SHA256。
- 对 GPQA Diamond 198 道完整题干做空白归一化后的包含检查，没有与校准文本直接重合；该检查不代表语义污染鉴定。

## 测试协议

- 123 机器的 `tp_a4c4_gqa_vllm` 容器；重新查询空卡。最初 3 卡启动前被其他任务占用，空卡检查拒绝启动；随后选择 0/4，0 健康、4 Warning，无其他进程。用户明确允许使用不健康空卡。
- 模型 `/mnt/share/weight/Qwen3.6-35B-A3B`；正式测试 TP2、MTP3、prefix OFF、FULL_DECODE_ONLY，真实 target/draft graph replay 必须出现。
- AIS Bench 沿用用户指定 `vllm_api_general_chat.py` 输入输出参数，GPQA Diamond 198 题，C8/C4 各新跑三次。max_out_len=32768、temperature=0.6、top_k=20、top_p=0.95、batch_size=32。
- BF16 参照来自上一轮已审计的 3 次结果，明确标记为历史参照，不计作本轮新跑。
- 每轮保留全部 198 题，长度截断、未抽取出答案均保留；不依据成绩选择性重跑。

## 对比参照

上一轮有效结果：BF16 165/165/169，旧 C8 162/167/162，旧 C4 165/164/158；每轮分母 198。均值分别 84.0067%、82.6599%、81.9865%。本轮结果已完成审计，见开头的对比表。

## 校准完成核对

20 份原始统计全部为 127259 tokens，导出 C8/C4 各 20 份 scale。5120 通道中 4776 个在新数据上观察到更大 amax。

| 对比 | C8 | C4 适配 |
| --- | ---: | ---: |
| 同一新 amax 上，ModelSlim 指数与旧 ceil 公式不同 | 1132/5120 | 978/5120 |
| 相比上一轮实际 scale，新 scale 更大 | 2313 | 2415 |
| 相比上一轮实际 scale，相同 | 2616 | 2506 |
| 相比上一轮实际 scale，新 scale 更小 | 191 | 199 |

本次同时更换校准数据与 scale 指数规则，最终分数差异反映整套方案；不能单独归因于其中一项。

[独立校准包](../artifacts/msmodelslim-vscale-123-20261113/modelslim-vscale-calibration-20261113.tar.gz) 含原始统计、40 份 scale、全部输入 token IDs、58 个请求响应、数据/配方/关键源码和逐文件哈希。SHA256：`b2ed061d5399a3a17fee256dcd884ee0e8f7b6fe1c412e04bf1f1b21e7dd4217`，本地与远端一致。
## 最终审计

| 模式 | 三轮输出达到长度上限 | 三轮无法抽取答案 | 三轮 MTP token 接受率 |
| --- | --- | --- | --- |
| C8 | 4 / 3 / 2 | 4 / 3 / 2 | 73.1207% / 73.5057% / 73.2153% |
| C4 | 3 / 2 / 1 | 3 / 1 / 1 | 74.3084% / 73.6479% / 72.9020% |

长度上限与无法抽取答案的题目均保留在原评分中。全部六轮 prefix 查询/命中为 0、抢占为 0、空响应为 0；每轮 198 道题的题干、选项顺序和参考答案一致。六组新响应内容哈希互不相同，并与旧结果不同。

40 份 scale 和 40 份相关冻结源码哈希核验通过；原 calibration 目录所有文件的前后 SHA256 一致。31 项静态 scale 相关单元测试通过，8 份本次新增/修改 Python 文件通过 Ruff lint/format。C8/C4 各 10 请求语义冒烟通过。任务结束后本任务 API、worker、矩阵驱动进程均已退出。没有修改 msmodelslim 源码或正式 QFA kernel。

本次卡组为 123 的 0/4，4 卡为用户允许使用的 Warning 空卡；旧参照来自原来的 0/2 卡组。数据集与参数公式同时变化，卡组与随机采样也存在差异，表格不用于单独归因某一项变化。

## 接入改动与复用

- [mxfp_static.py](../vllm_ascend/attention/mxfp_static.py)：支持读取 `v_scale_e8m0`，校验量化 dtype、shape、有限 scale/倒数及 divisor 冲突；旧 amax sidecar 继续使用原公式。
- [launch_mxfp_model.py](tools/launch_mxfp_model.py)：增加显式 `--scale-dir`、BF16 校准入口和用户授权的 `--allow-unhealthy`，仍要求空卡/显存检查。
- [校准输入收集](tools/calibrate_modelslim_v.py)、[scale 导出](tools/export_modelslim_v_scales.py)、[完整实验驱动](tools/run_modelslim_v_experiment.py)、[最终实验审计](tools/audit_modelslim_v_experiment.py)。[原矩阵驱动](tools/run_gpqa_matrix.py)增加独立 scale 根目录及按模式复用历史参照。

新服务使用 `launch_mxfp_model.py --scale-dir` 指向下列目录，并按所需模式设置 `--mode c8` 或 `--mode c4`。旧默认 calibration 路径没有切换。

```text
/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/msmodelslim-vscale-123-20261113/scales/c8
/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/msmodelslim-vscale-123-20261113/scales/c4
```

本地已解包到 `artifacts/msmodelslim-vscale-123-20261113/calibration/scales/c8` 和 `c4`，74 个校准证据文件逐项 SHA256 核验通过。原始统计位于同级 `calibration-raw`。

## 结果与附件

- [最终整体审计](../artifacts/msmodelslim-vscale-123-20261113/final-audit.json)、[精度摘要](../artifacts/msmodelslim-vscale-123-20261113/gpqa-summary.json)、[逐轮精度与运行指标](../artifacts/msmodelslim-vscale-123-20261113/gpqa-audit.json)。
- [导出 scale 清单](../artifacts/msmodelslim-vscale-123-20261113/calibration/scales/export-c8-c4.json)、[原 scale 前后核验](../artifacts/msmodelslim-vscale-123-20261113/old-calibration-after.json)、[进程清理核验](../artifacts/msmodelslim-vscale-123-20261113/final-process-check.json)。
- [完整新实验包](../artifacts/msmodelslim-vscale-123-20261113/qwen36-modelslim-vscale-gpqa-final-123-20261113.tar.gz)：241 份证据及逐文件 SHA256 索引，含全部新六轮逐题结果、请求配置、metrics、服务/评测日志、校准输入与 scale、40 份冻结源码、ModelSlim 关键源码及旧参照摘要。历史 BF16/旧 C8/C4 完整回答仍在[上一轮报告及证据包](qwen36-prefix-mtp3-graph-gpqa.md)。
- 完整新实验包 SHA256：`6889685e223cc9bd17027095448c1fad9ad048ae1c104dddcbf1d2aa54bf39ad`；13,609,598 字节。归档成员与嵌入校准索引均已逐项校验，本地下载哈希与远端一致。
- 原始 ModelSlim 提交归档 SHA256：`a85aa09c7bb178913debd4e5251bc139a410fc98978d6690b8066e5cb1f0e3a8`，本地与 123 一致。

共享运行目录：`/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/msmodelslim-vscale-123-20261113`；正式评测子目录为 `gpqa-r1`。详细报告和附件保留在 vllm-ascend 项目中。
