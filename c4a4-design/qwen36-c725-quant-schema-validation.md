# Qwen3.6 C7.25 / quant_schema 验证（第 2、3、4 项）

状态：三项均已完成。九轮为完整 GPQA Diamond198，共 1782 个成功请求、0 请求失败。

## 第 2、4 项：模型精度

| K/V 方案 | quant_schema | 三轮正确题数（各 198 题） | 平均准确率 | 样本标准差（百分点） |
|---|---:|---|---:|---:|
| C7.25 | 0 | 156, 158, 157 | 79.2929% | 0.5051 |
| C7.25 | 1 | 163, 159, 159 | 80.9764% | 1.1664 |
| C7.25 | 2 | 162, 158, 161 | 80.9764% | 1.0513 |
| C7（历史） | 2 | 163, 164, 165 | 82.8283% | 0.5051 |
| C8（历史） | — | 160, 160, 161 | 80.9764% | 0.2916 |
| BF16（历史） | — | 165, 165, 169 | 84.0067% | 1.1664 |

第 2 项使用 C7.25/schema2 的三轮结果，相比历史 C7/schema2 均值变化为 -1.8519 个百分点。第 4 项本次最高均值为 80.9764%，schema1、schema2 并列。这是三次随机采样的观测结果，不据此宣称统计显著提升；历史 C7/C8/BF16 未在本次重跑。

三组均使用同一 198 题及同一题序、选项、答案参考；保留全部轮次，未按分数选择重跑。每组先执行语义检查，再连续完成三轮。

| schema | 轮次 | 正确/成功 | 输出到长度上限 | 无效提取答案 | 空回答 | preemption | MTP token 接受率 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 156/198 | 6 | 4 | 0 | 0 | 73.476% |
| 0 | 2 | 158/198 | 3 | 3 | 0 | 0 | 73.081% |
| 0 | 3 | 157/198 | 1 | 1 | 0 | 0 | 72.941% |
| 1 | 1 | 163/198 | 3 | 3 | 0 | 0 | 73.135% |
| 1 | 2 | 159/198 | 3 | 3 | 0 | 0 | 73.152% |
| 1 | 3 | 159/198 | 4 | 4 | 0 | 0 | 74.033% |
| 2 | 1 | 162/198 | 2 | 2 | 0 | 0 | 73.068% |
| 2 | 2 | 158/198 | 5 | 5 | 0 | 0 | 73.183% |
| 2 | 3 | 161/198 | 2 | 2 | 0 | 0 | 73.538% |

## 第 3 项：同一真实输入的算子比较

采集第一层 GQA：language_model.model.layers.3.self_attn.attn。TP rank0/1 各保留 prefill、Q4 MTP decode 一份，共四份输入。prefill Q=[118,8,256]、KV=118、Q scale=TND；decode Q=[4,8,256]、KV=122、Q scale=N2TGD。显式 eager 诊断服务负责采集；正式模型评测启用图模式。

reference：把保存的 FP4/E8M0 Q/K/V 精确反量化到 BF16，再调用 NPU npu_fused_infer_attention_score_v2。它衡量量化输入之后 QFA 计算（含 P 量化）相对 BF16 attention 的误差，不包含原始 Q/K/V 到 FP4 的输入量化误差。QFA kernel 页实际为 512 token，混合池逻辑 block 为 4096 token；BF16 reference 无损拆分为 128-token 页，仅重排页表。未使用的 KV padding 不参与 attention，CPU 反量化时将其置零，所有有效数据保持原值。

相对 BF16 NPU reference 的 L2 相对误差：

| Rank | 阶段 | schema0 | schema1 | schema2 |
|---:|---|---:|---:|---:|
| 0 | prefill | 7.0010% | 7.3204% | 7.3841% |
| 0 | q4 | 9.3603% | 9.5059% | 7.2858% |
| 1 | prefill | 6.8229% | 7.2181% | 7.3474% |
| 1 | q4 | 6.8795% | 6.9736% | 7.7926% |

12 组 cosine 范围为 0.9960366–0.9979614。所有输出有限，36 次实际 NPU 图回放与 eager 输出逐位一致；schema2 还逐位复现采集时的模型输出。BF16 NPU reference 与抽样 CPU FP32 attention 的相对 L2 为约 0.14%–0.18%，抽样元素均满足 atol=0.01、rtol=0.01。CPU FP32 只用于检查 reference，不取代主 BF16 NPU reference。

两个 prefill 样本中 schema0 的相对 L2 最小；Q4 的 rank0 为 schema2 最小、rank1 为 schema0 最小。没有一个 schema 在全部四份输入上最优。该短上下文、单层样本不代表所有层、长上下文或所有 batch；模型排序以上述完整 GPQA 为准。max/mean absolute error、RMSE、cosine 及完整输出均保存在 operator-results。

## 配置与 scale

- 模型 /mnt/share/weight/Qwen3.6-35B-A3B；123 服务器，tp_a4c4_gqa_vllm 容器；启动时确认空闲的 2/3 卡，允许 Warning。
- TP2、MTP3、FULL_DECODE_ONLY、prefix OFF。九轮均核验 target/draft 的真实图回放，prefix queries/hits 为 0。
- 沿用 AIS Bench vllm_api_general_chat.py：batch32、temperature0.6、top_p0.95、top_k20、max_out_len32768。没有额外指定采样 seed。
- C4 K：npu_dynamic_mx_quant(key[:count], dst_type=float4_e2m1fn_x2, block_size=32, scale_alg=2, dst_type_max=7.25)。
- V 指数：floor(log2(amax*8/7.25+9.6e-7))-2；直接加载独立 E8M0 sidecar，推理不重新套用旧公式。
- Q 保持 alg0；没有 Hadamard，本次不执行第 5 项。

V 使用原 BF16 校准的 58 条 qwen3_cot_vlm_text、127259 个有效 token 的逐通道 amax。此次重新导出 scale，没有重新采集激活；相同分母 7 的复算精确复现原 C7 全部 5120 通道。20 份新 sidecar 中 295/5120 通道的 scale 缩小一半，其他不变。保留旧 C7、C8、legacy 共 60 份 scale，前后哈希一致。K 的 ceil 规则与带 epsilon 的 V floor 规则在精确边界仍有区别：这些 V amax 中 45 个通道与 ceil(log2(amax/7.25)) 不同；实际实现严格遵循验证文档分别指定的 K/V 公式。

## 代码与审计

在 attention_c4_mxfp.py 与 mxfp_static.py 增加经校验的 mxfp_static.k_quant_max，默认保留 7.0，实验显式指定 7.25。quant_schema 已有后端支持，launch_mxfp_model.py/run_gpqa_matrix.py 新增传参入口。新增独立 scale 导出、真实输入采集、BF16 reference 重放、九轮调度和审计工具；不改 QFA kernel 或历史 scale。

静态 scale 单测 40 项通过；C7.25 原生 K 边界及 3 次 graph replay 通过。最终核验原运行 44 份源码、恢复运行 45 份源码、12 份算子包/环境文件、60 份旧 scale、20 份新 scale。所有本任务进程均在 123 按 boot ID 和 PID start ticks 核验退出，没有重置设备或停止其他任务。没有提交或推送代码。

## 完整证据

恢复运行目录：/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/c4-c725-schema-123-20261113-r2

原始证据目录：/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/c4-c725-schema-123-20261113

主机在 schema0 首次启动期间重启，该次未开始 GPQA；恢复后在独立目录完整执行 schema0/1 六轮，schema2 三轮保留。恢复前后源码、算子及 scale 均核验一致。原中断记录与恢复记录均随归档保存。

[完整证据包](../artifacts/c4-c725-schema-123-20261113/c4-c725-schema-123-20261113-complete.tar.gz)（包含全部九轮完整回答、metrics、日志、四份真实输入及所有输出、源码与哈希、历史参照回答副本；包哈希见同目录 archive-info.json）。

[最终审计](../artifacts/c4-c725-schema-123-20261113/final-audit.json)；[算子证据包](../artifacts/c4-c725-schema-123-20261113/operator-evidence.tar.gz)。
算子证据包 SHA256：ffcba4aed31b2ad4649531e6d8608e70862c38e310ad848ec0fa284430aaf748。
