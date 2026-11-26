# Qwen3.6 C4：K 使用 scale_alg=2、dst_type_max=7

日期：2026-11-13。状态：三轮完整 GPQA、最终审计、任务进程退出核验全部完成。

C4 K 改为 alg2/max7 后，三轮为 **163/164/165（每轮198题）**，平均 **82.8283%**；与相同 V scale、旧 K alg0 的三轮均值完全相同，差异 **0.0000 个百分点**。本轮没有观察到均值精度提升，也没有观察到均值下降。三轮样本不足以据此证明两方案统计等价或稳定性改善。

| 方案 | 三轮正确题数 / 198 | 三轮均值 | 样本标准差（百分点） |
| --- | --- | ---: | ---: |
| C4 K alg2/max7（本轮） | 163 / 164 / 165 | 82.8283% | 0.5051 |
| C4 K alg0、相同 V scale（历史对照） | 166 / 162 / 164 | 82.8283% | 1.0101 |
| C8、ModelSlim V scale（历史参照） | 160 / 160 / 161 | 80.9764% | 0.2916 |
| BF16（历史参照） | 165 / 165 / 169 | 84.0067% | 1.1664 |

C8/BF16 没有在本轮重跑。C4 本轮比历史 C8 高1.8519 pp，比历史 BF16 低1.1785 pp。比较包含随机采样、批处理和运行时段的差异，不作单一因素因果或显著性推断。

## 接入改动

仅修改 [attention_c4_mxfp.py](../vllm_ascend/attention/attention_c4_mxfp.py) 的 K 调用：

~~~python
k, ks = torch_npu.npu_dynamic_mx_quant(
    key[:count],
    dst_type=torch_npu.float4_e2m1fn_x2,
    block_size=32,
    scale_alg=2,
    dst_type_max=7.0,
)
~~~

保留 key[:count] 的有效 token 切片。7.0 与用户指定 7 的浮点参数语义相同。Q 仍为默认 alg0；V 沿用此前保存的 ModelSlim 静态 C4 E8M0 sidecar，不重新校准或导出。K 为每 token/head 沿 D 的 block32 动态统计，V 为校准数据的 per-channel 静态统计；普通范围尺度规则接近，二次幂边界仍不能宣称逐 bit 完全一致。

本地/共享目录后端 SHA256：1a49e9d6d1ecc3002773608679bc7f8abadedef765bfd115666bb3957d3c6ec4。与实验前清单比较，40 份相关源码中仅该文件改变；容器 editable 安装指向共享 vllm-ascend。没有修改 Q、C8 后端、V 量化/校准、权重、msmodelslim 或正式 QFA kernel；未提交或推送。

## 固定评测协议

- 141.61.133.123，容器 tp_a4c4_gqa_vllm，模型 /mnt/share/weight/Qwen3.6-35B-A3B，权重 BF16。
- 三轮有效 C4 均使用物理2/3卡。3 为用户允许的 Warning 卡，启动前确认无其他进程且显存充足。GPU memory utilization=0.75。
- TP2、MTP3、FULL_DECODE_ONLY、prefix caching OFF；target 和 draft 的真实 graph replay 均已核验。
- GPQA Diamond198；沿用指定 vllm_api_general_chat.py 的输入输出参数：max_out_len=32768、batch_size=32、temperature=0.6、top_k=20、top_p=0.95。
- 模型 max_model_len=65536、max_num_seqs=32、max_num_batched_tokens=2048。
- V scale：/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/msmodelslim-vscale-123-20261113/scales/c4。旧校准和新 C4/C8 共60份 sidecar 的实验前、维护时和最终 SHA256 一致。
- torch2.10.0、torch_npu2.10.0.post7、triton3.6.0、transformers5.19.0、vLLM0.33.1.dev0、CANN9.2；维护前后容器版本及OMP_NUM_THREADS=8一致。
- 此次量化参数变更没有重新执行 prefix 开启时的严格一致性验证；本轮精度测试固定关闭 prefix。

## 有效三轮审计

| 轮次 | 正确 / 198 | 精度 | 长度截断 | 未抽取答案 | MTP token 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 163 | 82.3232% | 0 | 0 | 72.9621% |
| 2 | 164 | 82.8283% | 2 | 2 | 73.2764% |
| 3 | 165 | 83.3333% | 0 | 0 | 72.8503% |

三轮有效评测 **594/594 成功请求、0失败、0空回答、0抢占**；prefix 查询/命中均为0，MTP drafted/accepted均大于0。截断及未抽取答案保留在198题分母和原评分中。所有题干、选项顺序、参考答案一致，正确题数由逐题结果重算核对；三组回答内容哈希不同。

结果文件 SHA256：

- 第1轮：f72444ceb63f45b2ed92bda5e570c09ef713887557e110fa9e33775cfa866796
- 第2轮：65781b9b75d788044cdfee32803f0d6dcba70e9b39e529a387147a0d809f645d
- 第3轮：d1ba881d2601e337a0b44f424b0910dc40436158d90ab2afcb1aac40de5453e2

AST检查、Ruff lint/format通过。真实Ascend950上三组BF16 [2,1,256] 边界输入的48个block scale与独立位运算参考一致；9次修改输入后的NPUGraph replay，FP4 payload与E8M0 scale字节均与eager一致。前后两次服务启动各10项语义冒烟通过。

## 启动失败与维护处理

0/4 卡组的r1/r2两次启动在HcclAllreduce / RunAicpuIndOpThreadInit报507018，device0的threadArray nullptr，均未进入GPQA。独立4通信组AllReduce在同一卡组通过，尚未证明根因。随后在2/3空卡启动成功，完成前两轮。

第三轮原尝试在收到149题时遇到用户确认的主机维护；该次不完整请求/回答不参与三轮统计。主机重启后已在123确认旧容器Running=false、Pid=0；SSH先恢复、共享NFS约17:01恢复后，启动原容器并在r4独立目录完整补跑第三轮198题。保留前两轮原结果及哈希，没有拼接部分题目或依据分数选择重跑。

r3原matrix-status.json未修改；r4/resume-reference保存原状态副本及带恢复依据的续跑引用。有效第1/2轮evidence_run指向r3/gpqa-r1，有效第3轮指向r4/gpqa-r1。卡组保持2/3，但第三轮经历服务重启，未额外改变原生成配置。200仅用于维护时读取和备份共享证据，没有在200执行本轮模型测试。

未重置设备或终止其他任务。最终在123按本次boot ID、PID和start_ticks逐项确认本任务driver、API、EngineCore、两个worker及记录的子进程均退出；设备查询正常。日志中的AIS Bench NFS临时文件清理警告及服务退出时shared_memory警告已保留；有效轮次推理/评分成功，完整结果及退出状态均经独立核验。

## 运行目录与附件

主要运行目录：

- 前两轮及维护中断现场：/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/c4-k-alg2-max7-123-20261113-r3
- 第三轮补跑与最终汇总：/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/c4-k-alg2-max7-123-20261113-r4

本地附件：

- [最终实验审计](../artifacts/c4-k-alg2-max7-123-20261113/final-audit.json)、[逐题及运行指标审计](../artifacts/c4-k-alg2-max7-123-20261113/gpqa-audit.json)、[精度摘要](../artifacts/c4-k-alg2-max7-123-20261113/gpqa-summary.json)。
- [进程退出核验](../artifacts/c4-k-alg2-max7-123-20261113/final-process-check.json)、[归档信息及SHA256](../artifacts/c4-k-alg2-max7-123-20261113/archive-info.json)。
- [最终证据包](../artifacts/c4-k-alg2-max7-123-20261113/qwen36-c4-k-alg2-max7-gpqa-123-20261113.tar.gz)：包含三轮完整结果及metrics/日志、r3中断证据、启动失败日志、40份冻结源码、原始改动diff、未变的C4 V scale、校验清单与复现脚本。r3维护快照独立保留。
- [旧K/同V的完整对照报告](qwen36-msmodelslim-static-v-gpqa.md)。

详细记录和附件仅保存在vllm-ascend项目。恢复检查在最终归档核验后删除。
