# Qwen3.6 W8A8C8：官方启动配置与最新版 AISBench 评测

更新：2026-11-14。状态：评测环境、数据和 12 份配置已准备；模型首次加载失败，正式精度轮次为 0/12，多模态图片完整性已通过。未修改模型权重或框架源码。

## 用户要求与测试范围

在 141.61.41.200 的 tp_a4c4_gqa_vllm 容器使用共享目录；模型为 /mnt/share/weight/qwen3_6_35b_a3b_w8a8_mxfp8_c8_fix_mtp。遵照官方 950DT 示例的启动设置，当前实际硬件为 950PR。

| 数据集 | 单轮题数 | 轮数 | 计划卡/端口 |
|---|---:|---:|---|
| GPQA Diamond | 198 | 3 | 0 / 8000 |
| AIME2026（I+II） | 30 | 3 | 1 / 8001 |
| MMMU-Pro standard 10 options，CoT | 1730 | 3 | 2 / 8002 |
| MMMU-Pro vision-only，CoT | 1730 | 3 | 3 / 8003 |

MMMU-Pro 两个子集均测是用户在本轮澄清后的明确要求。总计计划 11064 个正式请求，不拼接中断轮次、不按分数选择性重跑。

服务参数：TP1、DP1、MTP3、max-model-len=135168、max-num-seqs=128、max-num-batched-tokens=4096、gpu-memory-utilization=0.90、FULL_DECODE_ONLY、prefix caching ON。ascend_compilation_config 为 enable_npugraph_ex=true、enable_static_kernel=true、enable_super_kernel=false；保留视觉模块。
完整首次命令和环境保存在实验 services/gpqa/launch.sh、launch.json，使用 vllm serve。未加入 --kv-cache-dtype mxfp8，也未加入以前 C4 实验的旋转/静态 sidecar/TP2/prefix OFF 参数。

AISBench：temperature=1.0、top_p=0.95、top_k=20、min_p=0.0、presence_penalty=1.5、repetition_penalty=1.0；max_out_len=65536，batch_size=64。12 份配置均调用新版本 VLLMCustomAPIChat.get_request_body 核验，请求 max_tokens=65536，六个采样值全部一致。该检查不发送精度请求。

## AISBench 与数据来源

官方仓库 https://github.com/AISBench/benchmark 的 HEAD 已核对并固定为 1aead1f12bd35746672bda5a8a2330f58e7ba651（2026-11-13 16:41:58 +08:00）。源码 tar SHA256：7f83d5672189660fb146ea8bcbc1253b036537e7da82d25280d3bf2599bbc25b。

采用独立 benchmark-source 与 aisbench-venv（system-site-packages）安装，不覆盖旧 benchmark_cp。源码包版本字段为 3.1.0；判断本轮版本应以完整 commit 为准。启动评测时必须使用 bench_python.sh：它将新源码、独立 venv 的 site-packages 放在既有 PYTHONPATH 前面。已实际核实导入路径来自新源码，Pillow 为要求的 11.2.1；仅安装 editable 而不调整导入顺序会错误加载系统旧 AISBench，这是准备阶段已纠正的问题。

服务器直连下载超时；通过 Windows 本机下载、核对 SHA 后，经 SSH 传入相同共享目录。旧模型 Python 环境未升级。

- GPQA：沿用已有 gpqa_diamond.csv，拷贝前后哈希一致；官方新版 GPQADataset 载入 198 条，保持题序与选项轮换规则。
- AIME2026：ModelScope evalscope/aime26，固定 revision 12ae93b463353acd4d272dbbc7dbecbcfb902642，30 条；文件 SHA256：52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee。
- MMMU-Pro standard：https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv，140919931 字节，SHA256：360e0f939ab3f7bf28bf5914e1dc5be3afe80fc74857fc289ebe1bf7c9bc907f。
- MMMU-Pro vision：https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv，507321469 字节，SHA256：80dbb7fb75442d4c1d113ea4725127371a9553dc787466c4b3b39149f83dfb91。

MMMU-Pro 两个子集经正式 loader 均载入 1730 条，未丢题；standard 解码得到 2015 张图片、vision-only 得到 1730 张，均通过 Pillow.verify 并保存逐图 SHA256。正式题序、选项和答案已保存为 loaded.jsonl，见 dataset-integrity.json。

## 首次启动结果与兼容性阻塞

0 卡无其他进程、显存足够后启动；API PID93089，boot_id 8c32b0ef-381f-4553-8670-63b057011352。启动日志确认官方编译开关、MTP3、prefix ON 与视觉模块实际启用，但加载首个权重分片失败：

    ValueError: There is no module or parameter named 'layers.3.self_attn.mla_attn' in Qwen3_5Model.

故障发生于权重加载，不属于 GPQA/AIME/MMMU-Pro 精度失败，也不是 OOM 或已运行后生成错误。已按 session/boot 复查本任务服务无存活进程，未停止其他任务、未重置设备。

检查权重实际内容：
- model_quant_type=W8A8_MXFP8：权重与激活使用 MXFP8。
- 10 个 GQA 层 quant_type=Q_FP8_DYNAMIC_KV_FP8，fa_quant_type=FAKQuant。
- K/V 为普通 FP8 E4M3 的按 head 静态量化；fa_k.scale 与 fa_v.scale 为 FP32 [2,1]，不是 MXFP8 E8M0 分组 scale。
- 第 3 层 K scale=[0.01263427734375,0.0091552734375]；V scale=[0.0068359375,0.00689697265625]；offset 均为 0。
- 当前共享分支 FAKQuant 的权重 mapper 指向 MLA 层；现有普通 GQA C8 实现为 INT8，另有 MXFP8 GQA 后端，均不能直接替代这份普通 FP8 配方。
- 因此不能通过追加 --kv-cache-dtype mxfp8 来消除报错，那会改变 K/V 量化语义。

已向用户说明两条路线：保留指定权重并补接普通 FP8 GQA 后端（推荐），或使用与现有 MXFP8 KV 后端匹配的权重。路线确定前不改权重量化描述、不伪装成 C8 成绩。

## 证据与继续入口

共享实验目录：
/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/w8a8c8-official-aisbench-200-20261114

本地文件位于项目 artifacts/w8a8c8-official-aisbench-200-20261114。
关键文件：execution-checkpoint.json、benchmark-manifest.json、request-parameter-preflight.json、dataset-integrity.json、startup-diagnosis/、services/gpqa/api.log，以及 12 份 configs/*.py。

后续须先确定普通 FP8 GQA 接入或权重路线，验证真实缓存精度、prefix/MTP/图回放和多模态输出，再启动四组各三轮完整评测。再次拉起需新建独立 attempt 目录并复查卡/端口归属，保留本次启动失败证据。不得将准备工作写成精度测试完成。



准备阶段证据已归档并完成本地逐成员验证：55 个成员（54 个文件逐一比对清单），压缩包 SHA256 为 4989e525852103c688235aae32f3a4734bfed62b268b4d8175a4c64c8b21bd64。此包仅含环境准备、数据清单/载入结果、配置与失败诊断，不含精度成绩。
