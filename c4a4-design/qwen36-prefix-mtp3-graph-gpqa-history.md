---
title: Qwen3.6 prefix、MTP3、图模式与 GPQA 验证
type: tech-note
project: vllm-ascend
status: archived
tags: [kv-cache, kv-cache-quant, mxfp4, attention-backend, verification]
created: 2026-11-13
updated: 2026-11-13
summary: 迁移排障过程，最终结论见同名主报告
---

本文件保留整理前的历史过程，含已作废运行和中间状态；最终结果见 [完整验证报告](qwen36-prefix-mtp3-graph-gpqa.md)。

## 整理前状态（历史快照）

已在 141.61.133.123 的 tp_a4c4_gqa_vllm 容器完成迁移，使用健康空卡 0、2，TP2、memory fraction=0.75。共享模型、代码、QFA vendor 和静态 V scale 路径保持。下文 200 等待和早期失败记录为历史过程，不代表当前状态。

BF16 三轮完整 GPQA 已完成：165/198、165/198、169/198，平均 84.0067%，均为 TP2、MTP3、prefix 关闭、FULL_DECODE_ONLY，真实 target/draft replay 已核验。r6 C8 三轮有效结果为 162/198、167/198、162/198，平均 82.6599%；C4 服务正在加载，尚无正式分数。

C8 MTP3 的异常已定位到静态 V scale 被跨层覆盖：MTP 混合规划让 41 个 cache group 共享物理页池，初始化/回收时遍历所有层填充全池，最终每层实际读到第 39 层的 scale。两个 rank、10 个 GQA 层实取快照均匹配末层 256/256 通道；第 3 层只有 75/256（rank0）和 70/256（rank1）与自身校准值相同。80 份实际 QFA 快照中，所有 Q4 与逐 Q1 的结果逐位相同，排除了该样例的 QFA causal 多 token 计算差异。

已修复 C8/C4：随 KV 写入，仅恢复当前层 slot 所属 64-token 单元的两份 32-token 静态 scale；移除 V2 runner 的全池广播和无法区分页所有者的回收补填。12 组真实 NPU cache/graph 回归、109 项 attention CPU 回归、3 项 runner 清零回归通过。最终源码本地/123 共 36 项 hash 一致，正式续测位于 gpqa-mtp3-graph-123-20261113-r6，并沿用已审计的 r3 BF16 三轮。 C8 修复后图模式/MTP3 的 10 条语义请求全部通过：排序为 [-2,0,5,9,9]，3032-token 标记问答为 ALPHA42；target/draft 真实 replay 已确认，随后补充的 poisoned-tail 测试发现还需同时写完整 scale pair，r5 C8 结果因此作废，完整修复由 r6 重跑。

此前 C4 prefix/MTP 的 HTTP、命中与真实图回放证据保留；严格一致性曾失败，但这些运行早于共享池 scale 修复，不能作为最终结果，待修复后重测。旧 C8 未完成的 GPQA 已排除，未产生或采用分数。未降低严格判定标准。

## 任务口径

原在 200 的 tp_a4c4_gqa_vllm 容器准备；2026-11-13 用户要求打包该容器并迁移至 141.61.133.123 空卡，保留共享目录。模型为 /mnt/share/weight/Qwen3.6-35B-A3B，TP=2。先验证 C4 prefix caching 和原生 MTP，再对 BF16/C8A8/C4A4 各执行三次完整 GPQA；精度测试统一 MTP=3、prefix caching 关闭、图模式开启。

用户允许显存足够时共用 0/1 卡。本轮启动器采用 gpu-memory-utilization=0.45，并在启动前要求每卡空闲空间不小于 45% 总 HBM + 8192 MiB。当前其他任务各占约 103 GiB，剩余约 9 GiB，不足以加载模型。未停止其他进程，也没有强行加载模型。原指定容器此前已停止，现已恢复其 bash 入口。

运行根目录：`/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/gpqa-mtp3-graph-20261112`（目录名保留开始准备时的日期）。上级任务环境 env.sh、QFA vendor、checkpoint 和静态 V sidecars 继续使用上一阶段的已验证安装。

## 已完成验证

- CPU 回归：96 passed，17 个依赖弃用告警，logs/unit-r1.log。
- 小显存 NPU 图验证：6 次真实捕获、24 次 replay 全部通过，c4-graph-operator-r1.json。
- 每张图包括原生 Q/K 动态 MXFP4 量化、静态 V Triton writer、QFA metadata 和正式 mode5/schema2 QFA。
- B=1/2，Q=1/4/11，覆盖 N2TGD/TND；长度依次 512→513→519→514（第二请求最后变为 63），检验跨页和回滚时长度更新。
- 每次更新真实 Q/K/V 和 slot/length tensors；replay 不重新进入 Python。图/eager 输出逐位一致，缓存字节一致，独立 CPU golden 最大误差均为 0。
- 九份 GPQA 配置离线验证通过；首份 AIS Bench dry-run 退出 0，没有发起 GPQA 模型请求。
- prefix 输入离线验证为约 20035 tokens，超过 16384，覆盖混合 scheduler 页边界。Transformers 5.19 的 apply_chat_template 显式使用 return_dict=False，确保拿到 token ID 列表。
- 26 份修改/新增 Python 文件通过 Ruff 0.14.0 lint/format；git diff --check 通过。

这些结果不等于整网 prefix/MTP/图模式通过。该小节描述首轮准备时状态；最新完整结果见顶部和结果表。

## 本轮代码适配

- mxfp_static.py：仅加载 target GQA 的 sidecar，排除 is_draft_layer；C4 开放原生 qwen3_5_mtp 和图执行，仍拒绝 KV transfer；MTP 要求 draft kv_cache_dtype=auto/bfloat16。
- utils.py、worker/v2/attn_utils.py：按实际 spec 的 draft 归属和 speculative_config.kv_cache_dtype 判断 packed cache，避免 BF16 draft 被全局 C4/C8 dtype 错分配。
- worker/v2/aclgraph_utils.py：在 target UpdatableGraph 的真实 replay 路径增加一次性日志，和已有 draft replay 日志共同作为验收证据。
- 图模式选择 MRV2 原生 ACLGraph FULL_DECODE_ONLY，compilation mode=0，enable_npugraph_ex=false，enable_static_kernel=false。mode=0 关闭 Torch 编译变换，FULL_DECODE_ONLY 保留真实 NPU 图捕获/replay；不是 eager 替代图模式。最终以真实 replay 日志验证。
- target/draft 均开启图模式，draft 使用 BF16；MTP=3 的 target 验证通常每请求 4 个 query。三种精度使用同一 draft 精度和运行配置。

## qscale 选择

现有调用不是按整个 batch 的 token 总数选择布局，而是比较 G × max_query_len 与 80。G=本 rank Q heads/KV heads：

| 条件 | layout_q_descale |
|---|---|
| G × max_query_len <= 80 | N2TGD，形状 [Hkv,T,G,D/64,2] |
| 大于 80 | TND，形状 [T,Hq,D/64,2] |

本模型 TP2 为 Q8/KV1，因此 Q<=10 选择 N2TGD。普通 decode Q=1 和 MTP3 verify Q=4 均为 N2TGD；长 prefill 为 TND。metadata 与正式 QFA 使用相同布局。入口为 attention_c8_mxfp.py 的 _qfa_query_scale_for_layout，C4 共用此逻辑。

## GPQA 配置冻结

模型输入/输出来源：`/mnt/share/t00970481/benchmark_cp/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`。

保留 max_out_len=32768、batch_size=32、stream=false、temperature=0.6、top_k=20、top_p=0.95、原 extract_non_reasoning_content 后处理。源配置未设置 seed，三次正式运行也不新增 seed。仅替换模型路径、服务名/abbr 和本任务端口 18086。

数据沿用 benchmark_cp/bench.sh 使用的 gpqa_gen_0_shot_cot_chat_prompt.py：GPQA Diamond 198 题、原 CoT chat prompt、原答案提取和评价器。CSV、模型配置和数据集配置 SHA256 保存在 benchmark-manifest.json。九份配置采用源文件内容快照，详细参数核验见 config-validation.json。

服务统一 max-model-len=65536、max-num-seqs=32、max-num-batched-tokens=2048；为 32768 输出预算保留足够输入空间。三轮使用不同 work-dir，不启用 --reuse；每轮要求真实结果包含 198 条逐题详情。

## 队列与完成判据

工具脚本位于本项目 c4a4-design/tools：

1. launch_mxfp_model.py：每 30 秒检查 0/1 卡余量；默认等待上限 12 小时，满足显存门槛后才启动服务。服务就绪与端口均有检查。
2. check_prefix_mtp.py：prefix 冷/热、共享前缀分支、反序并发，比较生成 token IDs，检查真实 prefix 命中；MTP3 还必须观察到 draft 和 accepted token 计数。
3. run_gpqa_matrix.py：先执行 C4 图模式 prefix+MTP0 与 prefix+MTP3；成功后依次 BF16/C8/C4，每种 MTP3+prefix关闭+图模式，先 smoke，再执行三轮完整 AIS Bench。要求 target 与 draft 的真实 replay 日志。
4. 任何启动、功能检查或正式评测失败均停止队列并记录 matrix-status.json；只清理本任务拥有的服务，不停止其他任务。
5. 完整九轮后生成 gpqa-summary.json/csv，包含每轮正确数/198、准确率、三轮均值/样本标准差/min/max，以及 C4 对 BF16/C8 的百分点差。MTP0/3 greedy 比较另存，不把逐字相同率当作 GPQA 准确率。

200 原等待服务 PID 1526 和矩阵 PID 3155 已核验归属后停止，旧 matrix-status.json 标记 migrated_to_123，完成数仍为 0/9。123 使用独立运行目录 gpqa-mtp3-graph-123-20261113，避免覆盖准备阶段证据。

容器内查看：

```bash
RUN=/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/gpqa-mtp3-graph-20261112
cat "$RUN/matrix-status.json"
cat "$RUN/c4-prefix-mtp0-launch.json"
tail -n 30 "$RUN/logs/matrix.log"
```

本地准备阶段证据保存于项目 artifacts/gpqa-mtp3-graph-20261112。详细结果继续存本项目，不写入共享知识库。先前无 MTP、eager 的基线结果见 [首轮接入验证](qwen36-c4-qfa-static-v-validation.md)，不能作为本轮 MTP3+图模式 GPQA 结果。

## 2026-11-13 迁移 123

- 200 容器已提交为 c4a4-qfa-runtime:20261113，image ID 为 6ce8a39d1cef8e445c404d5127c899dc05f1f7ff7aa84415c258fe45be545238。共享归档目录为 /mnt/share/t00970481/A4C4-Quant/container-migration-20261113；压缩、SHA256 校验及 123 加载过程正在执行。
- 依用户要求使用 /mnt/share/t00970481/docker_run.sh。原脚本已备份，只将镜像和容器名改成可由 C4A4_IMAGE/C4A4_CONTAINER_NAME 覆盖，默认行为保持；加载后启动同名 tp_a4c4_gqa_vllm。
- 123 首次新鲜检查 0、2 卡 idle 且 health=OK，各约 84 GiB 总 HBM、79.8 GiB 空闲。启动前再次检查空闲/健康/余量；TP2 使用 0,2，memory fraction=0.75，预留 8192 MiB。历史快照不替代启动前检查。
- 运行目录为 /mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/gpqa-mtp3-graph-123-20261113。使用独立 Triton/Torch 扩展缓存，源码、正式 QFA、checkpoint、V scale 继续使用原共享路径。九份新评测配置已生成且离线核对通过。
- 尚未产生本轮 GPQA 分数；待实际 123 设备兼容验证、整网 prefix/MTP/图验证通过后开始九轮评测。

### 123 环境已启动、预检已通过

镜像压缩与双端 SHA256 校验完成，123 成功加载相同 image ID，通过指定脚本启动容器 eadf02235e3d。完整 SoC 为 Ascend950DT_9574 / 3510（200 为 Ascend950PR_958b），已通过 asys 在宿主及容器内核实。

容器内 Torch 2.10.0+cpu / torch_npu 2.10.0.post7、OpenSSL 3.5.6 正常。123 CPU 回归 96 passed；独立缓存下重新编译桥接和 Triton，C4 图 6 captures / 24 replays 全部通过，graph/eager 位相同、cache 字节相同、CPU golden max_abs=0。此验证证明本次正式算子链可在 123 执行，不代替整网验收。

123 初始 C4 prefix+MTP0 TP2 服务已启动（容器 PID 1113，矩阵 PID 1114，仅为启动时快照），设备 0,2，memory fraction=0.75，FULL_DECODE_ONLY。当前阶段 waiting_for_service，后续由矩阵执行 prefix+MTP3 与九轮 GPQA。GPQA 仍为 0/9，无新分数。

123 证据独立保存于项目 artifacts/gpqa-mtp3-graph-123-20261113；29 份本地/共享源码已在启动前核验 SHA256 一致。200 旧队列保持停止，不会重复占卡测试。

### 123 整网 prefix 结果与 MTP 排障

当前诊断目录为 gpqa-mtp3-graph-123-20261113-r2。首轮服务已完成 7 档整网图捕获，但 /reset_prefix_cache 返回 404：当前上游仅在 VLLM_SERVER_DEV_MODE=1 时注册该接口。启动器只对 prefix 验证启用此既有开关，正式 GPQA 为 0；首轮日志保留。

C4 MTP0 第二轮取得真实 target UpdatableGraph replay 日志，五次长前缀请求全部 HTTP 200，prefix 命中增量 65536 tokens。cold/warm 串行 token IDs 相同；反序并发的 water/solar 两请求分别在 token 4/31 出现差异，严格一致性检查失败，不能声称完整 prefix 验收通过。

同配置 BF16 MTP0 对照真实命中 78336 tokens，五次请求 token 一致，严格检查通过。因此暂不能把 C4 差异归因于所有精度共同的普通 batch 波动。

补充正式 QFA batch 探针覆盖 KV=(512,513)/(20037,20035)/(32768,32769)，Q=1/4，单请求与双请求使用相同 packed Q/K/V。全部设备结果与各自实际 metadata 的 CPU kernel reference 逐位一致。跨 batch 差异最大为 1.52587890625e-5，参考也复现相同差异；分段随 batch 改变。这说明算子数学路径本身可能存在细小 batch 差异，但尚未证明它是整网差异的全部根因。保留 c4-batch-operator.json 和全部分段信息。

MTP3 启动已发现并修正两处接入问题：

- 上游把 qwen3_5_mtp 方法名标准化为 mtp；C4 校验改为接受标准名/旧名，同时要求 draft hf_config.model_type=qwen3_5_mtp、draft cache=auto/bfloat16，不扩大到其他架构。
- dense hybrid 页对齐错误地把 state_content_bytes 显式定义的 C4/C8 packet 当作普通 K/V。现在 packet 保持逻辑 token block，后续统一物理 page stride padding；新增 C4/C8 与 BF16 draft 不可整除场景和共享 backing 页写入隔离回归。当前相关 CPU 回归 107 passed，原有混合分页回归继续执行。

prefix 严格一致性结论与 GPQA 结果分别记录。调度器新增 --gpqa-only --prefix-report，可携带已有 prefix 验收报告执行明确关闭 prefix 的正式精度测试；不会把已失败的 strict check 改成通过。当前九轮 GPQA 仍为 0/9。

MTP3 后续修复：实际预热的 BF16 K/V 写入遇到 uint8 cache。源码确认上游 FullAttentionSpec.merge 新建 spec 时不保留 is_draft（该字段 compare=False）；draft 模型创建阶段已正确应用 kv_cache_dtype=auto，无需修改模型构建。问题发生在合并后重新按全局 MXFP 配置选择缓存视图。is_mxfp_kv_quant 现额外检查 spec 的实际物理 dtype：C4 为 uint8，C8 为 float8_e4m3fn；BF16 spec 即使丢失 draft 标记也不会进入 packet 路径。

新增测试直接执行 FullAttentionSpec.merge 和 _reshape_kv_cache_v2，验证合并后的 draft 为 BF16 两元缓存，target 保持各自 packet 判别。四组核心回归 109 passed，另外既有 hybrid alignment 2 passed。失败服务的 API/EngineCore/worker 均按已记录 PID、命令、工作目录环境和设备标识核实后清理；没有停止其他用户进程。当前 c4-prefix-mtp3-r4 重新加载中，MTP 整网尚未验收，GPQA 仍 0/9。

<!-- GPQA_RESULTS_BEGIN -->
## GPQA 阶段结果（BF16、C8A8 已完成）

| 模式 | 第1轮 | 第2轮 | 第3轮 | 均值 | 样本标准差 |
|---|---:|---:|---:|---:|---:|
| BF16 | 83.3333%（165/198） | 83.3333%（165/198） | 85.3535%（169/198） | 84.0067% | 1.1664 pp |
| C8A8 | 81.8182%（162/198） | 84.3434%（167/198） | 81.8182%（162/198） | 82.6599% | 1.4580 pp |
| C4A4 | 待完成 | 待完成 | 待完成 | — | — |

BF16 三轮均为 198 次成功请求、0 请求失败、0 prefix query/hit、0 preemption，真实 MTP 接受计数均大于零；三轮触及 32768 输出上限且未提取到有效选项的题数依次为 3、2、1，沿用源配置计入原评分，不删题、不补跑更长答案。题目、选项、标签顺序在各轮间完全一致。

完整逐题证据已归档至本项目 artifacts/gpqa-mtp3-graph-123-20261113-r3/bf16-results.tar.gz 并核验 SHA256；prefix/MTP 诊断为同目录 prefix-diagnostics.tar.gz。最终将使用 audit_gpqa_results.py 对全部九轮再次核验。
<!-- GPQA_RESULTS_END -->
### C8 MTP3 语义异常定位（2026-11-13）

C8 的 10 请求原 smoke 仅检查 HTTP 成功及非空输出；人工核查发现排序输出为 [-2,0,2,5,9]、3032-token ALPHA42 提取返回不完整文本，正式长请求 draft 接受率异常接近 99%。已按记录的父子 PID、命令与 runtime 环境核实并停止本任务矩阵和 C8 服务，未完成的首轮不计入分数。BF16 三轮完整证据独立保留。matrix-status.json 标记 diagnostic_c8_generation_anomaly。

smoke_test.py 新增 --check-answers，正式调度器强制检查算术、排序、长上下文标记；实际 BF16 记录通过，原 C8 错误记录能被拒绝。诊断目录为 /mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/c8-mtp-diagnostic-123-20261113。

- C8 图模式 + MTP0：10 请求全部通过语义检查，含排序和 ALPHA42。
- C8 eager + MTP3：复现相同排序错误和长上下文提取错误，说明问题不依赖图回放。eager 仅用于诊断，不替代用户要求的正式图模式。
- C8 独立 QFA：KV=513/3032，普通 pitch=278528 与 MTP padded pitch=524288 四例，K/V/Kscale cache bytes 全部正确；Q4 与四次对应 Q1（causal/no-mask）逐位一致。对反量化后 FP32 attention 的 max_abs 为 0.007826/0.004602，属于不同 P 计算路径的数值差异记录，未据此更改算子门限。

当前继续检查整网 MTP 的多组 KV、GDN 状态和实际输入；尚未确定根因，也未修改 QFA 算子数学实现。先前 C4 MTP3 真实回放/accepted 计数仅证明可执行，不等于本轮精度或语义验收通过。
## 静态 V scale 共享池覆盖修复（2026-11-13）

根因证据位于运行目录 c8-mtp-diagnostic-123-20261113/static-v-scale-root-cause.json；actual-qfa/rank0、rank1 包含每层前四步 QFA 输入/输出和物理 cache 快照。C8 graph/MTP0 语义通过，C8 eager/MTP3 与 graph/MTP3 同样失败；逐层采样发现错误的 V descale，而 Q4 与 Q1 算子结果一致。

`vllm_ascend/ops/triton/mxfp_v_scale.py` 直接写当前 slot 所属 64-token 单元的两份 E8M0 通道 scale，支持 C8 PA_NZ 和 C4 PA_BBND；尊重 storage offset、block stride，跳过负 slot，连续同一 64-token 单元只写一次。每个 chunk 的首 token 即使位于 group 中部也写入。该 kernel 不读取设备数据到 CPU，真实图重放使用更新后的 slot 和 scale buffer。KV 写入通知在 V scale 更新完成后发出。

`worker/v2/model_runner.py` 不再全池初始化静态 scale，`C8KVBlockZeroer` 已删除。共享零页和新分配页仍由原 tuple-aware zeroer 处理，实际 layer 在写入 KV 时写入自己的 scale。BF16 路径不使用此 kernel，其已完成的三轮无需重跑。

12 组 NPU 测试覆盖 C4/C8，(H,D)=(1,128)/(1,256)/(2,256)，padding=0/4096，非零 storage offset，负 slot 后的 slot0，两层不同 scale、不同页、页回收和捕获后地址/scale 更新。比较整个 backing allocation（含 K/V/Kscale、其他 owner 和 padding），均逐字节相同。测试位于 tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mxfp_v_scale_scatter.py。

r4 只启动过服务准备，正式请求前已停止，用于补全空 slot 与写入通知的边界处理；r5 随后因 scale pair 尾部缺陷被停止，其 C8 第一轮 162/198（81.8182%）已标记无效，不计入正式结果。r6 为补齐 scale pair 后的正式评测目录。旧 prefix 结果在 r5 中明确命名为 prefix-validation-report-before-scale-fix.json，最终验收需使用修复后重测报告。

### 尾部 scale pair 的 NaN 隔离

补充试验将未使用的 V scale 字节设为 255，仅写有效 32-token 分组时，C8 L=17 和 L=513 的正式 QFA 输出为 NaN，而 L=33 有限。原因是 QFA 消费最终 64-token 单元的两份 scale，不能假定无效半组的反量化不会执行。此问题与跨层全池覆盖独立。

最终 writer 在当前层拥有的 64-token 单元内写两份相同的静态通道 scale。26 组 C4/C8 正式 QFA 测试覆盖 L=1/17/33/65/513、Q=1/4/11、N2TGD/TND，并在图捕获后再次将 V scale 全部毒化为 255：eager/graph 与全初始化 cache 的结果均逐位相同且有限。另 12 组整块 allocation 字节回归重新通过。测试脚本为 c4a4-design/tools/check_mxfp_v_scale_tail.py。

所有 r5 C8 分数和部分结果均保留为故障证据，不被续跑脚本复用。BF16 未使用该量化路径，仍沿用 r3 的三轮完整结果。修复后 prefix 重测输出目录为 prefix-after-scale-pair-fix-123-20261113-r2。


2026-11-13 r6 首轮审计：C8 run1=162/198，198 HTTP 成功、0 preemption、0 prefix query/hit，MTP 接受率 73.4509%；3 条 length-limited/无效提取答案仍计入 198，不延长输出、不筛题或复用旧响应。与 BF16 三轮的题目、选项和参考答案完全相同。

C8 r6 三轮完整审计：每轮 198 请求成功、0 失败、0 prefix query/hit、0 preemption；MTP token 接受率为 73.4509%、72.7000%、73.3678%。三轮分别有 3、1、3 条输出达到 32768 上限且未提取到有效选项，全部保留计分。三轮平均比 BF16 低 1.3468 pp；这是当前固定校准与原采样配置下的三轮观测，不代表统计显著性结论。
