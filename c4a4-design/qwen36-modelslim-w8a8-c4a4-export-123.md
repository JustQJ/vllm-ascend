# Qwen3.6 W8A8 + C4A4 ModelSlim 联合导出

本次目标是从原 BF16 模型重新逐层导出 W8A8 MXFP8 线性层、Q/K C7.25 动态 MXFP4、V C7.25 静态 per-(head,channel) MXFP4，附带 QKV Hadamard 矩阵和部署元数据。导出及远端/本地归档验收已完成；vLLM 接入的四组服务均通过加载、图回放和烟测，正在同机三类数据集正式评测。当前 GPQA、AIME 和 MMMU standard 各三轮已审计通过，累计九轮且服务已退出；vision-only 第1轮仍在运行，完整十二轮精度对比尚未完成，最新进展见文末。

## 固定配置

- 输入：/mnt/share/weight/Qwen3.6-35B-A3B。
- ModelSlim 基线：cb83735f94604d308fa67784b1c7d5b8981addf0；独立源快照，原 C8 源及产物未改。
- W/A：MXFP8 per-block，范围保持原官方配方；o_proj、视觉与 MTP 保留浮点规则。
- Q 与 K：MXFP4，block32、scale_alg=2、dst_type_max=7.25。Q 不再沿用旧 alg0。
- V：旋转后重新收集 amax，e=floor(log2(amax*8/7.25+9.6e-7))-2；uint8(e+127) 编码，RNE E2M1。
- QKV：seed0、D256、R=diag(signs)H/16；全部 head/层使用相同矩阵，SHA256 ca70b9def213b38eb4066626b2808ec33d5694c38f7f92085703936e8925d22d。
- Q/K 在 RoPE 后旋转；V 在 observer/cache 前旋转；attention 输出在元素级 gate 前乘 R.T。
- 部署配置声明 quant_schema=2。校准 attention 使用 QKV QDQ 后的 BF16 eager 计算，**未模拟 schema2 的内部 P 量化**，metadata 明确 p_quantization_simulated=false。本次不把该声明当作真实 QFA 图执行证据。
- 复用的是 58 条校准文本数据；V amax 重新采集，并逐层记录样本/token 数。未复用旧 BF16/C8 的 V scale。

## 实现

本地 ModelSlim 中新增 fa3_c4_quant、C4 dynamic/static/rotation IR 和独立 YAML；AscendV1 saver 导出 QK_MXFP4_DYNAMIC_V_MXFP4_PER_CHANNEL、fa_v.scale/offset、c4_rotation.matrix 及 self_attn.c4_config。optional/c4_calibration 每层保存 amax、指数、样本和 token 数。

原 fa3_quant/C8 分支不改变。Q/K 校准调用原生 NPU npu_dynamic_mx_quant 再反量化，以匹配 alg2/max7.25；V 软件 RNE 已与部署用 Triton cache writer 逐元素对齐。

本地 10 个改动文件已和 123 冻结源逐 SHA256 核验；共冻结 4075 个源/配置文件。source-freeze.json SHA256：
04989b7bcb5e416653e679a5a21c14ebee4eaea90cab084e26654c48b1ff1aa0。

## 已通过验证

CPU 导出格式/参数拒绝测试，以及 123 0 卡 NPU 测试共 11 项通过：
矩阵 hash/正交性/非对称性、逆旋转位置、FP4 ties-to-even、静态 scale/配置 roundtrip、AscendV1 导出、原生 Q/K C7.25 精确对齐、部署 V Triton 精确对齐、真实 Qwen 旋转后 V amax、Q/K QDQ 和 gate 前还原。最终测试记录 test-c4-npu-r3.json/log。

保留两次开发期验证失败：第一次配置负例未捕获 ModelSlim 的 SchemaValidateError；第二次使用 npu_quantize 交叉验证 V 时，当前 CANN 明确不支持 FP4 输出，已改用部署实际 Triton writer。它们都发生在模型导出前，不是权重/精度结果。

## 当前导出

- 机器：141.61.133.123，容器 tp_a4c4_gqa_vllm，物理 0 卡，启动前健康 OK、无进程，显存占用低于 10%。
- 根目录：/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/modelslim-w8a8-c4a4-qk725-hadamard-123-20261115-propagate-r2。
- 输出：上述根目录 /weights；不覆盖之前权重。
- driver PID 566058/start_ticks16112603/session566057；量化 PID567163/start_ticks16118296/session567163；boot d0fe8e99-2a4d-419c-8a4d-1a93d69e8615。
- remote job1794728914083358；export-status.json 为权威阶段，status.py 按 boot/start_ticks/session 验证身份。
- 200 未启动计算，123 其余任务未操作。

导出完成后还需检查十层 V 统计与矩阵、完整权重索引、MTP 官方 unpack/norm 往返规则、视觉/o_proj 保留值、所有权重与源 hash，并保存最终证据包。不能把“已启动”表述为导出成功。本次 scope 是 ModelSlim 导出；现有 vLLM 仍需接入新增 checkpoint 元数据和旋转权重键后才能直接加载，尚未宣称服务可用或精度通过。

## 逐层输出传播修正（2026-11-15 15:49）

首个尝试位于 /mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/modelslim-w8a8-c4a4-qk725-hadamard-123-20261115，在复核 GeneratedRunner / LoadProcessor 时发现：默认 runner 将 observer 前向结果继续传播；LoadProcessor.is_data_free=True，不会在安装 QKV IR 后补跑。该尝试已按 boot/start_ticks/session 只向自己的量化 session 发送 SIGTERM；剩余进程为空，driver 收到 -15 并退出，4075 源文件未改。部分权重、日志、冻结源和停止证据保留，不作为本次正式产物。

r2 的 C4FA3QuantProcessor.postprocess 在完整 GQA 层安装 Q/K/V IR 后，使用同一批输入重新计算整层输出，再写 request.outputs。每层保存 optional/c4_propagation 证据，最终要求样本数=58，Q/K/V 调用数各=58。普通线性注意力层不额外重复前向。新增真实 Qwen 回归测试已确认传给下一层的结果与量化后再次执行的输出逐元素相同，共 11 项检查通过。

r2 从原始 BF16 完整重新导出，未复用前次部分权重或 amax。源码冻结 SHA 和进程身份以本报告当前导出段为准；旧尝试的身份仅保留为历史。

## 首个 GQA 层实跑检查（2026-11-15 16:10）

layer3 的静态 V 统计为 58 条/127259 token/512 通道，Q/K/V 量化后传播各调用 58 次；C7.25 和旋转元数据匹配。已完成 layer0–3 并进入 layer4，原 driver/量化进程身份一致。此项是局部校准与传播检查，完整权重、矩阵张量及最终归档仍待导出结束后验收。

checkpoint-first-gqa-1610 共 6 文件已下载本地，逐成员 SHA/长度/数量通过；manifest SHA256 ad7a65afec06e9bb2774f60a53425d1371177f81854e4378bb0ec4030859c8ec。没有重复启动或修改冻结源。

## 导出验收与新增评测授权（2026-11-15 19:15）

导出于 19:00:45 exit0，driver/量化进程均已退出。最终审计通过：63231 个张量、4075 冻结源未变、十层 V scale/矩阵/58 样本传播记录正确；MTP 768 专家切片与官方 7 个 BF16 norm 往返规则匹配，视觉333/o_proj11保留。证据包9243600字节、110成员，SHA256 ed0339960d530e8f79f22eae0bc24665cd07f5652772b277b8cda57e9c09558a；远端及本地 .NET TarReader 全成员 SHA/长度/精确集合通过。大权重保留共享。

用户现已追加授权：导出完成后接入 vLLM-Ascend，GPQA Diamond、AIME2026、MMMU-Pro standard与vision-only各三轮，与已完成的同机 W8A8C8 比较。此前 export-only 为历史范围，本次继续接入与测试。新独立推理根为 /mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/modelslim-w8a8-c4a4-aisbench-123-20261115，避免改动共享运行源；仅123计算。采样、65536输出、batch64、MTP3、图/prefix配置沿用C8。新增加载实现直接读取scale/rotation/config，Q与K均C7.25；尚待回归与服务验收，不把导出成功当成精度结果。


## vLLM 接入与启动（2026-11-15 19:24）

五处加载/后端修改加一份新单测已同步到独立 inference-source。首批 192 项回归通过；把 dtype 门禁留到目标 Attention 实例化、避免影响 BF16 MTP 描述映射后，相关92项再次通过。新路径用 --kv-cache-dtype mxfp4 和 mxfp_static.mode=checkpoint，禁止 sidecar 参数覆盖 metadata；直接装载十份 V scale 与矩阵，缺失/非法值拒绝，输出逆旋转在 gate 前。首轮运行前冻结1873源/扩展；四原数据SHA、3745图片与12份实际请求参数预检通过。

新 pipeline PID588887/start17404715/session588886，remotejob1794741835237212；首个GPQA driver589478/start17406263、API589603/start17406474/session589603，0卡18200，仍启动中。只有完整文本/图片、10层scale与rotation SHA、target UpdatableGraph和draft prefill/decode实际回放通过后才正式评测；其余三组自动排队/并行，端口18201/18204/18205。未使用200。

## 启动兼容修正（2026-11-15 19:41）

首个推理尝试 E0 在 19:26 因 omitted speculative KV dtype 被继承为 target mxfp4，触发原 C4 的 BF16 draft 契约拒绝。无正式请求，全部本人会话已退出、remaining=[]。保留所有日志；新 r2 仅显式设置 speculative_config.kv_cache_dtype=bfloat16，并复核12请求与1873源SHA。r2 pipeline594704/start17473949/session594703，API595608/start17475662/session595608，0卡18200。

r2 在 19:37 加载首个权重分片时暴露第二个兼容问题：ModelSlim 标准 saver 保留 fa_quant_type=FAKQuant 全局头部，旧 MLA regex 先于 C4 suffix 生效，将 fa_v.scale 误改为 mla_attn。r2 未进行正式请求，pipeline/driver/API 自身会话均退出、remaining=[]。新修正把显式 C4 recipe 与 MXFP8 一样从 FAKQuant 通用映射中排除；新增包含真实全局头部、Q/K/V 六类 scale/offset、二次嵌套 mapper 的回归覆盖。权重没有修改。

r3 独立目录为 /mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/modelslim-w8a8-c4a4-aisbench-123-20261115-r3，准备新的推理源副本；E0/r2 原冻结源和日志保留。此次完整方案比较包含 Q/K/V 格式、Hadamard、新校准及误差传播差异。C4 draft 显式 BF16，C8 历史运行的 draft 实际格式仍需按源码与日志核实，不先宣称完全相同。
## 实际加载验收与静态编译恢复（2026-11-15 20:15）

r3 的完整 194 项单测及真实 checkpoint 映射预检通过：30 个相关源键对应 20 个实际参数，包含十份 V scale 和十份矩阵，二次 mapper 稳定。19:50 十个 GQA 层真实加载成功；audit_loaded_parameters.py 将 post-load V 参数 SHA 与 safetensors 原始导出逐一核对，十层完全相同，矩阵 SHA 与 Q/K alg2 max7.25/schema2 同时通过。

r3 随后在 static_kernel 编译时因 cwd 过长，生成文件名超过 Linux NAME_MAX 而报 OSError36。虽然框架继续捕获图，但不符合本次静态编译要求，因此未进入正式评测。先向本人 API session 发送 SIGTERM；超过90秒仍在捕获后，按 boot/start_ticks/session 核验，仅对同会话3个本人进程 SIGKILL。19:59 pipeline/driver 退出，service-exit remaining=[]。原日志、stop/force-exit 证据和冻结源保留，没有正式请求，未操作其他任务。

唯一活跃尝试为 r4：/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/modelslim-w8a8-c4a4-aisbench-123-20261115-r4。推理源 symlink 指向 r3 冻结1873文件，只将服务 cwd 改为 /tmp/c4-20261115-r4-<dataset>，static_kernel 仍为 true；正式前增加静态编译失败门禁。pipeline653285/start17626718/session653284，job1794744055252330；GPQA driver653877/start17628255/session653877，API654000/start17628479/session654000，0卡18200。20:09 r4十层实际 V SHA再次与导出匹配，31/31和后续50/50静态算子JSON编译成功；20:13图捕获12/49，尚未完成完整图和烟测，正式请求仍0。

新增运行库 observer v2 PID655909/start17644946/session655908，job1794744237618908，在各组进入正式评测后读取本人进程 maps，并将实际加载的库按SHA复制到 runtime-library-snapshot，避免临时编译目录退出后清理。它不初始化NPU、不发模型请求。原observer v1在正式前按归属安全替换，记录保留。最终同时核验1873推理源、3707上游vLLM源、2033外部库/源码/env清单；外部清单属于首次启动时补录，不能称为启动前冻结。

comparison-boundaries.json 已明确：C4 draft显式BF16；C8旧命令未指定draft dtype，当前上游代码推断会继承target mxfp8，但旧基线未冻结这两处上游实现，也没有直接dump，不能将此推断当历史实测。因此比较为同机完整配方，包含Q/K/V、Hadamard、新校准传播以及可能的draft缓存差异，不作为仅KV位宽的消融结果。

integration-checkpoint-200528 已本地验证整包与所有成员SHA/长度/精确集合：1691222字节，214成员+manifest共215；整包SHA336a649ab0a1271592f885c4c0826d887ebe7ebe53a32813376423ce79893447，manifest644d08a11588b23181fbdd4e7d755ab8aaac72b118d8628da8159d22758463bf。该包是接入检查点，非12轮最终精度归档。最终工具 audit_evaluation_rounds.py / audit_runtime_final.py / archive_comparison.py 已语法检查；本地正确验证器是 verify-local-comparison-v2.ps1，已PowerShell语法通过，旧v1替换串导致的格式错误保留且不用。所有最终工具仍需真实12轮产物验收。

## 首服务真实图与烟测通过（2026-11-15 21:08）

GPQA 的 target49/49、draft prefill49/49 和 decode19/19 图捕获完成；20:52:29日志确认 target UpdatableGraph、draft prefill/decode 三条真实回放路径。文本烟测返回 READY.，蓝色PNG返回 blue；十层C4加载门禁通过。21:05检查共98次静态编译成功，零静态错误、零Traceback。正式GPQA第1轮从20:52:30开始，0卡原API654000/start17628479/session654000持续运行，暂无完整轮次审计成绩，不发布部分分数。

调度已在实时空卡检查后自动并行启动其它3组：AIME卡4/18201，driver870575/start17936470、API870835/start17936680/session870835；MMMU standard卡6/18204，driver870740/start17936665、API871108/start17936888/session871108；vision-only卡7/18205，driver871027/start17936871、API871259/start17937084/session871259。21:05三组均处于首阶段图捕获，约15/49，暂无错误；尚未宣称这三组烟测通过。原pipeline653285及GPQA身份一致，没有重复启动，也没有使用200。

observer v2已为GPQA保存86个实际映射库的SHA及快照（对应路径长度总计161120674字节），采集于20:52:47正式运行期。最终仍须复核这些快照与全部源/权重，并完成12轮问题、选项、参考答案和原始回答审计；本阶段只确认首服务接入和真实运行。

21:10补充：integration-checkpoint-210908 已下载并完成本地整包及逐成员SHA/长度/精确集合核验，1805484字节，242成员+manifest共243文件，整包SHA1bd83322ab1803d594468cff420088b8318916e0b9b0e7b82d3b29aa6772b959，manifest236b8d26268b60383fdc138f206b9c2a3ce13f253d969af02e0d5e5227376b43。该检查点保存首服务烟测、图回放和运行库清单，非12轮最终归档。

## 首轮完整内容审计（2026-11-15 21:24）

GPQA第1轮198个正式请求已完成，audit_evaluation_rounds.py --partial首次对本轮C4真实产物验收通过：逐题问题/选项/参考/原始回答/答案提取与官方逐题重评分一致，198成功0失败，MTP有效、prefix配置和实际C4 cache metrics通过；一次warmup单独计数，触顶/无效答案保留。未修改审计断言或回答。GPQA原服务继续第2轮，其余AIME/standard/vision原进程仍在静态编译及图捕获，未发生新失败。完整对比成绩尚待12轮验收。

checkpoint-2123包含首轮完整回答/预测/metrics/infer日志、配置与当前4组状态；3485178字节，97成员+manifest共98文件，本地整包及每个tar/展开成员SHA/长度/精确集合核验通过。整包SHA1b5babb65feb40445b92d5b0d39d64998671732586158ca6b6aa980806742be0；manifestSHA ad4dbbf4ef6c7edfff55985501b488116f1860ff8e494cdcf818c3b6aa1f347e。新增snapshot_c4_progress.py只保存已完整审计轮次，不包含未完成轮的部分答案，禁止覆盖旧tag。仍需最终全源/权重/进程/12轮归档验收。

## GPQA前两轮内容审计（2026-11-15 21:38）

GPQA第2轮完成并由同一轮审计工具通过；累计2轮396正式请求0失败，题目/选项/参考/原始回答/提取/官方重评分及请求数、MTP、prefix均通过，未改断言。原GPQA服务继续第3轮；AIME/standard/vision已完成首阶段捕获，21:35在prefill阶段29/49、29/49、30/49，零新错误且身份一致。

checkpoint-2136完整保存前两轮回答、预测和日志，6157700字节、110成员+manifest共111文件，本地整包与每个tar及展开成员SHA/长度/精确集合通过。整包SHA9584328625a51ce6b39e0ace25103c19adf453ec3e9ee16ad757504cf0ada437；manifest90da8c8866eef30a8f12392bfa7e34235594ea1037cf9359cf83d4c4290e582f。只更新完成轮次检查点，最终12轮对比尚未完成。

## 四组服务均进入正式评测（2026-11-15 21:53）

AIME于21:43:00、MMMU standard于21:42:52、vision-only于21:42:54进入第1轮。三组均通过文本READY、蓝色PNG及target UpdatableGraph/draft prefill/decode真实回放，MMMU另通过真实file://图片的1-token短烟测。四组每组98次static kernel compile success，日志均未出现Traceback或NAME_MAX错误；pipeline、observer、四driver和四API的boot/start_ticks/session与记录一致。GPQA仍在第3轮，前两轮396正式请求已审计通过，不重复启动或重跑。

Observer已完成全部四组运行库清单和快照采集：GPQA86、AIME86、standard88、vision88个映射路径。新三组采集于21:43:19–20，最终仍需逐库快照/源/权重与退出审计。这些是运行期观测，不冒充启动前冻结。

runtime-observation-2153.json保留各组烟测、真实file://诊断、运行状态、身份、运行库清单SHA与日志尾部；46368字节，SHA256 0c2fb475114222ad6cca5d7775fcac00ae807efde103ce5c4bb1d631ed891b6b，已下载本地并核验SHA/长度。完整轮次检查点仍为checkpoint-2136；本次未新增完整轮审计，也未发布部分分数。

## GPQA三轮完整审计与退出（2026-11-15 22:08）

GPQA第三轮已完成，audit_evaluation_rounds.py --partial累计通过3轮594正式请求0失败；原始回答、题目/选项/参考、官方答案提取与逐题重评分、正式请求和独立warmup计数均按原断言通过。AIME与两组MMMU仍在各自第1轮正常运行，原身份一致，22:05日志无Traceback或静态编译错误。未重跑任何已完成轮，尚不发布部分分数。

GPQA于21:57:16正常退出，driver不存在，原API654000/start17628479/session654000只剩Z状态。22:07按boot/start_ticks/session复查，原服务会话无活跃进程、18200端口关闭；未执行kill/reset。gpqa-exit-observation-2208.json共757字节，SHA256 e14796b9b910974b47ba776b8a11e2d448204667a48ba150b880ab62f9ee84a4，已本地核验。

checkpoint-2206保存GPQA全部三轮完整回答/预测/日志与当前各组状态，9361771字节，144成员+manifest共145文件；本地整包、每个tar/展开成员SHA、长度及精确集合通过。整包SHA bcaffe2583d232259eefbad43224960116b379445359454aac0188eee7975da1；manifestSHA 38fe3963b39a59c8ff870c5e225f4290898167bbb33e19d605b2ea743ccdba43。所有旧检查点保留，本包不是最终12轮对比归档。

## AIME首轮审计通过（2026-11-15 22:22）

AIME第1轮30题已完成并通过原轮审计工具的逐题问题/参考/原始回答/提取/官方重评分、配置和请求数核验，未修改断言。累计通过4轮624正式请求0失败（GPQA全部3轮及AIME第1轮）；22:20 AIME原4卡API进入第2轮，两组MMMU仍在各自第1轮正常生成，boot/start_ticks/session一致，无新Traceback或静态编译错误。GPQA保持正常退出，不重跑已审计轮次。

checkpoint-2221含4轮完整回答/预测/日志及当前各组状态；10872940字节，158成员+manifest共159文件，本地整包及每个tar/展开成员SHA、长度、精确集合核验通过。整包SHA e8c152832e7c132db426790b306c54e6024092ec28af47e6f20017a16bfaa1a6；manifestSHA 34ad561dfbe39dae9b3761f9101bc87d05de7e65bc217b1fb18ff50abb170255。所有旧检查点保留；剩余8轮与最终对比审计、归档仍待完成。

## AIME第二轮监控恢复与五轮审计（2026-11-15 23:01）

AIME第二轮于22:42完成30题推理和官方评分，infer/eval worker均记录Task state is finish并退出；AISBench共享status_tmp JSON出现Extra data解析警告并被官方逻辑清空，进度监控丢失完成事件，CLI停在LocalRunner的monitor_p.join()。发现后先确认两组MMMU正常，再保存原始日志、临时状态、30条预测/30份评分、进程身份和官方监控源码。此次属于评测监控故障，回答生成已经完整结束。

严格核验boot d0fe8e99-2a4d-419c-8a4d-1a93d69e8615、monitor PID1505739/start18596623/session870575、父CLI1503926/start18439029及仅剩该监控子进程后，使用pidfd只对监控PID发送SIGTERM。API、driver、会话组及其它任务均未停止；未修改官方冻结源、临时状态、回答或分数。原CLI随后正常完成summary并exit0，原driver自动进入第3轮。第二轮result SHA d1f535b5238d6220d456ae4e5a9d724b5dc210bcc941f5655a6a87ab0de3d767、prediction SHA56c894793f9c66b48feec95dd06f52c4cf9d5da38b44a425100647fe3b47ed4b前后完全一致，没有重新推理。

原audit_evaluation_rounds.py --partial未经修改通过全部5轮654正式请求0请求失败，含GPQA3轮和AIME前2轮；问题/参考/原始回答、官方提取和逐题重评分、配置/metrics/独立warmup计数均通过。AIME继续第3轮，两组MMMU仍第1轮。监控故障及人工恢复不能省略为“全程无异常”，最终报告应保留本条；最终12轮比较尚待完成。

aime-r2-monitor-recovery-2257保留19成员+manifest共20文件，manifest SHA d0e31ba3441d14540cdeb24c501a81de47371ec41fdf9cf2309a8e52777b001e；已本地逐成员SHA/长度/精确集合通过。checkpoint-2259完整保存5轮回答/预测/日志及当前状态，12482424字节、172成员+manifest共173文件；本地整包及每个tar/展开成员SHA/长度/精确集合通过。整包SHA0dc0160a823b0d7d48faafa7530794545bb6ddf64fd7ba5503ddaf9de81441bf，manifest5258a427b43acc9135e1d4a7400d8a182746f146a280ab5d55678ac805d9a73a。所有旧检查点、故障现场和冻结源保留，不能重跑已审计5轮。
## 六轮文本评测审计与AIME退出（2026-11-15 23:38）

AIME第3轮已自行完成并exit0，本轮未采取监控恢复操作。原audit_evaluation_rounds.py --partial累计通过6轮684正式请求0请求失败（GPQA和AIME各3轮），所有问题/选项/参考/原始回答/提取/官方逐题重评分、配置、MTP/prefix和正式请求数均按原断言通过。AIME第2轮的监控恢复历史继续保留，不将六轮请求成功误述为全程无异常。

AIME服务于23:25:14正常退出，service-exit remaining=[]；23:36按boot/start_ticks/session复核，driver870575不存在，原API870835/start17936680/session870835及resource tracker仅Z状态，同session无活跃进程、18201端口关闭，无需kill。aime-exit-observation-2336.json SHA0d877307c04346883a52df16242a7b45a4f6301f7c9d2af0139340de435f1a70已随检查点本地核验。两组MMMU仍在原卡6/7的第1轮持续生成，身份匹配且无API Traceback/NAME_MAX；AISBench临时监控文件解析WARNING仍需在终态关注，不据此中断推理。

checkpoint-2337包含完整六轮回答、预测、日志与各组状态，14125556字节、188成员+manifest共189文件；本地整包及每个tar/展开成员SHA、长度和精确集合验证通过。整包SHA923188dda46e483331a0d4c2d3cb2810d1ce8c2b41f1e35626c675343ad2226a；manifest ef4aad032ae0a17d3a5265b84f0d28312b5f5cddd7688be8428d7878ad769326；partial audit SHA0a6319dc57405f8f8101f5da3cfaabbdf65f146c125ca779f1d604c200809be3。所有旧检查点保留，剩MMMU两个子集各3轮；完整12轮比较及最终归档未完成，不重跑已审计轮。
## MMMU standard首轮审计（2026-11-16 01:24）

MMMU standard第1轮自行完成并exit0，原driver于01:16:43自动进入第2轮；本轮没有人工监控恢复或重新推理。原audit_evaluation_rounds.py --partial累计通过7轮2414正式请求0请求失败，包含GPQA3轮、AIME3轮及standard第1轮。完整问题/选项/参考/原始回答/提取/官方逐题重评分、配置和正式请求数均按原断言通过。vision-only仍在第1轮生成，两组原API与driver的boot/start_ticks/session匹配，无新Traceback或NAME_MAX错误；仍需关注AISBench临时监控文件解析WARNING的终态影响，不据此中断推理。

checkpoint-0122已下载并通过本地整包、每个tar及展开成员SHA/长度/精确集合核验：39255785字节，201成员+manifest共202文件。整包SHA33f449abd08e2b3b133c9021f8e20d2f5d54f5fb616d1e34e5f0fdb6e7d19688，manifest64bb1066b64f7e2b6c90f5e63f4d5820b4b0806bb55d79e414a32431935b628f，partial audit SHA6d4b48cd8fbbb74527360d1e13e6a101f0c08c6afcc11739fc0304d5dc258e0c。旧检查点及AIME第2轮恢复历史全部保留；本包不是最终归档，剩余5轮与完整对比审计尚待完成，不发布部分分数。

## MMMU standard第二轮审计（2026-11-16 05:12）

MMMU standard第2轮自行完成并exit0，原driver于04:52:17自动进入第3轮（CLI1525452）；本轮未人工干预监控或重新推理。05:05只读检查确认原pipeline、observer及两组MMMU driver/API的boot/start_ticks/session均匹配；standard第3轮与vision-only第1轮持续生成，未见API Traceback或NAME_MAX。AISBench status_tmp解析WARNING仍存在，推理期间不据此中断或提前停止监控。

原audit_evaluation_rounds.py --partial按原断言通过8轮4144正式请求0请求失败，包含GPQA3轮、AIME3轮及standard前2轮；完整问题/选项/参考/原始回答/提取、官方逐题重评分、配置、metrics和独立warmup计数均通过。partial audit SHA fe9253330850fc4093ba4403a8e6b6f41f4890cfd42f00de7cdd8f653b551927。已审计轮次不重跑，AIME第2轮监控故障及恢复历史继续保留。

checkpoint-0509保存8轮完整回答、预测、日志和各组状态：64539447字节、214成员+manifest共215文件，已通过本地整包及每个tar/展开成员SHA256、长度和精确集合核验。整包SHA ce761b8c293db7065eb5d34ce190989dbfa172c6e6805d2a43fa8ab99502d775，manifest SHA 91183e7f59b46f90ea209d711cf36e3474ec7e2ceb7822c396f57a976c05faf9。所有旧检查点保留；仍剩standard第3轮及vision-only全部3轮，最终对比审计与完整归档尚未完成，不发布部分分数。

## MMMU standard全部三轮审计（2026-11-16 08:40）

MMMU standard第3轮自行完成并exit0，原driver于08:30:38按流程退出服务remaining=[]；本轮未人工干预监控或重新推理。08:36退出核验确认同一boot，driver870740 absent，API871108/start17936888/session871108及同session resource tracker仅Z，无活跃同session进程，18204关闭，无需kill。退出证据standard-exit-observation-0836.json（8734字节，SHA af47f61cfe409832c12aad55efc925a5c322e322b27e136a2eff2f4ba7e5a2a8）已包含在本地完整核验的新检查点中。

原audit_evaluation_rounds.py --partial按原断言通过9轮5874正式请求0请求失败，包含GPQA3轮、AIME3轮和MMMU standard全部3轮；逐题问题/选项/参考/原始回答/提取、官方重评分、请求数、配置和metrics通过，partial audit SHA 379f8d709829cf889e457706bc6288c8fd3cd2bf6b1792aa042d3827d8db08e9。未重跑已审计轮次，AIME第2轮监控故障和恢复证据继续保留。

checkpoint-0838保存9轮完整回答、预测、日志和各组状态：89508033字节、228成员+manifest共229文件，已通过本地整包及每个tar/展开成员SHA256、长度和精确集合核验。整包SHA 41cb9088ca2f9c0c2d7a0afd4b2e0db543bffb80576ce91e200923667a459280，manifest SHA dd567f9735e48c8dd0e9504730de557bb48f6452689352c4440ae0c271f68aa9。所有旧检查点保留；当前仅原卡7 API871259的vision-only第1轮继续生成，pipeline/observer及vision driver/API身份匹配，无API Traceback/NAME_MAX。还剩vision-only全部3轮和最终对比/完整归档，不发布部分分数。


## 远端监控连接中断（2026-11-16 10:36）

10:35本次检查中remote-dev连续三次报告connection_unavailable、submission=not_sent；随后本机到141.61.133.123:22的TCP只读探测被目标主动拒绝。状态命令未提交，未重启、停止或更改任何服务，也未访问200。最新已确认的运行状态仍为10:20:41：123卡7 vision-only第1轮持续生成，pipeline/driver/API身份匹配，无API Traceback/NAME_MAX；当前运行状态因连接不可用而未知，不能据此断言评测失败或停止。9轮5874正式请求的已审计证据与checkpoint-0838保持不变。

本地remote-connection-observation-1036.json保留连接失败及最后一次成功核验摘要；本段和本地checkpoint的latest_monitor_check尚未同步远端。继续原15分钟恢复检查，连接恢复后首先运行原E/status.py读取新鲜身份/轮次，禁止重复启动。最终全量审计与归档仍待完成。

## 用户要求暂停（2026-11-16 10:58）

用户明确要求“先暂停工作，不要跑了”。自动化123-modelslim-w8a8-c4a4已设为PAUSED，不再自动恢复或补跑。暂停收尾确认指定容器仅有bash PID509307，无推理、driver或observer；没有启动新的评测。已审计9轮及全部旧检查点、AIME监控恢复证据保留，vision-only的3轮和最终审计/归档尚未完成。

暂停前已确认123主机重启，新boot ae4fe88f-5e1e-4d73-a71d-c088d8a4f38a，原容器10:32:15退出255；原fstab NFS 172.27.0.50:/Share→/mnt/share已恢复，并刷新原空bash容器挂载。10:57:20只读status.py已成功读取共享证据：旧pipeline/driver/API均boot_mismatch，vision首轮没有完整result，旧状态文件中的running/watching为重启前残留状态。host-reboot-observation-1053.json及paused-after-reboot-observation-1058.json保留本地证据。本段及暂停checkpoint尚未同步远端；后续须用户明确要求恢复才可继续。
## 迁移 vision-only 至 200（2026-11-16）

用户明确要求改用 141.61.41.200 的 tp_a4c4_gqa_vllm 容器、物理卡5/6，沿用共享目录继续 vision-only 测试；该指令覆盖此前暂停及禁止200计算的历史限制。原123已完成并审计的9轮5874请求不重跑；主机重启中断的vision首轮现场保留，三轮在新目录完整执行，不拼接部分回答。

新根目录：/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/modelslim-w8a8-c4a4-vision-200-20261116。卡5/18305分配第1、3轮，卡6/18306分配第2轮；各为TP1。模型、冻结源码、QK C7.25/QKV Hadamard/schema2、65536输出、batch64、采样、MTP3、prefix、npugraph_ex及静态kernel配置保持原方案。旧r4及全部检查点不覆盖。

启动前两卡无NPU进程、显存基线约4.7GiB；卡5健康OK，卡6既有Alarm，保留告警，不reset。1873推理源、3707上游源和2033外部文件初次SHA核对全部一致；驱动还会核验所有导出权重与实际请求参数，通过真实图回放、参数SHA和图片烟测后才正式评测。当前启动驱动为PID552/start998980/session552、boot afa3953c-8112-46b5-8664-7246db73f7d5；仅用于身份匹配，后续必须新鲜核验。

恢复检查已绑定本次会话，automation_id=200-vision-only-c4a4，每15分钟检查，无新情况静默。原123恢复检查已不存在，不重建旧任务。执行状态查看新根status.py与pipeline-status.json；本地记录在artifacts/modelslim-w8a8-c4a4-vision-200-20261116。最终仍需逐题审计、实际运行库及进程退出核验、完整本地归档验收。

本次vision的C4在200运行，C8历史基线仍在123；最终报告必须声明跨机器完整方案比较，不能称同机或仅KV位宽消融。C8 draft格式的历史证据限制继续保留。
200启动检查点补充：全部23份导出权重/配置SHA与1873/3707/2033份冻结文件校验通过；3份实际请求参数预检通过。卡5 API1156/start1030887、卡6 API1233/start1030939，两路均已实际加载10层C4参数并进入49档图捕获，静态编译正在成功推进，最新检查零Traceback/静态编译错误。首次QFA桥接JIT后共享.so SHA仍与原冻结清单完全相同。完整烟测及正式评测尚未开始，不能将服务启动当作精度完成。

control-start.tar.gz（494536字节、58成员+manifest）已下载本地，整包及所有成员SHA/长度/精确数量核验通过；SHA7771d90c777923095fa6e3fb117736b6efd32c2fb204b6daf74a5622cd7342d6。此包仅为启动控制文件与检查点，不是最终评测归档。最终两卡运行时审计/归档工具已生成并语法通过，仍待真实完整12轮证据验收。
## 200 vision-only 两路正式评测启动（2026-11-16 20:24）

卡5/18305和卡6/18306均已完成target49档、draft prefill49档与decode19档图捕获，每路98次静态编译成功，零静态编译错误/NAME_MAX/Traceback。READY、blue及真实file图片短烟测通过；两路实际加载的十层V scale SHA均与导出一致，矩阵与QK C7.25/schema2门禁通过。运行库observer已保存卡5共89、卡6共88个映射路径及快照。

两路AISBench从20:12开始正式执行，卡5第1轮、卡6第2轮，后续卡5自动第3轮；20:24检查身份匹配、API持续生成。AISBench status_tmp JSON解析竞争警告仍存在，不据此停止推理或提前kill监控；待完整预测/评分/worker退出后再判断是否确实终态卡住。尚无新增完整轮审计，不发布vision分数。

runtime-start-observation-2024.json已下载本地，9454字节及SHA cd0e3a17a2a4f4f9d4cb5f2f0b495a14a754e5aaf60edfe859708ee51f4ba5dc核验通过；最新执行检查点已同步本地。原9轮与123中断记录保持。
## 200 vision-only 第2轮完成，待三轮联合审计（2026-11-17 12:54）

卡6第2轮已完整保存1730条正式预测，物理行JSON解析、题目ID唯一性及请求success字段检查通过，AISBench CLI自然退出码0，正式结果JSON已生成；未执行监控进程恢复或重跑。卡6 API、driver和observer均已退出，service-exit.remaining为空，18306端口关闭。卡5第1轮仍运行，第3轮依原计划等待，尚不发布vision分数。

卡6日志在12:54:26由wrapper触发abort模式关闭后记录了一次AsyncLLM output_handler的EngineDeadError及semaphore退出警告；关闭前无Traceback，完整预测和评分文件生成在先。原日志及SHA保留，不能将退出期日志隐去或写成全程无异常。静态编译错误和NAME_MAX仍为0，最终仍需运行时联合审计。

里程碑round2-complete-observation-20261117T0454.json已下载本地，30838字节、SHA ab168eda087d4f9aee1a1bce7883f0cbba8cd905b461f4664841e2409561129a核验一致；执行检查点已同步。此为单轮完成证据，未替代最终12轮审计和整包逐成员本地验收。
## 200 vision-only 第1轮完成、第3轮自动启动（2026-11-17 13:09）

卡5第1轮已完整保存1730条正式预测，物理行解析、唯一ID及success检查通过，AISBench CLI自然退出码0，正式结果已生成。第3轮于13:08依原driver自动启动，复用同一正式API/冻结配置，未重启服务、未修改参数、未执行监控恢复。原两轮现均完成，仍等待第3轮及最终12轮联合审计，不发布vision分数。

round1-complete-round3-start-observation-20261117T0509.json已下载本地，30099字节、SHA e68e0e0213f91a9aae557730d0fcf401bf85404e7b664addafe35b3611412556核验一致；含第1轮预测/结果/metrics证据SHA及第3轮新鲜CLI身份。最新checkpoint已同步。卡6退出期日志边界与原123中断/监控恢复历史继续保留。
## 200 vision-only 最终完成与本地封存（2026-11-18）

三轮均完整完成，每轮1730题：第1轮1291正确（74.6242774566%）、第2轮1285正确（74.2774566474%）、第3轮1277正确（73.8150289017%）。均值74.2389210019%，样本标准差0.4059982178个百分点；C8历史均值74.5857418112%，差值-0.3468208092个百分点。5190正式请求全部success，三轮CLI均自然退出0；第2轮保留1条输出触顶，均无无效提取/空回答。原9轮5874请求保持，不重跑、不拼接中断轮。

| 数据集 | C4正确 r1/r2/r3 | C8正确 r1/r2/r3 | C4均值% | C8均值% | 差值pp |
|---|---|---|---:|---:|---:|
| GPQA Diamond | 163/165/163 | 170/159/170 | 82.6599326599 | 84.0067340067 | -1.3468013468 |
| AIME2026 | 27/29/26 | 27/28/28 | 91.1111111111 | 92.2222222222 | -1.1111111111 |
| MMMU-Pro standard | 1320/1307/1307 | 1305/1319/1302 | 75.7996146435 | 75.6454720617 | +0.1541425819 |
| MMMU-Pro vision-only | 1291/1285/1277 | 1297/1284/1290 | 74.2389210019 | 74.5857418112 | -0.3468208092 |

运行时联合审计通过：两路实际加载10层V SHA、Hadamard/QK/schema、target及draft图回放、真实file图片烟测、89/88个映射库及各快照SHA、1873/3707/2033冻结文件和23份权重SHA；静态编译无错误。200的API/driver/observer/supervisor及两次CPU收尾session共9个均已退出，18305/18306关闭；旧123仅使用历史退出、重启和暂停证明，fresh_proc_check=false。200三轮无监控恢复；123 AIME第2轮曾恢复监控终态的历史保留。

结束硬件边界：启动卡5为OK，最终只读查询发现Alarm80E3A201（电压调节功能异常），准确发生时间及数值影响未知；卡6启动已有Alarm，结束仍有告警。两卡均无设备进程、计算利用率0；未reset/清告警。两路API在结果生成后的wrapper abort关闭期间各出现一次EngineDeadError及semaphore警告，原始日志与时序已保存，不能宣称全程无异常。硬件边界和最终退出证据由completion-evidence补充包封存。

比较为跨机器完整方案：vision C4运行在200，其他C4及全部C8历史基线在123；QK C7.25、QKV Hadamard、新V校准和量化传播均为方案差异。C4 draft显式BF16，C8历史draft实际格式未冻结，当前源码推断不能替代历史实测。迁移边界以comparison-boundaries-migration.json为准，原same-host文件保留为历史。

主包2725818594字节，SHA256 a68767be775f23c3ca57330ff1bb5261fbcc684d2744a760d6cd391df3a94e46；manifest SHA256 46e3eb3bfc763c9c485e257254b073f4c01c1395def6a52b32dc3a4370a6214c。远端及本地.NET TarReader完成16291成员（16290文件+manifest）的SHA/长度/精确集合核验，含48份正式预测/结果、3745图片及所有失败/重启/恢复历史；大权重未下载。补充包14537字节，SHA256 ed40919473b70c3f8ab5ee7ab42c5bcf535b9a0f80ab37993bb23214c847d72a，远端与本地16成员逐一核验通过。

收尾控制工具曾修复轮次展示顺序与继承same-host文本；精确停止本人CPU归档进程后保留原工具/状态/日志/未完成包SHA，未修改推理源、答案或正式结果。首次本地核验的全局evidence-map唯一性断言因历史根重复而阻断，按迁移根精确定位后重新完成全包核验；失败脚本及记录保留。主包控制状态是归档时快照，最终以包外completion-status.json和execution-checkpoint.json为准。

完整本地目录：artifacts/modelslim-w8a8-c4a4-vision-200-20261116。completion-report.md为结果总览；local-archive-verification.json、completion-evidence-local-verification.json及local-final-evidence-verification.json为验收记录。completion-status SHA256 16a58908f86bae3caa3b55e7f625364ceb55e1fdf4d4696db652a5f26d37c50c，最终checkpoint SHA256 d38b9513075b72caac29119bf1488fc467f7f162badaffe2b7d513e289d01bf1。本次automation_id=200-vision-only-c4a4已删除，不再续跑检查。没有提交或推送。