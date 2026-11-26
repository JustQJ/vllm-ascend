# Qwen3.6 ModelSlim W8A8C8 MXFP8 导出与精度评测（123）

> 最终状态（2026-11-15 10:34）：W8A8C8导出、12轮精度、最终审计和完整归档本地逐成员验收均已完成。结果见末节；以下过程记录按时间保留，旧调度不得重启。

2026-11-15，用户要求直接按 ModelSlim 配方导出新权重，并在 141.61.133.123 上继续既定评测。200 有其他任务，本轮不在 200 启动计算。此前普通 FP8 per-head 权重的启动失败与准备证据保留。

## 配方与环境

- 输入：/mnt/share/weight/Qwen3.6-35B-A3B。
- 独立实验根 R：/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/modelslim-w8a8c8-aisbench-123-20261115。
- 输出权重：R/weights；旧权重不修改。
- 容器：tp_a4c4_gqa_vllm；123 为 Ascend950DT，npu-smi 25.6.rc2。
- ModelSlim：本地干净源码 cb83735f94604d308fa67784b1c7d5b8981addf0，源码包 SHA256 2eb4f66b565120abdfa3e24ae952940e5a96678002b058d695b170003273d493。
- YAML：lab_practice/qwen3_5_moe/qwen3_6_35b_a3b_w8a8c8_mxfp8.yaml，SHA256 64bba864a4b56b618c1e62fc7672ef804f81ca2d37293ae7986f2ff480211162。未改量化配置。
- W/A MXFP8 per-block；Q/K MXFP8 per-block 动态；V MXFP8 per-channel 静态。o_proj、视觉模块与 MTP 按 YAML 保留浮点。
- 校准集 qwen3_cot_vlm_text，完整 58 条。index.jsonl SHA256 371d68d9220223dace2034982e756e1af0fedcb6bf75dbbaa3c6e373cc366432。重新执行整套逐层量化，不复用以前 BF16 V amax。
- 独立 quant-venv：Transformers 5.2.0、tokenizers 0.22.2、easydict 1.13。原 vLLM 推理环境不升级/降级。
- PEP660 安装未建立数据目录链接，已在本轮源码副本补齐 config/lab_practice/lab_calib 链接；tokenizers 与固定 Transformers 的版本冲突已在独立 venv 解决。

## 评测要求

官方 AISBench HEAD 于本次再次核对，仍为 1aead1f12bd35746672bda5a8a2330f58e7ba651。复用已校验的独立 AISBench 源码与 venv，运行结果落入新 R。

| 数据集 | 每轮题数 | 轮数 |
|---|---:|---:|
| GPQA Diamond | 198 | 3 |
| AIME2026 | 30 | 3 |
| MMMU-Pro standard 10 选项 CoT | 1730 | 3 |
| MMMU-Pro vision-only CoT | 1730 | 3 |

temperature=1.0、top_p=0.95、top_k=20、min_p=0.0、presence_penalty=1.5、repetition_penalty=1.0，max_out_len=65536、batch_size=64。12 个真实 API 请求体构造预检通过，未向模型发送正式请求。四个 loader 题数均完整，standard 的 2015 张图、vision-only 的 1730 张图通过图片校验。

服务保持用户指定的 TP1/DP1、MTP3、FULL_DECODE_ONLY、prefix ON，npugraph_ex/static_kernel=true、super_kernel=false，max_model_len=135168、max_num_seqs=128、max_num_batched_tokens=4096、memory_utilization=0.90。为实际启用新导出的 MXFP8 KV，增加 --kv-cache-dtype mxfp8；当前分支不会仅凭 checkpoint 描述自动开启该后端。

## 01:14 初始执行与继续入口（历史）

01:14 已启动 ModelSlim 导出：0 卡，driver PID97690，量化 PID98365，boot d0fe8e99-2a4d-419c-8a4d-1a93d69e8615。01:19 已完成第 0 个 decoder 层并进入第 1 层；该状态是历史检查点，后续需读取当前状态文件。

R/pipeline.py 等待现有导出完成，禁止重复启动；之后执行 audit_export.py，检查十层 KV recipe、V scale uint8/512元素、MTP/视觉/o_proj 浮点键完整性以及所有输出文件 SHA。验收成功才启动首个服务及文本/图像冒烟；首个服务进入正式评测后，才并行启动其余三个数据集。

计划 GPQA=0/18100，AIME=1/18101，MMMU standard=4/18104，vision=5/18105；4/5 是用户此前允许的 Warning 卡，启动前仍须检查无任务且显存充足。123 的 2/3 卡已有其他任务，不操作。每个数据集有独立服务与三轮完整输出；失败不自动重试或拼接部分题目。

运行状态：export-status.json、pipeline-status.json、各 dataset-driver-status.json、eval-logs/<dataset>/status.json。服务归属记录 boot/start_ticks/session；仅处理本任务进程。详细脚本与证据保存在 R 及本地 artifacts/modelslim-w8a8c8-aisbench-123-20261115。

01:19当时正式精度结果为 0/12 轮，未宣称完成。全部轮次结束后仍需审计完整问题/参考答案/成功数、输出上限与无效答案、实际 MTP/prefix/图执行证据，核验源码/权重不变，确认本任务服务退出，再保存完整报告与证据并本地核验。不得按分数选择性重跑，不提交或推送。

## 01:40 调度与依赖检查点

校准 token 复核为 127,259（58 条，最短 1,318，最长 2,958）；01:31 已完成首个 GQA 层 layers.3 的 FA3 量化，继续后续逐层导出。导出 PID98365 未重启。

123 的 1 卡在准备期间被其他任务占用。旧 pipeline102155 当时仍处于 waiting_for_export、无子任务，已按 boot/start_ticks 核验并仅停止该等待调度器；保留 pipeline-r1-retained 与 pipeline-replacement.json。现由 pipeline_v2.py 继续等待同一导出，PID109825/start_ticks11012543，状态 pipeline-r2-status.json。启动时从 0/4/5/6/7 实时选择无任务且空闲显存超过90%的卡，排除1/2/3；固定端口不变，实际卡号以各服务 launch.json 为准。无卡时等待，不操作其他任务。

已在空闲6卡通过推理依赖预检：共享目录编译的 AddRmsNormBias 与 CPU 参考最大绝对差为0，原生 MXFP8 动态量化形状正确。runtime-dependency-probe.json 记录实际加载的三个自定义 op_api 库；探针已正常退出。第一次1卡探针在 NPU 初始化之前因空卡断言退出，不是算子执行失败。

自动续跑检查 123-modelslim-w8a8c8 已更新到新调度器；每20分钟检查，正常运行静默，仅异常、需要处理或最终完成通知。服务级图模式/完整多模态检查与12轮正式精度仍待导出完成后执行；当前不能报告模型精度。

01:41 补充：最新空卡检查只剩6/7。等待调度器升级为 pipeline_v3.py，支持前一数据集完成并释放卡后复用，避免不足4张卡时一直等待。v2仍无子任务时按身份安全停止，记录保存在 pipeline-r2-retained 和 pipeline-r3-replacement.json。当前唯一活跃调度器 PID110944/start_ticks11029680/session110394，状态 pipeline-r3-status.json，remote job1794678070605947；旧r1/r2状态为保留历史，active-pipeline.json 为当前入口。导出进程一直保留，正式评测尚未开始。

## 04:19 导出验收通过，启动首个服务

03:58:27 ModelSlim 以 exit_code=0 完成导出，原量化进程已退出。输出10个权重分片，共23个文件、40,560,219,718字节；63,221个张量。04:19:04 audit_export_v3.py 验收通过，export-audit.json 保存全部文件SHA256、十层 QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL 及十份uint8/512元素V scale。权重和配方均未改动。

原审计工具有两项不符合实际 ModelSlim 导出格式的假设，失败日志完整保留，未启动任何正式评测：

- MTP原始两个合并专家张量被官方adapter拆为256组gate/up/down，共768个BF16张量，依据model_adapter.py第973–1000行；已对全部专家与原始对应切片逐元素核对。
- 7个MTP RMSNorm经过官方adapter第150–159、1035–1054行的BF16加1再减1往返，不能要求与原始参数逐字节相同。所有差异均精确匹配该规则，没有使用误差容限放行。最大绝对变化0.0078125，变化元素合计3809；详情mtp-norm-roundtrip-observation.json。未对权重作修正，继续测试用户指定的原ModelSlim方案。

MTP最终785个BF16张量、844,640,768元素按上述规则校验；视觉333个BF16张量、446,571,248元素与原始精确一致；11个o_proj共92,274,688元素与原始精确一致。审计v1/v2及pipeline v3/v4的失败状态保留，不属于模型精度失败。

当前唯一调度入口为active-pipeline.json→pipeline_v5.py / pipeline-r5-status.json。PID122269/start_ticks11975659/session121711，remote job1794687531501249。旧v3/v4已退出。新调度复用已通过的导出审计，不重新导出或重复审计。

GPQA首个服务在0卡、端口18100开始启动；driver PID122316/start_ticks11976149，API PID122358/start_ticks11976372/session122358；boot仍为d0fe8e99-2a4d-419c-8a4d-1a93d69e8615。launch.json及日志确认TP1、MTP3、prefix ON、FULL_DECODE_ONLY、npugraph_ex/static_kernel开启，以及kv_cache_dtype=mxfp8。状态为starting，完整图/文本/图片服务检查与12轮精度仍未完成；其他数据集尚未启动。

## 04:28 图捕获与本地检查点

首个服务已完成10分片加载及内存规划，31/31首批静态kernel编译成功。04:27第一组FULL图捕获及78/78静态kernel编译成功，随后进入下一组；总计49个待捕获shape，仍处于启动阶段。没有正式精度分数，不能仅据编译成功宣称完整服务验证通过。

保存checkpoint-0428静态快照，包含导出审计、v1/v2失败证据、当前v5调度、服务命令/日志和真实进程身份。72个成员（另加manifest）已下载到本地项目，全部SHA256和长度、精确文件数73均核验通过。该检查点的运行状态是采集时刻快照，后续以共享active-pipeline.json和实时状态为准。共享execution-checkpoint.json已更新为export_audited_service_starting；权重仅保留在共享目录。
## 05:33 服务通过冒烟，GPQA开始

首个服务完成图捕获。05:29文本请求返回“READY.”，蓝色PNG请求返回“ blue”；日志分别在05:29:37/39/50确认draft decode、target UpdatableGraph、draft prefill真实回放。烟测MTP accepted=6、drafted=9；该小样本不作为模型精度成绩。

GPQA于05:29:50进入第1轮正式AISBench，0卡/18100，原服务未重启。动态空卡调度随后在6卡启动AIME2026（driver451221/start12400403，API451315/start12400579/session451315，18101），7卡启动MMMU standard（driver451305/start12400565，API451479/start12400763/session451479，18104）。vision-only仍排队；不使用其他任务的卡。此时没有完成的正式轮次。

首次正式评测开始时补充runtime-environment-manifest.json：记录进程身份、实际映射的4个扩展/OPP库哈希、依赖版本、env.sh哈希和npu-smi快照，观察器未初始化NPU。841份已冻结推理源码/扩展重新核验，变化0。该环境记录的时间为首轮运行期，不伪称全部二进制均在启动前冻结。

checkpoint-0533包含51个成员及manifest，已下载到本地并核验全部成员SHA/长度和精确52文件数。后续仍需12轮全量题目、成功数、评分、MTP/prefix/图、源码/权重不变及服务退出审计；无正式最终成绩。
## 06:03 MMMU 本地图片请求兼容修复

05:49 MMMU standard 在 AISBench 首个 warmup 请求返回 HTTP400；正式推理未开始，无预测/结果文件。AISBench 外层 CLI 仍返回0，驱动通过结果文件存在性检查识别失败并正常释放自身服务，service-exit.json 的 remaining 为空。GPQA与AIME继续正常执行。06:02已有GPQA前2轮和AIME首轮的结果文件，共3轮待最终审计；未发布准确率。

已用官方loader保存的首条样本、官方MMPromptTemplate和VLLMCustomAPIChat构造两个MMMU子集的真实多模态请求体，确认图片URI为file://。05:57在GPQA API上以max_tokens=1诊断复现HTTP400，响应明确为“Cannot load local files without `--allowed-local-media-path`.”；该请求在推理前拒绝，没有生成答案，不属于正式GPQA样本。证据为mmmu-http400-diagnosis.json和两个first-request-diagnostic.json。此前12份参数预检仅使用普通文本，未覆盖本地图片传输；文本/蓝图base64烟测成功不能替代该检查。

独立恢复根N为R后缀-mmmu-r2；recover_mmmu.py于06:00启动，PID484409/start_ticks12583642/session484408，remote job1794693624769316，boot未变。仅新增服务参数--allowed-local-media-path，范围限定为数据集真实目录/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/w8a8c8-official-aisbench-200-20261114/datasets。此目录名保留历史准备来源，实际恢复计算仍仅在123。权重路径仍为R/weights；6份MMMU配置与原文件逐字节相同；采样、64k输出、batch64及模型编译配置不变。已复核841份推理源码/扩展SHA无变化。新服务还会在正式评测前使用实际file://图片进行独立短输出烟测。

恢复调度不修改原运行源：等待原尝试失败且按boot/start_ticks/session确认退出、remaining为空、无正式结果后，实时检查0/4/5/6/7空卡及显存再启动。原vision-only此刻仍在7卡启动阶段，恢复任务尚未占卡。GPQA/AIME不重跑；MMMU只在新N完整执行各3轮，不拼接题目。R/evidence-map.json记录GPQA/AIME来自R、MMMU来自N，最终归档必须同时包含R与N及原失败证据。

checkpoint-0603包含117成员及manifest，共118文件，已下载本地项目并逐成员SHA256、长度和数量验证通过。manifest SHA256为a910f5bdbda781b4ae3c60105a299a593cbff7d5ec173251c4419e9bc76bda34。该包是中间检查点，未包含最终12轮完整答案，不能视为最终归档完成。恢复后的服务可用性和正式精度仍待验证。

06:05补充：GPQA三轮均已生成结果，状态evaluations_complete_pending_audit；自身服务remaining为空，于06:03释放0卡。恢复调度已在0卡/18104启动MMMU standard，driver487396/start12600195/session487396，API487442/start12600365/session487442，boot未变；launch.json明确包含限定数据集目录的allowed-local-media-path。当前仍处于启动阶段，尚未验收本地图片烟测。AIME第2轮运行，原vision-only仍图捕获；恢复调度继续等待原vision退出后接续。recovery-start-observation-0605.json及最新execution-checkpoint已单文件哈希校验下载到本地。

## 06:37 六轮文本审计与 MMMU 双组运行检查点

GPQA与AIME2026各三轮全部结束，两个原服务正常退出，service-exit.json的remaining均为空。audit_evaluation_rounds_v4.py --partial已审计6轮、684个正式回答，成功684、失败0；全部问题顺序、选项、参考答案、原始回答和提取结果对应一致，逐题官方评分重新计算一致。无效答案与65536输出上限样本保留。每轮AISBench另有一次完整预热，服务成功计数为正式题数加1；服务端length计数包含预热，不能直接当作正式题目触顶数。

审计工具v1的字符串splitlines会把JSON字符串中的Unicode段落分隔符误当作物理行；v2误把结果中的origin_prediction当作模型后处理文本。现按物理JSONL行读取，并依据官方openicl_eval.py将origin_prediction与原始prediction核对，再单独核验模型/数据集后处理。原工具和失败日志均保留，未修改原始回答或结果；这些是审计工具修正，不是模型精度失败。

原MMMU vision-only也因同类file://权限问题在预热失败，原服务退出后由N自动恢复；未产生正式回答。N的standard与vision-only均通过文本、蓝色base64图片及实际file://图片短烟测，现分别在0卡/18104、7卡/18105运行第1轮，日志已有正常HTTP200、batch64和真实MTP接受计数。standard API487442/start12600365/session487442，vision API495710/start12644549/session495710；恢复调度仍为484409/start12583642/session484408。两组参数未变。原pipeline-r5已因保留的两个MMMU失败结束为failed；GPQA/AIME exit0，这个历史状态不能当作恢复调度的新故障。

当前AISBench目录来自git archive，没有.git。verify_aisbench_source_v2.py核对源码包的git PAX提交标识1aead1f12bd35746672bda5a8a2330f58e7ba651，并逐文件检查2766个源码文件，全部一致。源码包SHA256为7f83d5672189660fb146ea8bcbc1253b036537e7da82d25280d3bf2599bbc25b。仅允许官方runner在tmp生成的UUID_params.py临时配置，并单独记录；没有修改AISBench源文件。

新增最终工具audit_runtime_final.py和archive_final_evidence.py，语法检查通过，尚未在12轮完成后执行验收。入口依次为：使用R/bench_python.sh运行audit_evaluation_rounds_v4.py --root R（不带--partial），随后python audit_runtime_final.py --root R，最后python archive_final_evidence.py --root R。执行前需核实两个调度与全部自身服务退出；日志不要重定向到R或N内部，以免归档收录仍在变化的文件。最终审计包含841份推理源/扩展、23份权重、4个运行库、配方/校准/图片及AISBench源码核验；归档同时保留R/N、12轮完整回答、原失败记录、校准与源码。几十GB权重仅保留共享，归档保存完整哈希清单与配置。C8 scale证据包含已审计导出值、冻结的loader映射、完整模型加载及真实C8执行；目前未采集运行中NPU scale直接dump，最终报告需保持这个证据边界。

checkpoint-0637包含248成员及manifest，共249文件、成员总大小44,752,447字节，保存6轮文本完整预测与结果、当前两根状态、日志和工具。已下载本地项目并逐成员SHA256、长度及精确数量核验通过；manifest SHA256为8a61fb541ef55f317543bbc8a3740aa5e5d0207638d0c5edf694399ae511eb6d。该检查点仍不是最终12轮归档；最终成绩和完整完成通知需等待MMMU六轮、最终审计与归档本地核验。

本地最终归档核验脚本verify-local-archive.ps1已保存于本轮artifact根目录，并同步R。PowerShell语法检查通过；它将核验整包和每个成员SHA/长度/精确数量，以及12结果/12预测、3745图片、841源码和4运行库。最终包尚未生成，因此该脚本尚未完成真实最终包验收；请使用已安装的PowerShell 7 pwsh.exe执行。


## 07:51 MMMU 首轮审计及本地完整回答检查点

MMMU-Pro standard与vision-only各自第1轮已完整结束，两个原恢复服务均继续进入第2轮。API仍为0卡18104的487442/start12600365/session487442与7卡18105的495710/start12644549/session495710，boot未变；没有重新启动或更换配置。07:45核验两个服务身份与实时指标一致，请求abort/error均为0。

audit_evaluation_rounds_v4.py --partial现已通过8轮、4144个正式回答，成功4144、失败0，包含全部六轮文本和两个MMMU首轮。对实际MMMU结果格式的全部问题、选项、参考答案、原始回答、提取结果和官方逐题重评分核验通过，未改审计断言、原始证据或采样。剩余standard及vision-only各第2、3轮继续运行；本次未发布最终准确率。

checkpoint-0751包含274成员及manifest，共275文件、成员总大小235,701,152字节，保存8轮已审计完整回答及两根状态、日志、配置、工具。已下载本地项目，并逐成员SHA256、长度、精确文件集合与数量核验通过；manifest SHA256为02002f6acd9ceb04f66b63ddf6614567628d60926752f8e420ca3836ff9a225a。snapshot_progress.py可按新标签保存已审计完整轮次，旧检查点均保留。execution-checkpoint.json与evaluation-audit-partial.json已同步本地。

该检查点仍不是12轮最终归档。最终服务退出、源码/权重/运行库核验、完整归档与本地逐成员验收尚待剩余四轮完成；最终工具及其证据边界不变。


## 08:47 MMMU standard 第二轮审计检查点

MMMU standard第2轮已完整结束，原0卡服务于08:30:30进入第3轮；vision-only仍由原7卡服务执行第2轮。08:44按boot/start_ticks/session核验恢复调度、两个driver/API全部匹配，服务请求abort/error均为0；没有启动新服务或更换配置。

audit_evaluation_rounds_v4.py --partial现已通过9轮、5874个正式回答，成功5874、失败0，包括6轮文本、standard前2轮和vision-only首轮。全部问题、选项、参考答案、原始回答、提取与官方逐题重评分保持一致。剩余standard第3轮及vision-only第2、3轮继续；未重跑已完成轮次，也未按成绩筛选。

checkpoint-0846已下载并核验286成员及manifest，共287文件、成员总大小323,992,717字节。逐成员SHA256、长度和精确文件集合全部通过；manifest SHA256为1a9d04b41dffa6cd919733f649f4342c7d5bf39b4edbe5a9d8a5ff85717b6de1。本地及共享execution-checkpoint、partial audit与核验记录已更新，旧检查点保留。最终12轮的进程退出、源码/权重/运行库审计及完整归档本地验收仍未执行，本检查点不代表任务全部完成。


## 09:08 两个 MMMU 子集均进入末轮

vision-only第2轮已完整结束；09:05新鲜状态确认其进入第3轮，standard也继续第3轮。两个API仍为原0卡/18104与7卡/18105，恢复调度、driver/API的boot/start_ticks/session均匹配；09:04请求abort/error均为0。服务、采样配置和权重未改动。

audit_evaluation_rounds_v4.py --partial通过10轮、7604个正式回答，成功7604、失败0，包括六轮文本和两个MMMU子集各前两轮；逐题问题、选项、参考、原始回答、提取及官方重评分核验通过。现在仅剩standard与vision-only各第3轮，已审计轮次不重跑。

checkpoint-0907已下载并逐成员核验298成员及manifest，共299文件、成员总大小423,850,526字节；SHA256、长度及精确文件集合全部通过。manifest SHA256为133972cf1e2e27cea06e345a4125a8c184b749ea2ebc75412e74bb26be9d29b2。本地及共享检查点、partial audit和本地核验记录已更新；全部旧证据保留。最终12轮审计、服务退出核验和完整归档本地验收仍待执行，尚不宣布全部完成或发布最终成绩。


## 09:49 MMMU standard 三轮结束，11 轮内容审计通过

MMMU standard第3轮已完整结束，服务在09:35:07退出且service-exit.json的remaining为空。09:44在123按boot/start_ticks/session核验：其driver487396已不存在，原API487442仅为Z状态，不是活跃服务；18104端口已关闭。vision-only仍由原7卡API495710/start12644549/session495710执行第3轮，恢复调度和该组driver/API身份一致，请求abort/error均为0。没有重启、更换配置或干预其他任务。

audit_evaluation_rounds_v4.py --partial已通过11轮、9334个正式请求，成功9334、失败0；包含全部六轮文本、standard三轮及vision-only前两轮。问题、选项、参考答案、原始回答、提取结果和官方逐题重评分全部核验通过，审计断言与原始证据未改。现在仅剩vision-only第3轮；已审计轮次不重跑。

checkpoint-0947已下载本地，逐成员SHA256、长度和精确文件集合核验通过：310成员及manifest，共311文件，成员总大小516,971,757字节。manifest SHA256为93d61c29766cfd353b86c0140dc746be4b7455a6c7ce8e7295bdaf0c1ec1faeb。本地及共享execution-checkpoint、partial audit与本地核验记录已更新，旧检查点保留。完整12轮的最终运行时审计、所有自身进程退出检查、完整归档和本地逐成员验收仍待末轮完成；本检查点不是最终归档。



## 10:34 最终完成：12 轮审计及完整归档本地验收

全部计算仅在141.61.133.123的tp_a4c4_gqa_vllm容器执行。MMMU vision-only第3轮于10:23结束，服务10:23:22退出，恢复调度10:23:27完成退出。10:24新鲜进程检查及最终运行时审计确认本任务20个已记录的进程身份和对应session均无活跃残留；所有服务退出记录remaining=[]。原pipeline-r5的failed仅表示保留的两次MMMU预热失败，未覆盖或伪装成成功。

本次W8A8C8 MXFP8正式精度如下，三轮各自完整评测，平均值为三轮准确率算术平均：

| 数据集 | 第1轮正确数 | 第2轮正确数 | 第3轮正确数 | 平均准确率 | 样本标准差（百分点） |
|---|---:|---:|---:|---:|---:|
| GPQA Diamond | 170/198 | 159/198 | 170/198 | 84.0067340067% | 3.2075014955 |
| AIME2026 | 27/30 | 28/30 | 28/30 | 92.2222222222% | 1.9245008973 |
| MMMU-Pro standard10 CoT | 1305/1730 | 1319/1730 | 1302/1730 | 75.6454720617% | 0.5244954755 |
| MMMU-Pro vision-only CoT | 1297/1730 | 1284/1730 | 1290/1730 | 74.5857418112% | 0.3760928959 |

12轮共11064个正式请求，全部成功，0个请求失败。无效提取答案共7个、正式输出达到65536 token上限共11个、空回答0个，均保留并参与原官方评分，没有拼题或按分数选择重跑。每轮另有一次完整AISBench warmup，服务计数与正式请求口径分开核验。逐轮明细见[生成报告](../artifacts/modelslim-w8a8c8-aisbench-123-20261115/report.md)和evaluation-audit.json。

最终audit_evaluation_rounds_v4.py不带--partial通过全部问题、选项、参考答案、原始回答、提取和官方逐题重评分。audit_runtime_final.py通过23个权重文件、841份推理源码/扩展、4个实际运行库、2766份AISBench源码及配方、校准和图片哈希；真实target UpdatableGraph、draft prefill/decode回放、MTP活动、prefix配置和计数、服务参数与进程退出均核验通过。evaluation-audit.json SHA256为9c5cfe221e24f0d6a7f0a235ba467483e2535757835fb421f2e34f231024ef1f。

导出使用原ModelSlim配方、58条/127259 token整套逐层校准。768个MTP专家张量与原始切片精确一致；7个MTP RMSNorm因官方adapter的BF16加1再减1往返造成3809元素舍入变化，最大绝对差0.0078125，全部精确符合转换规则，未修补权重。视觉333张量及11个o_proj与源张量一致。C8 V scale证据包括导出值、冻结loader映射、完整加载与真实C8执行，未取得运行中NPU scale直接dump；运行库清单首次记录于首轮GPQA运行期，不冒充启动前冻结。

完整归档同时包含R与MMMU恢复N、12份正式结果及12份完整预测、3745张数据图片、校准记录、841份推理源码/扩展、4个运行库、源版本和权重manifest，以及全部原始失败、恢复和历史检查点。约40.56GB的23个权重文件保留在共享R/weights，不随证据包下载。

- 完整包：modelslim-w8a8c8-aisbench-123-20261115-complete.tar.gz，2,120,863,598字节。
- 包SHA256：9387015f79fb2907cd9ccd6a69be365c1f2c6319bfc035e1dc7bbddcfd8bdc1d。
- artifact-manifest.json SHA256：759c281e22488804752ab4a0a5cea8486a9bd2d9b1c94fef3cb8e468d98f524c。
- 远端和本地均核验6699个内容成员及1个内嵌manifest，共6700文件，逐成员SHA256、长度、精确集合和数量全部匹配。
- 本地目录：[artifacts/modelslim-w8a8c8-aisbench-123-20261115](../artifacts/modelslim-w8a8c8-aisbench-123-20261115)；Windows .NET TarReader核验完成时间2026-11-15T02:34:02.1371179Z，记录为local-archive-verification.json。

归档内容保持不变；归档内report/checkpoint记录的是打包时“待本地验证”阶段，本节与归档外local-archive-verification.json及完成检查点记录最终验收。原始要求、失败历史及旧检查点保留。未提交或推送，详细记录仅存本项目。

