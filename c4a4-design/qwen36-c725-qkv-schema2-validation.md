# Qwen3.6 C7.25 K/V + QKV Hadamard + schema2

状态：200双组并行评测已全部完成。C4三轮均值84.0067%，同机BF16为83.5017%；完整审计、进程退出、归档下载及2145个成员的本地校验均已通过。以下保留原取消、迁移与换卡历史。

## 本轮配置

- 模型 `/mnt/share/weight/Qwen3.6-35B-A3B`；123服务器原容器 `tp_a4c4_gqa_vllm`，5/6卡。启动前均无任务进程，显存足够，Warning按用户此前授权允许。
- TP2、MTP3、FULL_DECODE_ONLY、prefix OFF，显存比例0.75且预留8192MiB。沿用原AIS Bench输入输出/采样参数及GPQA Diamond198，完整3轮，不改长度上限或筛选答案。
- K：FP4 E2M1，block_size=32、scale_alg=2、dst_type_max=7.25。Q：原alg0。
- V：`e=floor(log2(amax*8/7.25+9.6e-7))-2`，静态per-channel E8M0。复用上轮QKV旋转BF16校准的58条/127259 token逐通道amax，重新导出20份独立scale；不是重新采集激活。
- 5120通道中449个scale与QKV C7不同。逐通道用C7公式精确复现旧值，20份新scale保留rotation元数据，原120份scale不变。
- QKV Hadamard：seed0、D256、矩阵SHA `ca70b9def213b38eb4066626b2808ec33d5694c38f7f92085703936e8925d22d`，RoPE后旋转，输出在gate前乘R.T；与上一轮完全相同。
- 59份源码、12份已安装算子/环境文件、120旧scale、20新scale预检通过。模型后端未改动，仅添加独立实验驱动。

## 对照与证据

历史QKV Hadamard C7/schema2：164/172/165，共3轮，均值84.3434343434%。本轮只运行C7.25 QKV/schema2，BF16、C8及全层单算子不重跑。旧记录、scale与完整归档不改动。

运行目录：`/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/c4-c725-qkv-schema2-123-20261114`。

驱动：[c725_qkv_experiment.py](tools/c725_qkv_experiment.py)，提供run/status/finalize三个入口。禁止在原目录重复启动run；先查driver-status、matrix-status和进程身份。

[本地检查点](../artifacts/c4-c725-qkv-schema2-123-20261114/execution-checkpoint.json)。三轮完成后核验198成功/0失败、问题选项与参考答案、MTP、prefix零复用、target/draft真实图回放和全部哈希，在123核验自身进程退出，然后生成最终报告及完整证据归档。本地下载和全部成员SHA验证完成后才标记全部完成。

若机器维护或运行中断，保留原始证据，在独立恢复目录完整补跑未完成轮次；已审计的完整轮次不重跑，不拼接题目、不按分数选择性重跑。不得重置设备、操作其他任务、提交或推送。详细记录只存本项目。

## 取消记录

首次启动在预热期间遇到HCCL RunAicpuCommInit/get kernel failed，尚无正式GPQA结果。用户随后要求停止，未执行换卡通信检查或重启。现有20份新scale、旧scale及启动失败日志均保留；未停止其他任务或重置设备。取消证据见本地artifacts/c4-c725-qkv-schema2-123-20261114/cancellation.json。

## 200迁移（2026-11-14）

用户在停止123尝试后，明确要求迁到141.61.41.200、同一共享目录、同名tp_a4c4_gqa_vllm容器继续。新运行目录：`/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/c4-c725-qkv-schema2-200-20261114`。驱动PID4667，boot ID `8c32b0ef-381f-4553-8670-63b057011352`。

- 1/7卡均健康OK、无任务进程、单卡114688MiB；启动前余量满足0.75显存比例与8192MiB预留。两rank各3次小规模HCCL AllReduce精确校验通过，预检进程已退出。
- 共享模型代码和12份算子/环境文件哈希、torch2.10.0/torch_npu2.10.0.post7/Triton3.6.0/Transformers5.19.0与原实验一致。仅新增[c725_qkv_200_experiment.py](tools/c725_qkv_200_experiment.py)，用于迁移校验及200主机报告。
- 新驱动在加载前及最终审计都检查20份V sidecar与123已生成文件SHA256一致。Q/K/V量化参数、Hadamard矩阵、schema2、TP2/MTP3/FULL_DECODE_ONLY/prefix OFF和原AIS Bench参数保持不变。
- 123失败尝试完整复制进新根目录prior-attempt-123；原目录仍保留，未重写旧driver状态。新根migration-observation.json记录授权及设备、主机变化。
- 200为Ascend950PR、npu-smi26.2.rc1，历史123为Ascend950DT、25.6.rc2；最终与C7均值的对比属于跨机器历史参照，需要标明这一限制。
- 先查status再操作，禁止重复启动run。完成三轮后在200运行finalize，不能将其日志写进待归档根目录。进程退出必须在200按boot/start_ticks检查。

[200执行检查点](../artifacts/c4-c725-qkv-schema2-200-20261114/execution-checkpoint.json)。新归档须包含prior-attempt-123及三轮完整回答；本地逐成员验证工具位于该artifact目录。全部完成前不宣称精度结果。

2026-11-14 12:07：200模型启动及smoke通过，target/draft均已记录真实ACL graph replay。第1轮正式GPQA已开始，随后自动运行第2/3轮；尚无完整轮次精度。

## 用户指定换卡与BF16并行对照（2026-11-14）

当前新根目录：`/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/c4-c725-qkv-bf16-parallel-200-20261114`。

- C4：0/1卡、18086端口，C7.25 K/V、QKV Hadamard、schema2；BF16：2/3卡、18087端口，原生BF16无旋转。
- 两组同时独立运行，各3轮GPQA Diamond198，共6轮1188题。保持TP2/MTP3/FULL_DECODE_ONLY/prefix OFF、原采样及长度限制，显存比例0.75、8192MiB预留。
- 旧200 1/7卡第一轮在153已发、121已收、0请求失败时按用户要求中断，所有自身进程已退出。无完整轮次，旧部分回答全部排除，不拼接、不计分；原因是用户换卡指令。
- 模型及量化后端不改；新增4份支持独立端口和指定mode的评测工具及[并行驱动](tools/parallel_c725_bf16_experiment.py)，旧工具保留。新工具不继承历史完整分数，C4/BF16各3轮均为新跑。
- 旧200整个实验复制进新根prior-attempt-200，其内含123失败记录。20份C7.25 V scale逐字节沿用；原校准与旧scale保留。
- 最终主要比较本次同机C4与BF16，同时列出123 C7 QKV/schema2历史均值，明确跨机限制。
- 在200运行parallel_c725_bf16_experiment.py status查看两组；两组全部完成且进程退出后运行finalize。其日志不可写入待归档根目录。完整包需含6个新完整结果、9历史参考、全部中断证据、scale和冻结源码，本地逐成员核验后才标记完成。

[并行执行检查点](../artifacts/c4-c725-qkv-bf16-parallel-200-20261114/execution-checkpoint.json)。当前禁止恢复1/7旧根或123任务；只跟进此并行根。
2026-11-14 并行启动核验：driver PID11898；C4 API11979（0/1，18086），BF16 API11978（2/3，18087）。两组HCCL预检通过，64源码/12算子环境/120旧scale/20新scale通过预检，本地64源码与冻结版本一致。两组target/draft均观察到真实ACL graph replay。当前C4阶段gpqa_running，BF16阶段gpqa_running，完整精度结果待三轮完成及最终审计。

## 200双组并行最终结果（2026-11-14）

本轮已完成全部六轮、内容审计、进程退出核验、完整归档下载和本地逐成员校验。C4在0/1卡，原生BF16在2/3卡；两组均为TP2、MTP3、FULL_DECODE_ONLY图模式、prefix关闭，原AIS Bench输入输出与采样参数，每轮完整GPQA Diamond198。

| 模型 | 第一轮 | 第二轮 | 第三轮 | 平均准确率 | 样本标准差（百分点） |
|---|---:|---:|---:|---:|---:|
| C4：K/V C7.25 + QKV Hadamard + schema2 | 165/198 | 166/198 | 168/198 | 84.0067% | 0.7715 |
| BF16：同机新测、无旋转及KV量化 | 165/198 | 166/198 | 165/198 | 83.5017% | 0.2916 |

本次C4均值比同机BF16高0.5051个百分点。历史123的C7 QKV/schema2为164/172/165，均值84.3434%，本次C7.25低0.3367个百分点；这是950PR与950DT的跨机器历史对照，不能将差异全部归于7→7.25，也不能凭三轮随机采样认定稳定提升。

C4的Q维持alg0，K采用block32/scale_alg2/dst_type_max7.25；V按floor(log2(amax*8/7.25+9.6e-7))-2导出。沿用历史QKV旋转后BF16校准58条/127259 token的逐通道amax，未重新采激活。20份C7.25 sidecar与123准备文件逐字节相同，5120通道中449个相对C7变化；seed0/D256矩阵及gate前R.T不变，旧120份scale未改。

六轮1188成功、0请求失败；所有题目、选项、参考答案及198个回答完整。C4三轮分别1/3/2个长度上限及无效提取答案，BF16为3/0/0，均保留计分；各轮无抢占。已核验MTP实际接受token、prefix查询/命中均为0、target/draft真实图回放，以及64源码、12算子环境文件、120旧scale、20新scale哈希。110条本任务进程身份记录在200按boot/start_ticks核验退出。

旧200 1/7卡任务按用户指令在153已发、121已收时中断，部分题目不计分、不拼接；完整旧200及更早123启动失败证据保存在prior-attempt-200中。本次六轮均为独立完整新运行。

完整包为38,030,355字节，含2145个成员（2144个证据文件加manifest），服务器与本地均核验每个成员的SHA256、长度和精确清单，含6份新完整结果、9份历史参考、20份新scale、20份原BF16校准及全部中断证据。

归档SHA256：`cfc48219e7cd5d2768305eb1c619684fb2e6c7309312c6051a38d28f5248b820`。

- [最终审计](../artifacts/c4-c725-qkv-bf16-parallel-200-20261114/final-audit.json)
- [本地归档核验](../artifacts/c4-c725-qkv-bf16-parallel-200-20261114/local-archive-verification.json)
- [完整证据包](../artifacts/c4-c725-qkv-bf16-parallel-200-20261114/c4-c725-qkv-bf16-parallel-200-20261114-complete.tar.gz)
- [六轮完整答案](../artifacts/c4-c725-qkv-bf16-parallel-200-20261114/results)

当前execution-checkpoint与本地校验回执在归档后更新，包内保留打包时的检查点；完整tar及manifest保持不变。下方或前文的启动/中断状态均为历史过程记录。
