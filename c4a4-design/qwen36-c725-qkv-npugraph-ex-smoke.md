# Qwen3.6 C4：官方950DT编译配置启动验证

最新结论（2026-11-14重编后）：**自定义OPP及扩展重编安装成功，原官方编译开关下服务启动和6个生成请求通过，target/MTP draft真实图回放通过。** 下文保留初次缺符号失败及后续修复证据。

2026-11-14初次尝试（重编前），结论：**当时200容器环境直接套用官方三个编译开关，服务启动失败，未产生生成输出。** 首个致因是框架注册Norm/Quant融合规则时无法加载自定义ACLNN符号；尚未完成npugraph_ex后端编译、静态kernel生成和模型图回放，不能把结果归为C4 QFA不支持npugraph_ex。

## 本次实际配置

- 主机141.61.41.200（950PR），容器tp_a4c4_gqa_vllm；0/1卡，启动前均无进程且HBM余量足够。0卡Warning按既有授权使用。采用的是官方950DT**编译配置**，硬件仍为200的950PR。
- 模型/mnt/share/weight/Qwen3.6-35B-A3B，BF16权重+C4 KV；TP2/MTP3，FULL_DECODE_ONLY，prefix OFF，显存比例0.75，max_model_len65536，max_num_seqs32，max_num_batched_tokens2048。
- C4保持K block32/scale_alg2/dst_type_max7.25；Q alg0；静态V保持C7.25已有20份sidecar。QKV Hadamard及schema2不变，未重新校准。
- 原模型启动命令仅改下列编译配置、删除compilation_config.mode=0，并隔离端口18088和缓存路径。框架实际解析为CompilationMode.VLLM_COMPILE=3。

```json
{
  "enable_npugraph_ex": true,
  "enable_static_kernel": true,
  "enable_super_kernel": false
}
```

保留原capture_sizes=[4,8,16,32,64,96,128]。没有复制官方示例的W8A8权重、TP1、CPU绑定或prefix开启等其他设置。

环境：torch2.10.0、torch_npu2.10.0.post7、vLLM0.33.1.dev0+empty、triton-ascend3.6.0+dev20261013220622、Transformers5.19.0、CANN9.2。完整argv与环境见证据目录launch.json；官方示例快照为official-qwen36.md。

## 结果与定位

1. 主模型和MTP权重加载成功，每rank报告33.12GiB；两rank分别安装10个GQA层的静态V量化。
2. api.log:47确认VLLM_COMPILE=3；:268-276进入torch.compile、确认三个开关true/true/false，设置LOCAL_WORLD_SIZE=2。
3. api.log:410-434首个致因：norm_quant_fusion_pass.py:58，AddRMSNormQuantPattern.pattern调用torch.ops._C_ascend.npu_add_rms_norm_bias，报错：

```text
RuntimeError: aclnnAddRmsNormBias or aclnnAddRmsNormBiasGetWorkspaceSize
not in libopapi.so, or libopapi.sonot found.
```

4. 框架默认fuse_norm_quant=True。graph_fusion_pass_manager.py:59-64创建AddRMSNormQuantFusionPass；该pass在注册时执行示例张量路径，即使当前模型权重为BF16，也可能在真正匹配模型图之前触发此依赖。C++适配器通过EXEC_NPU_CMD调用aclnnAddRmsNormBias。
5. 只读nm检查CANN libopapi.so、隔离QFA libcust_opapi.so、fla_npu libcust_opapi.so，均没有上述两个符号。editable vllm-ascend/vllm_ascend/_cann_ops_custom目录存在但无内容。检查结果及库SHA见library-symbol-check.json。
6. 健康检查未达到200，六个生成用例均未执行。没有npugraph_ex完成、static kernel生成或target/draft实际回放的成功证据。本轮不是GPQA精度测试。

当前最可能原因是现有可用推理环境未配齐框架编译融合所依赖的vllm-ascend自定义OPP；先前mode0跳过了该融合注册路径。下一步最小修复是隔离补齐/正确加载包含AddRmsNormBias的同版本vllm-ascend自定义算子，再原配置复测。也可单独诊断fuse_norm_quant=false，但那属于附加配置实验，不能记作官方原配置已通过；本轮未执行该变更或安装。

## 退出、完整性及证据

worker报错后API等待未退出，记录致因后先向身份核验的自身API发送TERM；54秒后仍未退出，再终止该API，由原driver清理其私有会话。api-forced-exit.json和service-stop-report.json保留全过程；result.json中API exit -9是清理结果，**不是最初致因**。

在200按boot ID/start_ticks验证12条自身进程记录已全部退出。前后64源码、12算子/环境文件、120旧scale及20份C7.25 scale哈希一致；没有修改模型代码、安装算子、重置卡、停止其他任务、提交或推送。

共享证据根：/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/c4-c725-qkv-npugraph-ex-200-20261114。
本地证据：artifacts/c4-c725-qkv-npugraph-ex-200-20261114。包含启动器、实际命令、原始日志、冻结源码、库符号检查、退出回执、清单及归档。

参考：[官方Qwen3.6-35B-A3B教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.6-35B-A3B.html)、[TorchAir静态Kernel说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/basic/static_kernel_compile.md)。

本地归档复核：110/110成员SHA与长度一致，整包497827字节，SHA256=e4f3f828f75f0a810ab9946626ff52ae116b2275232afa61729003b6bdf95655。检查点与本地验证回执为归档后更新，包内保留打包时快照。

[原始启动日志](../artifacts/c4-c725-qkv-npugraph-ex-200-20261114/logs/api.log)、[符号检查](../artifacts/c4-c725-qkv-npugraph-ex-200-20261114/library-symbol-check.json)、[进程退出](../artifacts/c4-c725-qkv-npugraph-ex-200-20261114/final-process-check.json)、[完整证据包](../artifacts/c4-c725-qkv-npugraph-ex-200-20261114/c4-c725-qkv-npugraph-ex-200-20261114-evidence.tar.gz)。

## 安装根因补核（2026-11-14，归档后只读调查）

原始安装并未省略vllm-ascend编译：install_vllm_ascend.sh第6行明确设置COMPILE_CUSTOM_KERNELS=1，并执行pip install --no-build-isolation --no-deps -e .；install-vllm-ascend-r2.log记录editable wheel构建/安装成功。但同一行错误地设置了SOC_VERSION=Ascend950PR_958b（首字母大写）。

此版本csrc/build_aclnn.sh第188行使用区分大小写的Bash正则 ^ascend950；未匹配时第227—228行打印skip build_aclnn并exit 0。envs.py直接读取环境变量，setup.py原样传给该脚本，没有转换大小写。故这组安装参数会跳过自定义ACLNN/OPP构建，同时允许后续C++扩展构建和pip安装成功。已只读核对原基线3288b04c4同样的分支，并在200 Bash独立验证：Ascend950PR_958b不匹配，ascend950pr_958b匹配。原pip日志没有保留内部skip输出；上述判断依据是保存的实际安装参数、源码控制流和独立正则验证，不声称在安装日志中找到了该行。

当前远端vllm_ascend/_cann_ops_custom实际只有.gitkeep，没有算子安装产物；之前“为空”指无OPP内容。正确命中的ascend950分支会包含add_rms_norm_bias，而此次加载库缺少它。QFA来自ops-transformer单独构建安装的包，因此此前QFA/C4可运行不证明vllm-ascend全套自定义OPP完整。

因此应将此前笼统的“容器环境问题”修正为：本次安装参数大小写错误导致ACLNN构建被成功跳过，且安装后的产物检查遗漏。这不是未执行过pip编译，也不是已证实Docker本身故障。后续修复应使用匹配分支的小写SOC参数构建同版本OPP，检查安装目录、两个ACLNN导出符号及实际调用后，再原配置重试服务。本次补核尚未重编、重装或重启模型；原失败归档及其SHA保持不变。

依据：[原安装脚本](../artifacts/c4-qfa-integration-20261112/runtime/install_vllm_ascend.sh)、[安装日志](../artifacts/c4-qfa-integration-20261112/runtime/logs/install-vllm-ascend-r2.log)、[自定义算子构建分支](../csrc/build_aclnn.sh)。
## 重编安装与原配置复测完成（2026-11-14）

在200/tp_a4c4_gqa_vllm使用小写SOC_VERSION=ascend950pr_958b、COMPILE_CUSTOM_KERNELS=1、MAX_JOBS=16，重新执行pip install -v --no-build-isolation --no-deps -e .，完整ascend950自定义算子分支和C++扩展均构建安装成功。_cann_ops_custom已包含实际OPP，两个AddRmsNormBias ACLNN符号存在；8个真实NPU调用通过独立FP64参考，确认实际加载新库。安装版本0.1.dev6603+g3288b04c4.d20261114。

同一模型、C7.25 K/V、QKV Hadamard、schema2、TP2/MTP3、FULL_DECODE_ONLY、prefix OFF在0/1卡复测，保持enable_npugraph_ex=true、enable_static_kernel=true、enable_super_kernel=false，实际VLLM_COMPILE=3。主模型7/7、draft prefill7/7、draft decode4/4捕获完成；冷启动健康检查1432.935秒（约23分53秒），其中静态包安装/Gloo同步曾有长等待，随后自行推进，不能仅凭60秒等待提示判断死锁。

6个请求均HTTP200且输出非空：17+25→42，法国首都→Paris，排序→1,2,7,9，中文矩阵定义，英文completion，以及3×7支铅笔→21。5例stop；英文completion按设置达到64 token长度上限。thinking用例未配置reasoning parser，content保留原始思考文本和关闭标签，最终答案正确。两个模型均有真实图回放日志，MTP draft_tokens=369，prefix_hits=0。本次是temperature=0启动/输出冒烟验证，没有重跑GPQA，不据此更新精度表。

64份模型源码、12份原QFA/环境文件、120旧scale和20份C7.25 scale哈希均保持。新OPP及扩展单独记录产物哈希；setup.py/build_aclnn.sh与构建前备份一致。依赖下载与共享目录时钟导致的构建重试、smoke-r1空卡门禁未通过及smoke-r2正式尝试均保留，详见完整报告。只清理本次服务，200按boot/start_ticks复核无残留；当前测试API已停止，重编安装产物保留。

[完整复测报告](../artifacts/c4-c725-qkv-npugraph-ex-rebuild-200-20261114/report.md)、[终审](../artifacts/c4-c725-qkv-npugraph-ex-rebuild-200-20261114/final-audit.json)、[实际拉起命令](../artifacts/c4-c725-qkv-npugraph-ex-rebuild-200-20261114/smoke-r2/launch.json)、[API日志](../artifacts/c4-c725-qkv-npugraph-ex-rebuild-200-20261114/smoke-r2/logs/api.log)、[构建日志](../artifacts/c4-c725-qkv-npugraph-ex-rebuild-200-20261114/logs/build-install-r4.log)。

完整证据包47748373字节，SHA256=686fc8af4eacb68eca5071bb315253f648df60b6561b2ab90b149a1379f0e43b。远端及本地均核验6850个清单成员的SHA/长度，另有清单本身，共6851项。[证据包](../artifacts/c4-c725-qkv-npugraph-ex-rebuild-200-20261114/c4-c725-qkv-npugraph-ex-rebuild-200-20261114-evidence.tar.gz)、[本地核验回执](../artifacts/c4-c725-qkv-npugraph-ex-rebuild-200-20261114/local-archive-verification.json)。归档保持封存，归档后的completion-checkpoint单独记录本地核验完成状态。