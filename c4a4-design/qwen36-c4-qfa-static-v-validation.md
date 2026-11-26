# Qwen3.6 C8/C4 静态 V 接入验证记录

首轮验证日期：2026-11-12；以下进程和设备描述为当时快照。当前已迁移 123，九轮 MTP3+图模式 GPQA 已完成且测试服务已关闭，prefix 并发严格一致性未通过；最终结果见 [后续验证](qwen36-prefix-mtp3-graph-gpqa.md)。设计与实现说明见 [接入方案](qwen36-c4-qfa-static-v-integration.md)。

## 完成结果

已按 BF16 校准 → C8A8 → C4A4 的顺序，在指定容器和模型上完成真实执行。最终 C4 服务保留运行，模型别名 `qwen36-c4`，容器内地址 `http://127.0.0.1:18086/v1`，TP=2，设备 0、1。

| 验证 | 结果 | 证据 |
|---|---|---|
| BF16 校准 | 20 条请求；10 GQA 层 × 2 ranks 共 20 个 sidecar | `calibration-summary.json`、`calibration-requests/manifest.json` |
| 校准覆盖 | 每层每 rank 5420 个真实 token；shape [1,256]；所有 maxima 有限、非负 | `calibration-summary.json` |
| CPU 回归 | 89 passed，17 个依赖弃用告警 | `logs/test-c4-c8-budget-r2.log` |
| NPU writer | 18 组逐字节相同 | `c4-writer-qfa.json`、`logs/c4-writer-qfa-r2.log` |
| 正式 QFA | 3 组与独立 CPU 参考输出相同，max_abs_error=0、RMSE=0 | `c4-writer-qfa.json` |
| BF16 整网 | 10/10 请求成功 | `bf16-smoke.json` |
| C8A8 整网 | 10/10 请求成功；10/10 文本与 BF16 相同 | `c8-smoke.json` |
| C4A4 整网 | 页预算修正后 10/10 请求成功；6/10 文本与 BF16 相同 | `c4-smoke-r2.json` |
| 三模式核验 | 算术、排序、标记检索断言通过；同 prompt 顺序/并发答案相同 | `bf16-c8-c4-comparison.json` |
| 源码检查 | 20 个变更 Python 文件 Ruff 0.14.0 lint/format 通过，git diff --check 通过 | 本地验证；代码哈希见 `source-manifest.json` |

测试是链路和基本行为验证，不能把 6/10 文本相同当作 60% 任务准确率。C4 的中文解释与 Python 函数名字/文字有所变化，实际算术 42、排序 [-2,0,5,9,9]、长上下文 ALPHA42 正确。代码生成只做输出检查，没有正式执行正确率测评。

每种整网 smoke 先执行五条顺序请求，再以反序、最多四个并发请求执行同一组 prompt。最长 prompt 3032 tokens，启用 chunked prefill，每次最多 1024 batched tokens；最大上下文配置 8192。温度 0、seed 0、关闭 thinking、最多生成 96 tokens。原始响应和耗时均保留；耗时包含冷启动/JIT，并非性能基准。

## 容量核验与修正

三次运行的可用于 KV cache 的规划预算均为 37.07 GiB，gpu-memory-utilization=0.65：

| 模式 | scheduler block | GDN padding | 日志规划 KV token 容量 |
|---|---:|---:|---:|
| BF16 | 1152 | 9.92% | 2,512,709 |
| C8 | 2560 | 22.14% | 3,445,321 |
| C4 最终 | 4096 | 3.82% | 5,852,364 |

C4 kernel block 仍为 512，scheduler/kernel 比为 8。初次 C4 运行虽然请求通过，但上游 early hybrid alignment 用 uint8 未压缩逻辑宽度计算页大小，导致 scheduler block 仍为 2560，容量收益被 padding 抵消。已在 NPUPlatform 的 hybrid 对齐阶段使用 C4 `customize_spec`，按包含 scales 的 272 B/token/head 预算；最终日志 `c4-server-r2.log` 确认上述数字，并重新完成全部十条整网请求。

C4 规划容量约为 C8 的 1.70 倍、BF16 的 2.33 倍。该指标含 GDN 混合页管理，不是全模型显存压缩率、最大实际并发测试或吞吐提升。

## 版本和部署路径

- 主机：SSH `141.61.41.200`；容器：`tp_a4c4_gqa_vllm`。
- 上传后的插件仓库：`/mnt/share/t00970481/A4C4-Quant/vllm-ascend`，`gqa-c4a4@3288b04c4` 加本次未提交修改。本地同分支；未执行 commit/push。
- 算子仓库：`/mnt/share/t00970481/A4C4-Quant/ops-transformer`，`f9067e085`。
- 参考仓库：`/mnt/share/t00970481/tmp2/vllm-ascend`，`qwen3.6-c4a4@c6d7e7d58`，只读参考。
- 模型：`/mnt/share/weight/Qwen3.6-35B-A3B`，BF16 权重，Qwen3_5MoeForConditionalGeneration，40 层（10 GQA、30 GDN）。
- 任务运行目录：`/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime`，下文记为 WORK。
- 硬件/CANN：Ascend950PR_958b，CANN 9.2；插件编译使用完整 SOC_VERSION。
- Python 3.11.10、torch 2.10.0+cpu、torch_npu 2.10.0.post7、triton-ascend 3.6.0+dev20261013220622。
- 上游 vLLM 使用插件 `.github/vllm-main-verified.commit` 固定的 `24420b7a341af37a9bc3186e6fc6062512e0aa58`，隔离安装在 WORK/upstream-python；本地构建版本标记 `0.33.1.dev0` 不是发布版号。
- QFA vendor：`c4vllm_transformer`，Python 包：`cann_ops_transformer_c4vllm`，隔离安装在 WORK/install、WORK/python。

容器原有 vLLM 0.30.0 缺本分支所需的 `kv_cache_planning`。已按固定 commit 安装上游 Python 部分，不覆盖原系统 vLLM。匹配的 Transformers、tokenizers、oss-harmony、FastAPI/Starlette、Pillow 等安装于任务目录，`env.sh` 指定搜索顺序。

容器宿主挂载 `/usr/lib64` 的 OpenSSL 3.0.12 与镜像 Python 不匹配。使用原镜像自带 OpenSSL 3.5.6 的库副本存入 WORK/compat-libs，通过该任务的 LD_LIBRARY_PATH 选择，未替换系统库。

## 校准产物

`WORK/calibration` 保存 20 个 `.pt`，包含原始逐通道 `v_max`、layer_name、rank、tp_size、model 和 observed_tokens。校准 maxima 全局范围 0.484375–17.875。C8 按 `/448`、C4 按 `/6` 分别生成 E8M0 scale；sidecar 数据可以共用，编码 scale 不相同。

校准语料来自参考分支流程的 20 条科学主题 prompt，当前仅作为首轮打通样本。`calibration-requests/prompts.jsonl` SHA256：`6ee63216486355ff0a0da3cdf270cd4d520cdb4360cc7dff559c8151950f559f`。每个 sidecar 的 SHA256 见 calibration-summary.json。采集完成后 `calibration.enabled` 已移除。

sidecar 限定相同模型路径和 TP=2，不能在更换 checkpoint 内容或 TP 后直接视为有效。本次没有全量权重哈希，也没有用代表性业务语料做精度验收。

## 复现入口

当前 C4 服务已经运行；其 PID 记录在 WORK/c4.pid，日志为 WORK/logs/c4-server-r2.log。在进入容器后使用该任务的 env.sh：

```bash
docker exec -it tp_a4c4_gqa_vllm bash
WORK=/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime
source "$WORK/env.sh"
```

模型未运行时，完整 C4 启动脚本为：

```bash
bash "$WORK/serve-c4.sh"
```

核心参数：

```text
--model /mnt/share/weight/Qwen3.6-35B-A3B
--served-model-name qwen36-c4
--tensor-parallel-size 2 --kv-cache-dtype mxfp4 --enforce-eager
--max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 1024
--no-enable-prefix-caching --gpu-memory-utilization 0.65
--language-model-only --host 127.0.0.1 --port 18086
--additional-config '{"mxfp_static":{"mode":"static","scale_dir":"/mnt/share/t00970481/A4C4-Quant/vllm-qfa-runtime/calibration","qfa_package":"cann_ops_transformer_c4vllm"}}'
```

模型 smoke（output 必须选新路径，脚本拒绝覆盖已有证据）：

```bash
cd /mnt/share/t00970481/A4C4-Quant/vllm-ascend
python c4a4-design/tools/smoke_test.py \
  --base-url http://127.0.0.1:18086/v1 --model qwen36-c4 \
  --output "$WORK/c4-smoke-new.json"
```

校准和 C8 分别使用 WORK/serve-calibrate.sh、WORK/serve-c8.sh。需要重新校准时，为新实验选独立 scale_dir/output-dir；BF16 服务就绪后才启用 `calibration.enabled` 并调用 tools/calibrate.py，完成后移除标记。原始执行脚本为 WORK/run-calibration.sh。

离线布局、校准和 C8/C4 路由回归：

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -m pytest -q \
  tests/ut/attention/test_kv_cache_mxfp4.py \
  tests/ut/attention/test_mxfp_static.py \
  tests/ut/attention/test_attention_c8_mxfp.py \
  tests/ut/attention/test_kv_cache_mxfp.py
```

NPU writer/QFA 对拍入口 tools/check_mxfp4.py。测试 T=31/32/33/511/512/513，(D,H)=(128,1)/(256,1)/(256,2)，负 slot、随机物理页、非连续 V、非零 storage offset、page padding 和 guard。QFA 测试 (KV,Q)=(33,33)/(513,3)/(513,1)，分别覆盖 TND 与 N2TGD Qscale，调用正式 mode5/schema2 并比较 ops 仓独立 CPU golden。

## 构建入口和已解决的兼容问题

- WORK/build_qfa.sh：构建 QFA 与 metadata；Ascend950；独立 vendor c4vllm；`KERNEL_TEMPLATE_INPUT=InOutLayoutType=2,HasVTail=false`，同时包含 C8 NZ 和 C4 BBND。使用本机已有离线依赖，构建后的 OPP 安装到 WORK/install，torch 扩展 wheel 安装到 WORK/python。成功日志 logs/build-r3.log。
- WORK/install-upstream.sh：精确上游 commit 的 VLLM_TARGET_DEVICE=empty 构建。成功日志 logs/install-upstream-r2.log。
- WORK/install_vllm_ascend.sh：MAX_JOBS=16、SOC_VERSION=Ascend950PR_958b、COMPILE_CUSTOM_KERNELS=1，`pip install --no-build-isolation --no-deps -e .`。成功日志 logs/install-vllm-ascend-r2.log。
- 静态 `npu_quantize` FP4 在本环境不支持，已改为 Triton V writer；native Q/K 动态 MXFP4 正常。
- 初版 writer 每 program 处理 4 个 token/head 行时，H=2 的 V 字节对拍失败。改为每 program 1 行后全部 18 组通过；未在缺少最小复现时归因于编译器缺陷。当前优先正确性，后续再优化批量处理。

## 证据位置和未验收范围

服务器所有证据位于 WORK。本地副本位于项目 `artifacts/c4-qfa-integration-20261112/runtime`，包含 JSON、成功阶段日志、启动/构建脚本、校准语料和 20 个 sidecar；实际修改文件的哈希见 source-manifest.json。

本次不构成生产质量/性能验收。待做：扩大校准集，固定 prefix logits 和任务精度评测，quant_schema 0/1/2 对比，prefix caching 开启后的冷/热复用，图模式/MTP/PD，以及固定负载下的吞吐、延迟和峰值显存。2026-11-12 首轮版本对非 eager、MTP 和 KV transfer 显式拒绝，prefix caching 当时未做整网验证。后续已扩展原生 MTP3 和 ACLGraph，123 整网实际回放通过，prefix 并发严格一致性仍失败；最新边界与 GPQA 结果见 [后续验证](qwen36-prefix-mtp3-graph-gpqa.md)。
