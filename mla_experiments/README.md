# MLA 实验脚本

启动 DeepSeek-V2-Lite-Chat，并通过在线请求采集 MLA 数据：

```bash
bash mla_experiments/collect_deepseek_v2_lite_chat.sh \
  /models/DeepSeek-V2-Lite-Chat
```

默认使用 TP=2，采集第 0、13、26 层。可以通过环境变量覆盖：

```bash
TP_SIZE=1 \
DUMP_LAYERS=0,6,13,20,26 \
DUMP_DIR=/workspace/mla-data/run-001 \
bash mla_experiments/collect_deepseek_v2_lite_chat.sh \
  /models/DeepSeek-V2-Lite-Chat
```

服务日志写入根目录的 `server.log`，在线请求响应写入 `response.json`。采集的
MLA 张量按照完整层名建立子目录，例如：

```text
/workspace/mla-data/run-001/
├── server.log
├── response.json
├── model.layers.0.self_attn.attn/
│   ├── prefill.rank0.pid1234.step0000.pt
│   └── decode.rank0.pid1234.step0000.pt
├── model.layers.13.self_attn.attn/
│   └── ...
└── model.layers.26.self_attn.attn/
    └── ...
```
