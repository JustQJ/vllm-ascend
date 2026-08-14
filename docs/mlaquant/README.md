# MLA 量化工作区

该目录统一保存 MLA 量化相关的设计、实验记录、数据分布分析和实现文档。

## 文档索引

- [数据采集说明](data-collection.md)：采集内容、配置方法、文件格式及限制。
- [计算流与 TP 聚合](data-flow-and-tp.md)：MLA 计算路径、采集点以及多卡数据保存策略。
- [模型兼容性](model-compatibility.md)：MLA 模型清单、Hugging Face 地址、参数规模、
  collector 覆盖范围和支持状态。
- [MLA 架构与数据流](mla_structure.md)：DeepSeek-V3.1 MLA 结构、权重与激活
  shape、prefill/decode 数据流。

后续新增的 MLA INT2/INT4 量化设计、校准结果、精度和性能实验文档也统一写入本目录。

可执行的数据采集入口见
[`mla_experiments/collect_deepseek_v2_lite_chat.sh`](../../mla_experiments/collect_deepseek_v2_lite_chat.sh)。
