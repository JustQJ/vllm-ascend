export ASCEND_SLOG_PRINT_TO_STDOUT=0 # 1/0 是否打屏
export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_HOST_LOG_FILE_NUM=1000
export VLLM_ENGINE_READY_TIMEOUT_S=1000
export ASCEND_PROCESS_LOG_PATH=/mnt/share/t00970481/tmp/plog

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# dsv31 模型路径作为参数
MODEL_PATH=$1

vllm serve $MODEL_PATH \
  --max_model_len 64000 \
  --max-num-batched-tokens 2048  \
  --served-model-name deepseek_v3 \
  --gpu-memory-utilization 0.94 \
  --async-scheduling \
  --max-num-seqs 32 \
  --port 9007 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --data-parallel-size 2 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-num-seqs 32 \
  --trust-remote-code \
  --speculative-config '{"num_speculative_tokens": 3,"method": "deepseek_mtp"}' \
  --additional_config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": true, "enable_mlapo": false, "enable_mla_unfused_preprocess": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
  # --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
  # --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
  # --enforce-eager \
  # --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
       
