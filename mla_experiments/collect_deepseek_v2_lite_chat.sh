#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${1:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
DUMP_DIR="${DUMP_DIR:-/workspace/mla-data/deepseek-v2-lite-chat}"

export VLLM_ASCEND_MLA_DATA_DUMP_DIR="${DUMP_DIR}"
export VLLM_ASCEND_MLA_DATA_DUMP_LAYERS="${DUMP_LAYERS:-0,13,26}"
export VLLM_ASCEND_MLA_DATA_DUMP_MAX_TOKENS="${DUMP_MAX_TOKENS:-256}"
export VLLM_ASCEND_MLA_DATA_DUMP_MAX_STEPS="${DUMP_MAX_STEPS:-16}"
export VLLM_ASCEND_ENABLE_MLAPO=0

mkdir -p "${DUMP_DIR}"

vllm serve "${MODEL_PATH}" \
  --served-model-name deepseek-v2-lite-chat \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len 4096 \
  --enforce-eager \
  --port "${PORT}" \
  >"${DUMP_DIR}/server.log" 2>&1 &

SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT INT TERM

echo "Waiting for vLLM server, log: ${DUMP_DIR}/server.log"
until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM server exited unexpectedly."
    tail -n 100 "${DUMP_DIR}/server.log"
    exit 1
  fi
  sleep 3
done

curl -f "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v2-lite-chat",
    "messages": [{
      "role": "user",
      "content": "请详细解释 MLA 如何压缩 KV Cache，并比较 BF16 与 INT2 latent KV 的存储开销。"
    }],
    "temperature": 0,
    "max_tokens": 16
  }' \
  | tee "${DUMP_DIR}/response.json"

echo
echo "MLA data saved to ${DUMP_DIR}"
