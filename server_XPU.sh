#!/usr/bin/env bash

set -euo pipefail

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
fi

MODEL_PATH=${MODEL_PATH:-/llama.cpp/models/Phi-3-mini-4k-instruct-fp16.gguf}
SERVER_BIN=${SERVER_BIN:-/llama.cpp/build/bin/llama-server}
PORT=${PORT:-8082}

echo "server_XPU.sh: Using SERVER_BIN=$SERVER_BIN"
echo "server_XPU.sh: Using MODEL_PATH=$MODEL_PATH"
echo "server_XPU.sh: Using PORT=$PORT"

if [ ! -x "$SERVER_BIN" ]; then
  echo "ERROR: llama-server binary not found or not executable: $SERVER_BIN" >&2
  exit 2
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: model file not found: $MODEL_PATH" >&2
  exit 3
fi

cd "$(dirname "$SERVER_BIN")"

exec "$SERVER_BIN" \
-m "$MODEL_PATH" \
--host 0.0.0.0 \
--port "$PORT" \
--gpu-layers -1 \
-sm none \
-mg 0 \
-c 4096