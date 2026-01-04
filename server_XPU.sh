#!/usr/bin/env bash

source /opt/intel/oneapi/setvars.sh

exec $HOME/Git/llama.cpp/build/bin/llama-server \
-m ~/Git/llama.cpp/models/Phi-3-mini-4k-instruct-fp16.gguf \
--host 127.0.0.1 \
--port 8082 \
--gpu-layers 30 \
-sm none \
-mg 0 \
-c 4096