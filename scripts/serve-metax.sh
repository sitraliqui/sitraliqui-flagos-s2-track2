#!/usr/bin/env bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
