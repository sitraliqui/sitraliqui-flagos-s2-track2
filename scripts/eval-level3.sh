#!/usr/bin/env bash
evalscope eval \
  --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 \
  --ignore-errors
