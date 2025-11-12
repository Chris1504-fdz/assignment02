FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app
COPY . /app

# Python deps only
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "transformers>=4.44.0" \
      "datasets>=2.19.0" \
      "accelerate>=0.33.0" \
      "peft>=0.12.0" \
      "scikit-learn>=1.3.0" \
      "pandas>=2.1.0" \
      "sentencepiece>=0.1.99"

# Ensure demo uses the unit adapter path
RUN sed -i '1iimport os' quick_demo.py && \
    sed -i 's#^LORA_DIR *= *\".*\"#LORA_DIR = os.getenv("LORA_DIR", "./lora_output_unit/final_adapter")#' quick_demo.py

# Train tiny LoRA then run demo
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -euo pipefail' \
'echo "[1/2] Tiny LoRA unit fine-tune..."' \
'python unit_test_lora.py' \
'echo "[2/2] Quick demo using the unit adapter..."' \
'python quick_demo.py' \
> /usr/local/bin/entry.sh && chmod +x /usr/local/bin/entry.sh

ENTRYPOINT ["/usr/local/bin/entry.sh"]
