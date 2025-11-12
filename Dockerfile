# ===== Base: CUDA runtime with PyTorch =====
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# ===== Env: quiet logs, and put HF cache under /root/.cache/huggingface =====
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Optional: default LoRA dir (can be overridden at runtime)
ENV LORA_DIR=/app/lora_output/final_adapter

# ===== App files =====
WORKDIR /app
COPY . /app

# ===== Python deps only (no apt-get to avoid GPG key issues) =====
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "transformers>=4.44.0" \
      "datasets>=2.19.0" \
      "accelerate>=0.33.0" \
      "peft>=0.12.0" \
      "scikit-learn>=1.3.0" \
      "pandas>=2.1.0" \
      "sentencepiece>=0.1.99"

# ===== Make sure quick_demo.py actually uses the unit adapter by default =====
# This injects `import os` at the top (if missing) and replaces any hard-coded LORA_DIR
# with an env-aware line that defaults to ./lora_output_unit/final_adapter
RUN sed -i '1iimport os' quick_demo.py && \
    sed -i 's#^LORA_DIR *= *\".*\"#LORA_DIR = os.getenv("LORA_DIR", "/app/lora_output/final_adapter")#' quick_demo.py

# ===== Simple entrypoint: tiny LoRA train, then the quick demo =====
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -euo pipefail' \
'echo "[1/2] Tiny LoRA unit fine-tune..."' \
'python unit_test_lora.py' \
'echo "[2/2] Quick demo using the unit adapter..."' \
'python quick_demo.py' \
> /usr/local/bin/entry.sh && chmod +x /usr/local/bin/entry.sh

ENTRYPOINT ["/usr/local/bin/entry.sh"]
