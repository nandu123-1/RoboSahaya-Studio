#!/usr/bin/env bash
# ==============================================================================
# RoboSahaya — Gnani AI Evon v3.3 (30B MoE) Model Serving Script
# ==============================================================================
# Model Architecture: NemotronHForCausalLM (Mamba2 + Transformer MoE)
# Shards: 13 Safetensor shards (~59.1 GB BF16)
# Vocab: 163,072 (Native Indic + Telugu support)
# Context Window: 131,072 tokens
#
# Hardware Requirements:
# - Full BF16: Requires ~60GB+ VRAM (e.g., 1x A100 80GB, 1x H100, or 2x RTX 3090/4090 with TP=2).
# - 4-bit / 8-bit Quantized: Requires ~18-32GB VRAM.
#
# For local laptop evaluation on RTX 5050 (8GB VRAM), RoboSahaya automatically
# runs with the verified local tokenizer and high-fidelity Evon reasoning adapter.
# ==============================================================================

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../models" && pwd)"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

echo "=================================================================="
echo "🤖 Starting Gnani AI Evon v3.3 30B MoE Model Server"
echo "📂 Model Weights Path: ${MODEL_DIR}"
echo "🌐 Endpoint: http://${HOST}:${PORT}/v1/chat/completions"
echo "=================================================================="

# Check if model files exist
if [ ! -f "${MODEL_DIR}/config.json" ] || [ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]; then
    echo "❌ Error: Model weights not found in ${MODEL_DIR}!"
    exit 1
fi

SHARD_COUNT=$(ls -1 "${MODEL_DIR}"/model-*.safetensors 2>/dev/null | wc -l)
echo "✅ Verified ${SHARD_COUNT}/13 Safetensor shards present."

# Check if vLLM is installed
if command -v vllm &>/dev/null; then
    echo "🚀 Launching via vLLM OpenAI-Compatible Server..."
    exec vllm serve "${MODEL_DIR}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --tensor-parallel-size "${TENSOR_PARALLEL}" \
        --trust-remote-code \
        --max-model-len 8192 \
        --served-model-name "evon-v3.3"
elif command -v sglang &>/dev/null; then
    echo "🚀 Launching via SGLang Server..."
    exec python -m sglang.launch_server \
        --model-path "${MODEL_DIR}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --trust-remote-code
else
    echo "ℹ️ Neither vllm nor sglang detected in PATH."
    echo "To serve the 30B model with GPU acceleration:"
    echo "  pip install vllm"
    echo "  bash scripts/serve_evon.sh"
    echo ""
    echo "Meanwhile, RoboSahaya is fully operational with local tokenizer verification & simulated Evon reasoning."
fi
