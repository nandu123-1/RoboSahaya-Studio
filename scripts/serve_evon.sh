#!/usr/bin/env bash
# ==============================================================================
# Optional local OpenAI-compatible reasoning endpoint.
# ==============================================================================
# Point EVON_API_URL in the RoboSahaya environment at this endpoint and configure
# the same EVON_API_KEY. This script does not download or identify model weights.
# ==============================================================================

set -euo pipefail

MODEL_DIR="${EVON_MODEL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../models" && pwd)}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
if [ -z "${EVON_API_KEY:-}" ]; then
    echo "Set EVON_API_KEY before starting the local endpoint."
    exit 1
fi

echo "=================================================================="
echo "Starting configured local reasoning model server"
echo "📂 Model Weights Path: ${MODEL_DIR}"
echo "🌐 Endpoint: http://${HOST}:${PORT}/v1/chat/completions"
echo "=================================================================="

# Check if model files exist
if [ ! -f "${MODEL_DIR}/config.json" ] || ! compgen -G "${MODEL_DIR}/*.safetensors" >/dev/null; then
    echo "Model config or safetensors weights were not found in ${MODEL_DIR}."
    exit 1
fi

echo "Found model config and safetensors files in ${MODEL_DIR}."

# Check if vLLM is installed
if command -v vllm &>/dev/null; then
    echo "🚀 Launching via vLLM OpenAI-Compatible Server..."
    exec vllm serve "${MODEL_DIR}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --api-key "${EVON_API_KEY}" \
        --tensor-parallel-size "${TENSOR_PARALLEL}" \
        --max-model-len "${EVON_MAX_MODEL_LEN:-8192}" \
        --served-model-name "${EVON_MODEL_ID:-local-reasoning-model}"
else
    echo "vLLM was not found in PATH. Install a compatible vLLM build to serve this model:"
    echo "  pip install vllm"
    echo "  bash scripts/serve_evon.sh"
    exit 1
fi
