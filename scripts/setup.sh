#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required but not found in PATH."
  echo "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "Then:    source ~/.zshrc  (or restart your shell)"
  exit 1
fi

echo "==> Installing Python dependencies (uv sync --all-extras)..."
uv sync --all-extras

if [ ! -f "${ROOT}/.env" ]; then
  if [ -f "${ROOT}/.env.example" ]; then
    cp "${ROOT}/.env.example" "${ROOT}/.env"
    echo "==> Created .env from .env.example"
  fi
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and set LEAPFLOW_LLM_API_KEY"
echo "  2. Run LeapFlow:"
echo "       ./scripts/run.sh --mock-host --prompt \"hello\""
echo "       # or: uv run leap --mock-host --prompt \"hello\""
echo "  3. (Optional) Build Swift OSHost:"
echo "       cd os_host && swift build -c debug"
