#!/usr/bin/env bash
# Source this to activate a venv + maturin-friendly pyo3 config.
#   source scripts/dev-env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip maturin pytest pytest-asyncio
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f .cargo/pyo3-config.txt ] && [ -f .cargo/pyo3-config.txt.example ]; then
  cp .cargo/pyo3-config.txt.example .cargo/pyo3-config.txt
  echo "[dev-env] Copied .cargo/pyo3-config.txt — edit lib_dir/executable for your interpreter if needed."
fi
export PYO3_CONFIG_FILE="$ROOT/.cargo/pyo3-config.txt"
echo "[dev-env] PYO3_CONFIG_FILE=$PYO3_CONFIG_FILE"
echo "[dev-env] python: $(python --version)  -- venv: $VIRTUAL_ENV"
