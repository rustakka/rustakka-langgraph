#!/usr/bin/env bash
# After Rust edits: rebuild the cdylib and rerun pytest.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/dev-env.sh
maturin develop --release -m crates/py-bindings/pylanggraph/Cargo.toml
pytest python/tests -v "$@"
