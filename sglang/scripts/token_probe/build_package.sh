#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
VERSION="${VERSION:-0.0.0.dev20261125+tokenprobe.db90f584}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$VERSION"
export SGLANG_BUILD_RUST_EXTS=all
export PYTHONDONTWRITEBYTECODE=1
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
cd "$ROOT"
mkdir -p dist
"$PYTHON" -m build --wheel --no-isolation --outdir dist python
