#!/usr/bin/env bash
# Direct shortcut to run the full Linux/WSL Parallel K-Means pipeline
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_project.sh" "$@"
