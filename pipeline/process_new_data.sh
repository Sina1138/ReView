#!/bin/bash
set -e  # Exit on error

# ============================================================
# ReView Data Processing Pipeline
# ============================================================
# Processes ICLR review data through the full scoring pipeline.
# Auto-detects available years from data/ directory by default,
# or accepts explicit year arguments.
#
# Usage (from project root):
#   ./pipeline/process_new_data.sh                     # Auto-detect all available years
#   ./pipeline/process_new_data.sh --year 2026         # Process a single year
#   ./pipeline/process_new_data.sh --year 2026 --force # Reprocess even if results exist
#   ./pipeline/process_new_data.sh --help              # Show all options
# ============================================================

# Resolve to project root (parent of pipeline/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================"
echo "ReView Data Processing Pipeline"
echo "========================================"

# Forward all arguments to the unified scoring pipeline
python pipeline/run_scoring.py "$@"
