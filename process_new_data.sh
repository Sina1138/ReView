#!/bin/bash
set -e  # Exit on error

# ============================================================
# ReView Data Processing Pipeline
# ============================================================
# Processes ICLR review data through the full scoring pipeline.
# Auto-detects available years from data/ directory by default,
# or accepts explicit year arguments.
#
# Usage:
#   ./process_new_data.sh                     # Auto-detect all available years
#   ./process_new_data.sh --year 2026         # Process a single year
#   ./process_new_data.sh --year 2026 --force # Reprocess even if results exist
#   ./process_new_data.sh --help              # Show all options
# ============================================================

echo "========================================"
echo "ReView Data Processing Pipeline"
echo "========================================"

# Forward all arguments to the unified scoring pipeline
python run_scoring.py "$@"
