#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8501}"
streamlit run app.py --server.address=0.0.0.0 --server.port="$PORT"
