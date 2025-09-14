#!/bin/bash
# Polygon S&P 500 stocks downloader runner script

set -euo pipefail

LOG_FILE="stocks_download_$(date +%Y%m%d_%H%M%S).log"

# Prefer venv python if available
if [ -x "$HOME/polygon_env/bin/python" ]; then
  PY="$HOME/polygon_env/bin/python"
else
  PY="$(command -v python3)"
fi

echo "Starting Polygon stocks downloader at $(date)"
echo "Log file: $LOG_FILE"

nohup "$PY" "$(dirname "$0")/polygon_stocks_downloader.py" > "$LOG_FILE" 2>&1 &
echo $! > download.pid

echo "Downloader started with PID $(cat download.pid)"
echo "Monitor with: tail -f $LOG_FILE"