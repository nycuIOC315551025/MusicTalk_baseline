#!/usr/bin/env bash
set -euo pipefail

MUSIC_ENV=/home/user/miniforge3/envs/musictalk-py312
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$MUSIC_ENV/bin/python" "$@"
