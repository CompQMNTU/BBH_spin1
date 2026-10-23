#!/bin/bash
# Run one bbh-run config entry on a shared workstation under a memory cap (no SLURM).
#   scripts/slurm/local_run.sh 48G configs/m4_pi4_U1xU1.yaml --index 2
# Thread pinning to 1; the cgroup cap kills the run (not the node) if it exceeds MEM. Logs to
# logs/<config>_<index>.log, detached from the terminal.
set -euo pipefail
MEM=$1; CONFIG=$2; shift 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
LOG="logs/$(basename "$CONFIG" .yaml)_$(echo "$@" | tr -c 'A-Za-z0-9=._-' '_').log"
nohup systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 \
    uv run --extra "${BBH_EXTRA:-ad}" bbh-run "$CONFIG" "$@" > "$LOG" 2>&1 &
echo "pid $! log $LOG"
