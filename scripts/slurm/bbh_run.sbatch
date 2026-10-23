#!/bin/bash
# One bbh-run config entry per array task (untested template: no SLURM cluster was available).
#   sbatch --array=0-3 scripts/slurm/bbh_run.sbatch configs/m4_pi4_U1xU1.yaml
#   sbatch --array=0-3 --gres=gpu:1 scripts/slurm/bbh_run.sbatch configs/m4_theta0_dense_gpu.yaml --set device=cuda:0
# Adjust --time/--mem per D. For variational configs export BBH_MAX_HOURS (a bit below --time):
# the chain then stops cleanly and the same job can simply be resubmitted to resume.
#SBATCH --job-name=bbh
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%A_%a.out
set -euo pipefail
CONFIG=$1; shift
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EXTRA=${BBH_EXTRA:-ad}          # BBH_EXTRA=cuda on GPU nodes
LIMIT=()
[ -n "${BBH_MAX_HOURS:-}" ] && LIMIT=(--set max_hours="$BBH_MAX_HOURS")
uv run --extra "$EXTRA" bbh-run "$CONFIG" --index "${SLURM_ARRAY_TASK_ID:-0}" \
    --set threads="${SLURM_CPUS_PER_TASK:-1}" "${LIMIT[@]}" "$@"
