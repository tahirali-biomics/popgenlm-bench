#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
: "${PLANTCAD_PYTHON:?set PLANTCAD_PYTHON to the Python interpreter}"
: "${PLANTCAD_REPO_ROOT:?set PLANTCAD_REPO_ROOT to the cloned repository}"
: "${PLANTCAD_CHECKPOINT:?set PLANTCAD_CHECKPOINT to the local checkpoint}"
: "${PLANTCAD_FIXTURE:?set PLANTCAD_FIXTURE to the fixture TSV}"
: "${PLANTCAD_REFERENCE:?set PLANTCAD_REFERENCE to the reference FASTA}"
: "${PLANTCAD_OUTPUT:?set PLANTCAD_OUTPUT to the comparison output TSV}"
: "${PLANTCAD_UPSTREAM_ZERO_SHOT_SCORE:?set PLANTCAD_UPSTREAM_ZERO_SHOT_SCORE to zero_shot_score.py}"
exec "$PLANTCAD_PYTHON" -I -B -u "$PLANTCAD_REPO_ROOT/scripts/plantcad_compare.py" \\
  --checkpoint "$PLANTCAD_CHECKPOINT" --fixture "$PLANTCAD_FIXTURE" \\
  --reference "$PLANTCAD_REFERENCE" --output "$PLANTCAD_OUTPUT" \\
  --upstream-zero-shot-score "$PLANTCAD_UPSTREAM_ZERO_SHOT_SCORE"
