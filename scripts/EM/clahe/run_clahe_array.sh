#!/bin/bash

# Usage:
#   sbatch run_clahe_array.sh input.tif output_dir
#
# Example:
#   sbatch run_clahe_array.sh \
#       /camp/project/.../M09_D17_10MHz_3nA_8x8x8_20V.tif \
#       /camp/project/.../M09_D17_10MHz_3nA_8x8x8_20V_CLAHE_slices

#SBATCH --job-name=CLAHE_LAUNCH
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --partition=ncpu
#SBATCH --output=slurm-launch-%j.out
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

INPUT_TIF="$1"
OUTPUT_DIR="$2"

if [ -z "$INPUT_TIF" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch run_clahe_array.sh input.tif output_dir"
    exit 1
fi

echo "--------------------------------------"
echo "Launching CLAHE array job"
echo "Input TIF:    $INPUT_TIF"
echo "Output Dir:   $OUTPUT_DIR"
echo "--------------------------------------"

ml purge
ml Anaconda3/2024.10
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate /camp/lab/windingm/home/shared/conda-envs/sleap

# Submit the array job via the Python launcher
python launch_clahe_array.py \
    -i "$INPUT_TIF" \
    -d "$OUTPUT_DIR" \
    --slices-per-job 10

echo "--------------------------------------"
echo "Launch script finished at $(date)"
echo "--------------------------------------"
