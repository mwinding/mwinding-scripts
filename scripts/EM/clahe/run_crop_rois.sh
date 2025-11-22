#!/bin/bash

# Usage:
#   sbatch run_crop_rois.slurm CLAHE_DIR ROI_PATH OUTPUT_DIR
#
# Example:
#   sbatch run_crop_rois.slurm \
#     /camp/project/.../C2/M09_D23-24_10MHz_3nA_8x8x8_20V_CLAHE_slices \
#     /camp/project/.../C2/rois \
#     /camp/project/.../C2/M09_D23-24_10MHz_3nA_8x8x8_20V_ROI_crops

#SBATCH --job-name=CLAHE_ROI_CROP
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=a100          # change if you prefer a CPU partition you know works
#SBATCH --output=slurm-crop-%j.out
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

CLAHE_DIR="$1"
ROI_SOURCE="$2"
OUTPUT_DIR="$3"

if [ -z "$CLAHE_DIR" ] || [ -z "$ROI_SOURCE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch run_crop_rois.slurm CLAHE_DIR ROI_PATH OUTPUT_DIR"
    exit 1
fi

echo "--------------------------------------"
echo "Starting CLAHE ROI cropping job"
echo "CLAHE dir:   $CLAHE_DIR"
echo "ROI source:  $ROI_SOURCE"
echo "Output dir:  $OUTPUT_DIR"
echo "--------------------------------------"

ml purge
ml Anaconda3/2024.10
source /camp/apps/eb/software/Anaconda/conda.env.sh

# Use whichever env has roifile + tifffile + skimage installed
# e.g. tifftest, or your sleap env if you pip-installed roifile there
conda activate tifftest

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 crop_rois_from_clahe.py \
    -c "$CLAHE_DIR" \
    -r "$ROI_SOURCE" \
    -o "$OUTPUT_DIR"

STATUS=$?

echo
echo "--------------------------------------"
echo "ROI cropping job finished at $(date)"
echo "Exit status: $STATUS"
echo "--------------------------------------"
