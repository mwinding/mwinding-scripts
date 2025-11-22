#!/bin/bash

# Usage:
#   sbatch run_crop_rois_csv.sh CLAHE_DIR ROI_CSV OUTPUT_DIR
#
# Example:
#   sbatch run_crop_rois_csv.slurm \
#     /camp/project/.../C2/M09_D23-24_10MHz_3nA_8x8x8_20V_CLAHE_slices \
#     /camp/project/.../C2/M09_D17_10MHz_3nA_8x8x8_20V_rois.csv \
#     /camp/project/.../C2/M09_D23-24_10MHz_3nA_8x8x8_20V_ROI_crops

#SBATCH --job-name=CLAHE_ROI_CSV
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=a100              # or whatever CPU/GPU partition you know works
#SBATCH --output=slurm-cropcsv-%j.out
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

CLAHE_DIR="$1"
ROI_CSV="$2"
OUTPUT_DIR="$3"

if [ -z "$CLAHE_DIR" ] || [ -z "$ROI_CSV" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch run_crop_rois_csv.slurm CLAHE_DIR ROI_CSV OUTPUT_DIR"
    exit 1
fi

echo "--------------------------------------"
echo "Starting CLAHE ROI CSV cropping job"
echo "CLAHE dir:   $CLAHE_DIR"
echo "ROI CSV:     $ROI_CSV"
echo "Output dir:  $OUTPUT_DIR"
echo "--------------------------------------"

ml purge
ml Anaconda3/2024.10
source /camp/apps/eb/software/Anaconda/conda.env.sh

# Use an env that has pandas + tifffile installed.
# Either tifftest, or your sleap env if you've got those in there.
conda activate tifftest

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 crop_rois_from_clahe.py \
    -c "$CLAHE_DIR" \
    -r "$ROI_CSV" \
    -o "$OUTPUT_DIR"

STATUS=$?

echo
echo "--------------------------------------"
echo "ROI CSV cropping job finished at $(date)"
echo "Exit status: $STATUS"
echo "--------------------------------------"
