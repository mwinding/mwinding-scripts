#!/bin/bash

# Usage:
#   sbatch run_crop-em.sh INPUT_TIFF ROI_CSV OUTPUT_DIR [Z_START] [Z_END]
#
# Example:
#   sbatch run_crop-em.sh \
#     /camp/project/.../C2/M09_D23-24_10MHz_3nA_8x8x8_20V_CLAHE.tif \
#     /camp/project/.../C2/M09_D17_10MHz_3nA_8x8x8_20V_rois.csv \
#     /camp/project/.../C2/M09_D23-24_10MHz_3nA_8x8x8_20V_ROI_crops
#
# With Z-range:
#   sbatch run_crop-em.sh INPUT_TIFF ROI_CSV OUTPUT_DIR 0 999

#SBATCH --job-name=CROP_ROI_STACK
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=400G
#SBATCH --partition=ga100
#SBATCH --output=slurm-cropstack-%j.out
#SBATCH --mail-user=${USER}@crick.ac.uk
#SBATCH --mail-type=FAIL

INPUT_TIFF="$1"
ROI_CSV="$2"
OUTPUT_DIR="$3"
Z_START="$4"
Z_END="$5"

if [ -z "$INPUT_TIFF" ] || [ -z "$ROI_CSV" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch run_crop-em.sh INPUT_TIFF ROI_CSV OUTPUT_DIR [Z_START] [Z_END]"
    exit 1
fi

echo "--------------------------------------"
echo "Starting ROI CSV cropping job (TIFF stack)"
echo "Input TIFF:  $INPUT_TIFF"
echo "ROI CSV:     $ROI_CSV"
echo "Output dir:  $OUTPUT_DIR"
if [ -n "$Z_START" ] || [ -n "$Z_END" ]; then
    echo "Z range:     ${Z_START:-default} – ${Z_END:-default} (inclusive)"
fi
echo "--------------------------------------"

ml purge
ml Anaconda3/2024.10
source /camp/apps/eb/software/Anaconda/conda.env.sh

# Env must include pandas + tifffile (+ numpy)
conda activate /camp/lab/windingm/home/shared/conda-envs/sleap

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CMD=(python3 crop-em-volumes.py \
    -i "$INPUT_TIFF" \
    -r "$ROI_CSV" \
    -o "$OUTPUT_DIR")

# Optional Z args (only append if provided)
if [ -n "$Z_START" ]; then
    CMD+=(--z-start "$Z_START")
fi
if [ -n "$Z_END" ]; then
    CMD+=(--z-end "$Z_END")
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
STATUS=$?

echo
echo "--------------------------------------"
echo "ROI cropping job finished at $(date)"
echo "Exit status: $STATUS"
echo "--------------------------------------"

exit $STATUS
