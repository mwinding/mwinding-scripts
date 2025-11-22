#!/bin/bash
#SBATCH --job-name=register-em
#SBATCH --output=register-em_%j.out
#SBATCH --error=register-em_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=ncpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --mail-user=$(whoami)@crick.ac.uk

# -----------------------------
# Argument parsing
# -----------------------------
if [ $# -lt 2 ]; then
    echo "Usage: sbatch run_jitter_rois.sh <ROI_DIR> <OUTPUT_DIR> [PATTERN]"
    echo "Example:"
    echo "  sbatch run_jitter_rois.sh /camp/.../cropped_rois /camp/.../aligned_rois"
    exit 1
fi

ROI_DIR=$1
OUTPUT_DIR=$2
PATTERN=${3:-"ROI_*.tif"}   # default if not given

echo "ROI directory:        $ROI_DIR"
echo "Output directory:     $OUTPUT_DIR"
echo "Glob pattern:         $PATTERN"
echo "-------------------------------------"

# -----------------------------
# Activate environment
# -----------------------------
source ~/.bashrc
conda activate fibsem   # change to your env name

# -----------------------------
# Run jitter correction
# -----------------------------
srun python jitter_correct_roi_stacks.py \
    --roi-dir "$ROI_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --pattern "$PATTERN" \
    --window-radius 3 \
    --blur-sigma 1.0 \
    --upsample-factor 10 \
    --max-abs-shift 20 \
    --sigma-z 2.0
