#!/bin/bash

#SBATCH --job-name=CLAHE_slices
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --partition=ncpu          # <-- whatever partition you used that works
#SBATCH --output=slurm-%j.out
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

# --------- arguments ----------
# Usage: sbatch run_clahe_slices.sh input.tif output_dir
INPUT_TIF=$1
OUTPUT_DIR=$2

if [ -z "$INPUT_TIF" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch run_clahe_slices.sh input.tif output_dir"
    exit 1
fi

echo "------------------------------------------"
echo "Starting CLAHE slices job"
echo "SLURM job ID:   ${SLURM_JOB_ID}"
echo "CPUs:           ${SLURM_CPUS_PER_TASK}"
echo "Mem:            ${SLURM_MEM_PER_NODE}"
echo "Input:          ${INPUT_TIF}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "------------------------------------------"
echo

# --------- environment ---------
ml purge
ml Anaconda3/2024.10
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate /camp/lab/windingm/home/shared/conda-envs/sleap

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --------- run Python ----------
python3 clahe_process.py \
    -i "${INPUT_TIF}" \
    -d "${OUTPUT_DIR}"

STATUS=$?

echo
echo "------------------------------------------"
echo "CLAHE slices job finished at $(date)"
echo "Exit status: ${STATUS}"
echo "------------------------------------------"
