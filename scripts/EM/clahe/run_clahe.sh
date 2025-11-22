#!/bin/bash

#SBATCH --job-name=CLAHE_processing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --partition=ncpu
#SBATCH --output=slurm-%j.out
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

##############################################
# ENVIRONMENT SETUP
##############################################

ml purge
ml Anaconda3/2024.10
source /camp/apps/eb/software/Anaconda/conda.env.sh

# Use your shared conda env
conda activate /camp/lab/windingm/home/shared/conda-envs/sleap

# Make sure NumPy / OpenBLAS / friends use all allocated cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

##############################################
# ARGUMENTS
# Usage:
#   sbatch run_clahe.slurm input.tif output_dir tiff
#
#   $1 = input TIFF
#   $2 = output directory
#   $3 = output format (tiff | bdv)
##############################################

INPUT_TIF=$1
OUTPUT_DIR=$2
OUTPUT_FORMAT=$3   # tiff or bdv

if [ -z "$INPUT_TIF" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$OUTPUT_FORMAT" ]; then
    echo "Usage: sbatch run_clahe.slurm input.tif output_dir {tiff|bdv}"
    exit 1
fi

echo "------------------------------------------"
echo "Starting CLAHE job"
echo "SLURM job ID:   ${SLURM_JOB_ID}"
echo "CPUs:           ${SLURM_CPUS_PER_TASK}"
echo "Mem:            ${SLURM_MEM_PER_NODE}"
echo "Input:          ${INPUT_TIF}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Output format:  ${OUTPUT_FORMAT}"
echo "------------------------------------------"
echo

##############################################
# RUN PROCESSING
##############################################

python3 clahe_process.py \
    -i "${INPUT_TIF}" \
    -d "${OUTPUT_DIR}" \
    -o "${OUTPUT_FORMAT}"

STATUS=$?

echo
echo "------------------------------------------"
echo "CLAHE job finished at $(date)"
echo "Exit status: ${STATUS}"
echo "------------------------------------------"
